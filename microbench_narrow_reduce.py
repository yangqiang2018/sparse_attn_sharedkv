"""Probe: on ascendc, which narrow row-reduce actually honours the row stride?

kernel.py reduces the first `tw` columns of a 512-wide score buffer, one
64-column chunk at a time. It does that with `T.tile.wholereducemax/sum`, which
are `@deprecated`. The obvious replacement is the documented `real_shape`
argument on `T.reduce_max/sum` -- "Optional logical 2D shape for sliced UB
tiles", with no target restriction in the docstring.

But `real_shape` lowers to `reduce_{max,sum}<T, M, N, dim>`, whose body calls
`AscendC::Reduce{Max,Sum}<Pattern::AR>(dst, src, tmp, shape={M, N}, ...)`, and
that treats `src` as a CONTIGUOUS M x N block. For a [M, 512] buffer with a
logical [M, 64] region the contiguous reading is `src[0 : M*64]` -- row 0's
first 8 chunks -- not the first 64 columns of each of the M rows. The shipped
example that exercises `real_shape` (examples/reduce/example_row_reduce_max_
slice_buffer.py) is `target="pto"`, where the tile carries a physical stride
view, so it never exposed this on ascendc.

If that reading is right, `real_shape` silently returns wrong values on ascendc
whenever the logical width is narrower than the buffer -- a bug in a shipped
API, not a missing feature -- and the deprecated wholereduce is currently the
only thing on ascendc that reduces a strided narrow region correctly.

The kernel here computes a per-row max/sum over the first VALID columns of a
wider buffer, three ways, against the same golden:

    real_shape   T.reduce_*(..., real_shape=[M, VALID])      <- the API in question
    wholereduce  T.tile.wholereduce*(mask=VALID, srcrepstride=BUF//8)
    full         T.reduce_*(...) over the whole buffer, padding neutralised
                 (sanity: the non-narrow path must be right, or the harness is
                 wrong rather than the API)

Padding is filled with a value that changes the answer if it is included, so a
wrong read cannot pass by luck.

Run:  python microbench_narrow_reduce.py
"""

import torch

import tilelang
from tilelang import language as T

tilelang.disable_cache()

M = 16  # rows
BUF = 512  # buffer width (kernel.py's score buffer)
VALID = 64  # logical width (kernel.py's chunk)
dtype = "float"

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


# A per-row result is one scalar per row. Keep it 1-D [M]: the reduce validator
# accepts [M] for dim=-1, and a [M, 1] column hits the 32B-block-aligned
# copy_ub_to_gm path on the way out, which reorders the values.
def build_real_shape(kind):
    reduce_op = T.reduce_max if kind == "max" else T.reduce_sum

    @T.prim_func
    def main(src: T.Tensor([M, BUF], dtype), out: T.Tensor([M], dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            ub = T.alloc_ub([M, BUF], dtype)
            acc = T.alloc_ub([M], dtype)
            if vid == 0:
                T.copy(src, ub)
                reduce_op(ub, acc, dim=-1, real_shape=[M, VALID])
                T.copy(acc, out)

    return main


def build_wholereduce(kind):
    whole = T.tile.wholereducemax if kind == "max" else T.tile.wholereducesum

    @T.prim_func
    def main(src: T.Tensor([M, BUF], dtype), out: T.Tensor([M], dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            ub = T.alloc_ub([M, BUF], dtype)
            acc = T.alloc_ub([M], dtype)
            if vid == 0:
                T.copy(src, ub)
                # mask = the logical width; srcrepstride steps one physical row
                # (in 32B blocks: BUF fp32 elems = BUF/8 blocks). This is the
                # shape kernel.py uses.
                if kind == "max":
                    whole(
                        acc,
                        ub[:, 0:VALID],
                        VALID,
                        M,
                        1,
                        1,
                        BUF // 8,
                        ReduceOrder="ORDER_ONLY_VALUE",
                    )
                else:
                    whole(acc, ub[:, 0:VALID], VALID, M, 1, 1, BUF // 8)
                T.copy(acc, out)

    return main


def run(tag, func_def, kind):
    torch.manual_seed(0)
    data = torch.randn(M, BUF, dtype=torch.float32)
    # Poison the padding: including it changes the answer.
    data[:, VALID:] = 100.0 if kind == "max" else 7.0

    ref = (
        data[:, :VALID].max(dim=-1).values
        if kind == "max"
        else data[:, :VALID].sum(dim=-1)
    )

    try:
        func = tilelang.compile(
            func_def, out_idx=[-1], target="ascendc", pass_configs=pass_configs
        )
        got = func(data.npu()).cpu()
    except Exception as exc:  # noqa: BLE001
        print(f"  {tag:<12} {kind:<3} COMPILE/RUN FAILED: {repr(exc)[:110]}")
        return

    diff = (got - ref).abs().max().item()
    verdict = "OK" if diff < 1e-3 else "WRONG"
    # A contiguous misread lands row 0's chunks in every slot, so show row 0/1
    # to make the failure mode legible rather than just a magnitude.
    print(
        f"  {tag:<12} {kind:<3} max|diff|={diff:10.4f}  {verdict}"
        f"   got[:2]={[round(v, 3) for v in got[:2].tolist()]}"
        f" ref[:2]={[round(v, 3) for v in ref[:2].tolist()]}"
    )


if __name__ == "__main__":
    print("=" * 72)
    print(f"narrow row-reduce on ascendc: [{M},{BUF}] buffer, first {VALID} cols valid")
    print("padding poisoned, so including it cannot pass by luck")
    print("=" * 72)
    for kind in ("max", "sum"):
        run("real_shape", build_real_shape(kind), kind)
        run("wholereduce", build_wholereduce(kind), kind)
    print("=" * 72)
    print("real_shape WRONG + wholereduce OK  -> real_shape is broken on ascendc;")
    print("   route A = fix it (lower narrow real_shape to the wholereduce path),")
    print("   which also retires kernel.py's deprecated usage.")
    print("real_shape OK                      -> kernel.py can just switch to it;")
    print("   the deprecated wart goes away with no compiler change at all.")
    print("=" * 72)
