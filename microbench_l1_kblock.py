"""Probe: is an L1 K-block column slice the front-end equivalent of the
gemm_v0_fixp template's hand-computed kL0 offset?

Decomposing gemm_v0_fixp (the last private-primitive dependency in kernel.py)
turns the template's internal kL0 walk

    copy_l1_to_l0a<T, M, K>      (l0a, A[kL0Idx * M * 128], M,     kSize)
    copy_l1_to_l0b<T, N, K, true>(l0b, B[kL0Idx * N * 128], kSize, n_actual)

into a front-end column slice

    T.copy(a_l1[:, kk*128:(kk+1)*128], a_l0[pp])
    T.copy(b_l1[:, kk*128:(kk+1)*128], b_l0[pp], transpose=True,
           real_k=128, real_n=n)

The template's offset is the zN PHYSICAL one -- L1 is [cols/16][rows][16], so a
K-block strides by rows*128. The codegen computes the pointer as
OffsetOf(indices).back(), which is ROW MAJOR unless the buffer carries a layout:
AscendCopy::InferLayout returns an empty map, so the only thing that can put a
fractal offset there is an explicit T.annotate_layout(make_zn_layout(...)) --
which testing/python/language/test_tilelang_ascend_language_mma.py does and
kernel.py does not.

Nothing in either repo slices an L1 buffer's COLUMNS, so that half is untested.
But kernel.py does slice L1 ROWS at a runtime offset -- the front-end paged load
fills a ring slot one page at a time -- and that is known-good today WITHOUT the
annotation. So the annotation is not obviously free: it rewrites the row offset
as well as the column one.

Hence one kernel exercising BOTH, run under both settings:
  * segmented row fill   -- b_l1 written as two half-row copies (paged-load shape)
  * K-block column slice -- two hand-walked kL0 tiles accumulating into one L0C

A wrong row offset corrupts B's second half -> C's columns N/2.. are wrong.
A wrong column offset corrupts the second kL0 tile -> all of C is wrong.
Either way C != A @ B^T, and the two are told apart by WHICH columns fail.

Also the first caller of real_n (added for this decomposition, no caller yet)
and of an L0C->GM fixpipe with a runtime column extent.

Run:  python microbench_l1_kblock.py
"""

import torch

import tilelang
from tilelang import language as T
from tilelang.intrinsics import make_zn_layout

tilelang.disable_cache()

M, N, K = 64, 128, 256
KB = 128  # kL0 tile = the template's kL0Size
dtype, accum_dtype = "float16", "float"

pass_configs = {
    # Cube-only probe: keep it on one core so the hand-driven L0 sequence is
    # exactly what the cube sees (same reason microbench_paged_load turns it off).
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def _build_annotated():
    @tilelang.jit(out_idx=[-1], target="ascendc", pass_configs=pass_configs)
    def _k():
        @T.prim_func
        def main(
            A: T.Tensor([M, K], dtype),
            B: T.Tensor([N, K], dtype),  # K^T layout: the gemm is A @ B^T
            nact: T.Tensor([1], "int32"),
            C: T.Tensor([M, N], accum_dtype),
        ):
            with T.Kernel(1, is_npu=True) as (cid, _):
                a_l1 = T.alloc_L1([M, K], dtype)
                b_l1 = T.alloc_L1([N, K], dtype)
                T.annotate_layout(
                    {a_l1: make_zn_layout(a_l1), b_l1: make_zn_layout(b_l1)}
                )
                a_l0 = T.alloc_L0A([2, M, KB], dtype)
                b_l0 = T.alloc_L0B([2, N, KB], dtype)
                c_l0 = T.alloc_L0C([1, M, N], accum_dtype)
                with T.Scope("C"):
                    n_act = nact[0]
                    T.barrier_all()
                    T.copy(A, a_l1)
                    # Segmented row fill = the paged load's shape (two partial
                    # copies filling one buffer, dstIsSlice on both).
                    T.copy(B[0 : N // 2, :], b_l1[0 : N // 2, :])
                    T.copy(B[N // 2 : N, :], b_l1[N // 2 : N, :])
                    T.barrier_all()
                    # kL0 tile 0: column offset 0, correct under either rule.
                    T.copy(a_l1[:, 0:KB], a_l0[0, :, :])
                    T.copy(
                        b_l1[:, 0:KB],
                        b_l0[0, :, :],
                        transpose=True,
                        real_k=KB,
                        real_n=n_act,
                    )
                    T.barrier_all()
                    T.mma(
                        a_l0[0, :, :],
                        b_l0[0, :, :],
                        c_l0[0, :, :],
                        init=True,
                        k_actual=KB,
                        n_actual=n_act,
                        unit_flag=0b10,
                    )
                    T.barrier_all()
                    # kL0 tile 1: THE probe -- a non-zero K-block column offset.
                    T.copy(a_l1[:, KB : 2 * KB], a_l0[1, :, :])
                    T.copy(
                        b_l1[:, KB : 2 * KB],
                        b_l0[1, :, :],
                        transpose=True,
                        real_k=KB,
                        real_n=n_act,
                    )
                    T.barrier_all()
                    T.mma(
                        a_l0[1, :, :],
                        b_l0[1, :, :],
                        c_l0[0, :, :],
                        init=False,
                        k_actual=KB,
                        n_actual=n_act,
                        unit_flag=0b11,
                    )
                    T.barrier_all()
                    T.copy(c_l0[0, :, :], C[:, 0:n_act], unit_flag=0b11)
                    T.barrier_all()

        return main

    return _k()


def _build_plain():
    @tilelang.jit(out_idx=[-1], target="ascendc", pass_configs=pass_configs)
    def _k():
        @T.prim_func
        def main(
            A: T.Tensor([M, K], dtype),
            B: T.Tensor([N, K], dtype),
            nact: T.Tensor([1], "int32"),
            C: T.Tensor([M, N], accum_dtype),
        ):
            with T.Kernel(1, is_npu=True) as (cid, _):
                a_l1 = T.alloc_L1([M, K], dtype)
                b_l1 = T.alloc_L1([N, K], dtype)
                a_l0 = T.alloc_L0A([2, M, KB], dtype)
                b_l0 = T.alloc_L0B([2, N, KB], dtype)
                c_l0 = T.alloc_L0C([1, M, N], accum_dtype)
                with T.Scope("C"):
                    n_act = nact[0]
                    T.barrier_all()
                    T.copy(A, a_l1)
                    T.copy(B[0 : N // 2, :], b_l1[0 : N // 2, :])
                    T.copy(B[N // 2 : N, :], b_l1[N // 2 : N, :])
                    T.barrier_all()
                    T.copy(a_l1[:, 0:KB], a_l0[0, :, :])
                    T.copy(
                        b_l1[:, 0:KB],
                        b_l0[0, :, :],
                        transpose=True,
                        real_k=KB,
                        real_n=n_act,
                    )
                    T.barrier_all()
                    T.mma(
                        a_l0[0, :, :],
                        b_l0[0, :, :],
                        c_l0[0, :, :],
                        init=True,
                        k_actual=KB,
                        n_actual=n_act,
                        unit_flag=0b10,
                    )
                    T.barrier_all()
                    T.copy(a_l1[:, KB : 2 * KB], a_l0[1, :, :])
                    T.copy(
                        b_l1[:, KB : 2 * KB],
                        b_l0[1, :, :],
                        transpose=True,
                        real_k=KB,
                        real_n=n_act,
                    )
                    T.barrier_all()
                    T.mma(
                        a_l0[1, :, :],
                        b_l0[1, :, :],
                        c_l0[0, :, :],
                        init=False,
                        k_actual=KB,
                        n_actual=n_act,
                        unit_flag=0b11,
                    )
                    T.barrier_all()
                    T.copy(c_l0[0, :, :], C[:, 0:n_act], unit_flag=0b11)
                    T.barrier_all()

        return main

    return _k()


def _report(tag, func, n_act, dump):
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.float16) * 0.1
    b = torch.randn(N, K, dtype=torch.float16) * 0.1
    nact = torch.tensor([n_act], dtype=torch.int32)
    ref = (a.float() @ b.float().T)[:, :n_act]

    if dump:
        src = func.get_kernel_source()
        path = f"/tmp/l1_kblock_{tag}.cpp"
        with open(path, "w") as f:
            f.write(src)
        print(f"\n--- {tag}: wrote {path} ---")
        for ln in src.splitlines():
            if "copy_l1_to_l0" in ln or "copy_gm_to_l1" in ln or "copy_l0c_to_gm" in ln:
                print("   ", ln.strip()[:170])

    out = func(a.npu(), b.npu(), nact.npu())
    got = out.cpu()[:, :n_act]
    diff = (got - ref).abs()
    # Column N/2.. isolates the segmented ROW fill; every column feels a wrong
    # K-block COLUMN offset.
    lo = diff[:, : min(N // 2, n_act)].max().item()
    hi = diff[:, N // 2 : n_act].max().item() if n_act > N // 2 else float("nan")
    print(
        f"{tag:>10} n_act={n_act:3d}  max|diff| all={diff.max().item():9.5f} "
        f"cols[0:{N // 2}]={lo:9.5f} cols[{N // 2}:]={hi:9.5f}"
    )
    return diff.max().item()


if __name__ == "__main__":
    print("=" * 78)
    print("L1 K-block column-slice offset probe (zN annotate on/off)")
    print("bf16-free fp32 accum; noise floor should be ~1e-3, a wrong offset is O(1)")
    print("=" * 78)
    results = {}
    for tag, builder in (("plain", _build_plain), ("annotated", _build_annotated)):
        func = builder()
        for i, n_act in enumerate((N, 96)):
            results[(tag, n_act)] = _report(tag, func, n_act, dump=(i == 0))
    print("\n" + "=" * 78)
    for (tag, n_act), d in results.items():
        verdict = "OK" if d < 0.05 else "WRONG"
        print(f"  {tag:>10} n_act={n_act:3d} -> {verdict} (max|diff|={d:.5f})")
    print("=" * 78)
