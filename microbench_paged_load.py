"""Probe: can the *front end* express copy_pa's paged sliding-window KV gather
with plain `T.copy`, so kernel.py's 10 `T.copy_pa` calls can drop the private
copy_pa primitive (JZY line: compose in the front end from low-level primitives)?

copy_pa loads `COPY_ROWS` contiguous rows of a paged KV cache starting at a
RUNTIME row `S2_START`, walking one *page* at a time: look up the physical block
in the block table, copy that page's row-run in one DataCopy, advance. The whole
value over a per-row gather is the per-PAGE batched copy.

The front-end reconstruction is a compile-time-bounded page loop with a runtime
guard, doing per page:

    logical = cur // BLOCK_SIZE
    phys    = BT[logical]                       # runtime block-table lookup
    rem     = cur %  BLOCK_SIZE                 # runtime start within the page
    run     = min(BLOCK_SIZE - rem, remaining)  # runtime row count
    T.copy(KV[phys, rem:rem+run, :], out[done:done+run, :])   # <-- THE test

The one thing not yet proven anywhere is a `T.copy` whose row slice has BOTH a
runtime start (`rem`) AND a runtime length (`run`) -- 014 (#1337) only fixed the
runtime-END case `0:tw_a` (start fixed at 0). If this compiles and matches the
torch gather, copy_pa is expressible in the front end with no new primitive.

Runs the load to UB (easy to export); the L1/Nd2Nz destination copy_pa actually
uses is the same gather logic with dst scope L1 -- validated later in kernel.py.
"""

import torch

import tilelang
from tilelang import language as T

tilelang.disable_cache()

# --- shapes: BLOCK_SIZE small + window mid-page so it spans multiple pages ----
D = 128  # head dim
BLOCK_SIZE = 16  # rows per KV-cache page
NUM_PHYS = 8  # physical blocks in the cache
NUM_LOGICAL = 8  # logical pages (block_table maps logical -> physical, shuffled)
COPY_ROWS = 40  # window length (spans 3 pages of 16)
S2_START = 5  # runtime window start (mid first page)
COPY_ROWS_ALIGN = 48  # padded dst rows
MAX_PAGES = 4  # compile-time upper bound on pages the window can touch

dtype = "float16"

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def build():
    @tilelang.jit(out_idx=[2], target="ascendc", pass_configs=pass_configs)
    def _k():
        @T.prim_func
        def main(
            KV: T.Tensor([NUM_PHYS, BLOCK_SIZE, D], dtype),
            BT: T.Tensor([NUM_LOGICAL], "int32"),
            Out: T.Tensor([COPY_ROWS_ALIGN, D], dtype),
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                kv_ub = T.alloc_ub((COPY_ROWS_ALIGN, D), dtype)
                if vid == 0:
                    T.tile.fill(kv_ub, T.float16(0))
                    done = T.alloc_var("int32")
                    cur = T.alloc_var("int32")
                    done = 0
                    cur = S2_START
                    # compile-time bounded page walk; runtime guard stops early
                    for _pg in range(MAX_PAGES):
                        if done < COPY_ROWS:
                            logical = cur // BLOCK_SIZE
                            phys = BT[logical]
                            rem = cur % BLOCK_SIZE
                            run = T.min(BLOCK_SIZE - rem, COPY_ROWS - done)
                            # runtime start (rem) AND runtime length (run) row slice
                            T.copy(
                                KV[phys, rem : rem + run, :],
                                kv_ub[done : done + run, :],
                            )
                            done = done + run
                            cur = cur + run
                    T.copy(kv_ub, Out)

        return main

    return _k()


def golden(kv, bt):
    out = torch.zeros((COPY_ROWS_ALIGN, D), dtype=kv.dtype)
    for i in range(COPY_ROWS):
        s = S2_START + i
        logical = s // BLOCK_SIZE
        phys = int(bt[logical].item())
        rem = s % BLOCK_SIZE
        out[i] = kv[phys, rem]
    return out


def main():
    torch.manual_seed(0)
    # data on CPU, then move to NPU (avoid CPU-generator vs npu-default clash)
    kv = torch.randn(NUM_PHYS, BLOCK_SIZE, D, dtype=torch.float16)
    perm = torch.randperm(NUM_LOGICAL)  # shuffled logical -> physical
    bt = perm[:NUM_LOGICAL].to(torch.int32)
    ref = golden(kv, bt)

    func = build()
    print("init OK")
    src = func.get_kernel_source()
    with open("/tmp/paged_load_cg.cpp", "w") as f:
        f.write(src)
    print("=== copy_gm_to_ub lines (runtime-slice emission) ===")
    for i, ln in enumerate(src.splitlines()):
        if "copy_gm_to_ub" in ln:
            print(f"{i:5d}: {ln.strip()[:140]}")
    print("=== end (full dump at /tmp/paged_load_cg.cpp) ===")

    kv_npu = kv.npu()
    bt_npu = bt.npu()
    torch.npu.synchronize()
    out = func(kv_npu, bt_npu)
    torch.npu.synchronize()

    got = out[:COPY_ROWS].cpu()
    exp = ref[:COPY_ROWS]
    err = (got.float() - exp.float()).abs().max().item()
    ok = err < 1e-3
    print(
        f"paged sliding-window gather: max|diff|={err:.4g}  {'OK' if ok else '!! MISMATCH'}"
    )
    if not ok:
        # per-row diff to see which page/boundary breaks
        for i in range(COPY_ROWS):
            d = (got[i].float() - exp[i].float()).abs().max().item()
            if d > 1e-3:
                print(f"  row {i:3d} (global {S2_START + i:3d}) diff={d:.4g}")
    print("done")


if __name__ == "__main__":
    main()
