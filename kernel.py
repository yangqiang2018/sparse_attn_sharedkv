"""TileLang kernel for the SparseAttnSharedKV op.

This module is the kernel half of the op whose host wrapper lives in
:mod:`api`. ``api.sparse_attn_sharedkv`` imports
:func:`build_sparse_attn_sharedkv`, :data:`DEFAULT_BLOCK_I` and
:data:`DEFAULT_CORE_NUM` from here and calls the returned ``@tilelang.jit``
function with 11 positional tensors (``out_idx=[11, 12]`` ⇒ returns
``(Output, LSE)``; workspaces are auto-allocated via ``workspace_idx``).

Scope of THIS file so far: the **SWA** (sliding-window attention,
``scenario == 1``) path only -- a faithful TileLang port of the Ascend C
``SparseAttnSharedkvSwa`` kernel (op_kernel/arch32/sparse_attn_sharedkv_swa_*).
SCFA/CFA (``scenario`` 2/3) are not implemented yet and raise.

Faithful mapping to the Ascend C SWA (per query row ``s`` of batch ``b``):

  s_global = act_kv - act_q + s                  # causal kv position
  ori_left  = max(s_global - ori_win_left, 0)    # window left  (inclusive)
  ori_right = s_global + ori_win_right + 1        # window right (exclusive)
  S   = (Q @ Kᵀ) * softmax_scale       over kv ∈ [ori_left, ori_right)
  # online softmax seeded with the per-head attention sink:
  m   = max(sink_h, rowmax(S));  p = exp(S - m);  p_sink = exp(sink_h - m)
  den = sum(p) + p_sink
  O   = (p @ V) / den            # K and V are the SAME ori_kv (shared KV)
  lse = m + log(den)

Because ``ori_win_right == 0`` and ``ori_win_left == 127`` (both asserted
upstream in :mod:`api`/:mod:`golden`), the attended window is at most
``ori_win_left + 1 == 128`` keys, so a single KV tile (``BI = 128``)
covers it -- the FlashAttention online loop degenerates to one pass.

NOTE (v1, to revisit on-device): grid is one core per (batch, query-pos),
hardware-scheduled, rather than the Ascend C metadata-driven per-core
load balancing. The ``metadata`` argument is accepted (contract parity)
but not used to drive scheduling yet. Numerics are unaffected.
"""

import tilelang
from tilelang import language as T

from metadata import SAS_META_SIZE

# KV window tile width. ori_win_left (<=127) + 1 <= 128, so the whole
# sliding window fits in one tile. Also sizes api.py's dummy cmp-indices.
DEFAULT_BLOCK_I = 128
# Default number of AI Cube cores (Ascend 910B). Only forwarded to the
# metadata scheduler; the TileLang grid uses one core per query position.
DEFAULT_CORE_NUM = 24

# Cube/vector data passing + auto sync are handled by the compiler.
_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def build_sparse_attn_sharedkv(
    *,
    batch: int,
    max_seq: int,
    total_tokens: int,
    ori_block_num: int,
    ori_block_size: int,
    ori_table_len: int,
    cmp_block_num: int,
    cmp_block_size: int,
    cmp_table_len: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    topk_cmp: int,
    cmp_ratio: int,
    scenario: int,
    ori_win_left: int,
    softmax_scale: float,
    dtype: str,
    core_num: int,
):
    """Build (JIT) the SparseAttnSharedKV kernel for one compile-time config.

    Returns a callable taking the 11 input tensors and returning
    ``(Output, LSE)``. See module docstring for the SWA contract.
    """
    if scenario != 1:
        raise NotImplementedError(
            f"only SWA (scenario=1) is implemented; got scenario={scenario} "
            "(SCFA/CFA pending)"
        )
    if n_kv_heads != 1 or n_heads != 64 or head_dim != 512:
        raise ValueError(
            f"SWA kernel assumes N1=64, N2=1, D=512 (got N1={n_heads}, "
            f"N2={n_kv_heads}, D={head_dim})"
        )
    if ori_win_left + 1 > DEFAULT_BLOCK_I:
        raise ValueError(
            f"ori_win_left={ori_win_left} exceeds single-tile window "
            f"(BI={DEFAULT_BLOCK_I}); multi-tile SWA not implemented yet"
        )

    return _build_swa(
        batch=batch,
        max_seq=max_seq,
        total_tokens=total_tokens,
        ori_block_num=ori_block_num,
        ori_block_size=ori_block_size,
        ori_table_len=ori_table_len,
        cmp_block_num=cmp_block_num,
        cmp_block_size=cmp_block_size,
        cmp_table_len=cmp_table_len,
        n_heads=n_heads,
        head_dim=head_dim,
        ori_win_left=ori_win_left,
        softmax_scale=float(softmax_scale),
        dtype=dtype,
    )


def _build_swa(
    *,
    batch,
    max_seq,
    total_tokens,
    ori_block_num,
    ori_block_size,
    ori_table_len,
    cmp_block_num,
    cmp_block_size,
    cmp_table_len,
    n_heads,
    head_dim,
    ori_win_left,
    softmax_scale,
    dtype,
):
    N1 = n_heads  # query heads (= 64)
    N2 = 1  # kv heads
    D = head_dim  # head dim (= 512)
    G = N1 // N2  # GQA group = block_M rows handled per core (= 64)
    BI = DEFAULT_BLOCK_I  # kv window tile (= 128)
    VEC_NUM = 2  # 2 vector cores per cube core
    G2 = G // VEC_NUM  # rows per vector core (= 32)

    accum_dtype = "float"
    idx_dtype = "int32"
    # Cast rounding modes, faithful to Ascend C: P->KV_T uses CAST_ROUND
    # (swa_block_vector.h:433); output->OUT_T uses CAST_RINT for bf16, else
    # CAST_ROUND (swa_block_vector.h:618-622). CAST_NONE (truncate) would drift.
    out_cast_mode = "CAST_RINT" if dtype == "bfloat16" else "CAST_ROUND"

    # One core per (batch, query position); mirrors the Ascend C bN2 x gS1
    # task space (kvHeadNum==1). Every (b, s) is computed unconditionally.
    block_num = batch * max_seq

    q_shape = [total_tokens, N1, D]
    ori_kv_shape = [ori_block_num, ori_block_size, N2, D]
    cmp_kv_shape = [cmp_block_num, cmp_block_size, N2, D]
    cmp_idx_shape = [total_tokens, N2, DEFAULT_BLOCK_I]

    @tilelang.jit(
        out_idx=[11, 12], workspace_idx=[13, 14, 15, 16], pass_configs=_PASS_CONFIGS
    )
    def kernel():
        @T.prim_func
        def sparse_attn_sharedkv_swa(
            Q: T.Tensor(q_shape, dtype),  # 0
            ori_kv: T.Tensor(ori_kv_shape, dtype),  # 1  (shared K & V)
            ori_block_table: T.Tensor([batch, ori_table_len], idx_dtype),  # 2
            cmp_kv: T.Tensor(cmp_kv_shape, dtype),  # 3  (unused, SWA)
            cmp_block_table: T.Tensor([batch, cmp_table_len], idx_dtype),  # 4 unused
            cmp_indices: T.Tensor(cmp_idx_shape, idx_dtype),  # 5  (unused, SWA)
            q_prefix: T.Tensor([batch], idx_dtype),  # 6  flat-token base
            act_q_lens: T.Tensor([batch], idx_dtype),  # 7  per-batch q len
            seqused_kv: T.Tensor([batch], idx_dtype),  # 8  per-batch kv len
            sinks: T.Tensor([N1], accum_dtype),  # 9  per-head sink
            metadata: T.Tensor([SAS_META_SIZE], idx_dtype),  # 10 (unused, v1)
            Output: T.Tensor(q_shape, dtype),  # 11 out
            LSE: T.Tensor([total_tokens, N1], accum_dtype),  # 12 out
            ws_kv: T.Tensor([block_num, BI, D], dtype),  # 13 gathered window
            ws_s: T.Tensor([block_num, G, BI], accum_dtype),  # 14 QKᵀ result
            ws_p: T.Tensor([block_num, G, BI], dtype),  # 15 softmax P
            ws_o: T.Tensor([block_num, G, D], accum_dtype),  # 16 PV result
        ):
            with T.Kernel(block_num, is_npu=True) as (cid, vid):
                # ---- Allocations: unconditional (shapes are compile-time).
                # The example kernels allocate at kernel scope, never inside a
                # conditional -- an `if` must not gate buffer allocation. ----
                q_l1 = T.alloc_L1([G, D], dtype)
                kv_l1 = T.alloc_L1([BI, D], dtype)
                p_l1 = T.alloc_L1([G, BI], dtype)
                acc_s_l0c = T.alloc_L0C([G, BI], accum_dtype)
                acc_o_l0c = T.alloc_L0C([G, D], accum_dtype)
                kv_ub = T.alloc_ub([D], dtype)
                s_ub = T.alloc_ub([G2, BI], accum_dtype)
                p_half = T.alloc_ub([G2, BI], dtype)
                o_ub = T.alloc_ub([G2, D], accum_dtype)
                o_half = T.alloc_ub([G2, D], dtype)
                m_i = T.alloc_ub([G2, 1], accum_dtype)  # running max
                m_raw = T.alloc_ub([G2, 1], accum_dtype)
                sink_ub = T.alloc_ub([G2, 1], accum_dtype)  # per-head sink
                psink = T.alloc_ub([G2, 1], accum_dtype)
                denom = T.alloc_ub([G2, 1], accum_dtype)
                m_2d = T.alloc_ub([G2, BI], accum_dtype)
                den_2d = T.alloc_ub([G2, D], accum_dtype)
                lse_ub = T.alloc_ub([G2, 1], accum_dtype)
                mask_ub = T.alloc_ub([BI // 8], "uint8")  # window bitmask (1 bit/col)
                pos_ub = T.alloc_ub([BI], accum_dtype)  # per-column kv positions

                # cid fixes (b, s) for this block. Compute unconditionally for
                # every (b, s) -- like the reference kernels. NOTE: wrapping the
                # body in a data-dependent `if s < act_q` made the CV-split leak
                # the thread var (v_thread) into the cube codegen; the examples
                # never gate cube+vector work behind such an `if`. The TND fast
                # test has no padding (s < act_q always), so this is exact;
                # BSND/TND padding handling is a separate (later) concern.
                b = cid // max_seq
                s = cid % max_seq
                act_q = act_q_lens[b]
                act_kv = seqused_kv[b]
                tok = q_prefix[b] + s

                s_global = act_kv - act_q + s
                ori_left = T.max(s_global - ori_win_left, 0)
                ori_right = s_global + 1  # ori_win_right == 0

                # ===== Load Q (all G heads of this token) → L1 =====
                T.copy(Q[tok, :, :], q_l1)

                # ===== Gather the sliding-window KV (paged) → workspace.
                # Each vid gathers its half of the BI rows. Rows past the window
                # clamp to the last valid position (ori_right-1); their columns
                # are masked to -inf below, so the duplicate KV they load never
                # contributes. Unconditional gather (mirrors the reference).
                for r in T.serial(BI // VEC_NUM):
                    row = vid * (BI // VEC_NUM) + r
                    pos = T.min(ori_left + row, ori_right - 1)
                    page = ori_block_table[b, pos // ori_block_size]
                    brow = pos % ori_block_size
                    T.copy(ori_kv[page, brow, 0, :], kv_ub)
                    T.copy(kv_ub, ws_kv[cid, row, :])
                T.copy(ws_kv[cid, :, :], kv_l1)

                # ===== Cube: S = Q @ Kᵀ  (K = ori_kv window) =====
                T.gemm_v0(q_l1, kv_l1, acc_s_l0c, transpose_B=True, init=True)
                T.copy(acc_s_l0c, ws_s[cid, :, :])

                # ===== Vector: scale + window mask + softmax (1 tile) =====
                T.copy(ws_s[cid, vid * G2 : (vid + 1) * G2, :], s_ub)
                T.tile.mul(s_ub, s_ub, softmax_scale)
                # Window mask via the idiomatic compare+select (VSEL), the same
                # pattern as example_sparse_flash_attn_mask.py -- NOT
                # if_then_else inside T.Parallel (a Select there fails to
                # vectorize and leaks a v_thread predicate into codegen).
                # Column j -> kv position ori_left+j; in-window iff <= s_global.
                for j in T.serial(BI):
                    pos_ub[j] = T.cast(ori_left + j, accum_dtype)
                T.tile.compare(mask_ub, pos_ub, T.float32(s_global), "LE")
                for i in T.serial(G2):
                    T.tile.select(
                        s_ub[i, :],
                        mask_ub,
                        s_ub[i, :],
                        -T.infinity(accum_dtype),
                        "VSEL_TENSOR_SCALAR_MODE",
                    )
                # load this vid's per-head sink logit
                T.copy(sinks[vid * G2 : (vid + 1) * G2], sink_ub)
                # running max includes the sink (sink = virtual key logit)
                T.reduce_max(s_ub, m_raw, dim=-1)
                T.tile.max(m_i, m_raw, sink_ub)
                # p = exp(S - m)
                T.tile.broadcast(m_2d, m_i)
                T.tile.sub(s_ub, s_ub, m_2d)
                T.tile.exp(s_ub, s_ub)
                # denom = sum(p) + exp(sink - m)
                T.reduce_sum(s_ub, denom, dim=-1)
                T.tile.sub(psink, sink_ub, m_i)
                T.tile.exp(psink, psink)
                T.tile.add(denom, denom, psink)
                # P → half → workspace for the PV matmul (round, matching Ascend C)
                T.tile.cast(p_half, s_ub, "CAST_ROUND", G2 * BI)
                T.copy(p_half, ws_p[cid, vid * G2 : (vid + 1) * G2, :])

                # ===== Cube: O = P @ V  (V = same ori_kv window) =====
                T.copy(ws_p[cid, :, :], p_l1)
                T.gemm_v0(p_l1, kv_l1, acc_o_l0c, init=True)
                T.copy(acc_o_l0c, ws_o[cid, :, :])

                # ===== Vector: normalize, write Output + LSE =====
                T.copy(ws_o[cid, vid * G2 : (vid + 1) * G2, :], o_ub)
                T.tile.broadcast(den_2d, denom)
                T.tile.div(o_ub, o_ub, den_2d)
                T.tile.cast(o_half, o_ub, out_cast_mode, G2 * D)
                T.copy(o_half, Output[tok, vid * G2 : (vid + 1) * G2, :])
                # lse = m + log(denom), element-wise via the TIR log. Use
                # T.serial, NOT T.Parallel: this (the last T.Parallel in the
                # kernel) is what failed to vectorize and left a surviving
                # parallel loop with an unbound v_thread thread-predicate that
                # leaked into the cube codegen. The working paged_flash_attn
                # example uses zero T.Parallel -- only T.tile.* and T.serial.
                for i in T.serial(G2):
                    lse_ub[i, 0] = m_i[i, 0] + T.log(denom[i, 0])
                T.copy(lse_ub, LSE[tok, vid * G2 : (vid + 1) * G2])

        return sparse_attn_sharedkv_swa

    return kernel()
