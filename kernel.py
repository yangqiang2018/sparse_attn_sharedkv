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

SCHEDULING (faithful to Ascend C ``ProcessBalance`` + ``PreloadPipeline``):
the grid is ``core_num`` physical cores (not one block per (b,s)); each
core walks its share of the ``block_num = batch*max_seq`` tasks in a
software-pipelined loop. Cube and vector are split MANUALLY with
``T.Scope("C")`` / ``T.Scope("V")`` and chained by cross-core flags so the
cube and vector overlap across tasks:

  cube   iter j:  QK(task j) -> ws_s          ||  PV(task j-1) -> ws_o
  vector iter j:  softmax(task j) -> ws_p     ||  output(task j-1)

In steady state ``cube PV(j-1) ∥ vector softmax(j)`` and
``cube QK(j+1) ∥ vector output(j-1)`` run concurrently -- the 3-stage
``SAS_PRELOAD_TASK_CACHE_SIZE``/``PRELOAD_NUM`` preload pipeline of the
reference. The cube↔vector workspaces (ws_s/ws_p/ws_o) and the carried
softmax state (denom/m_i) are double-buffered by ``j % 2`` so task j+1's
cube does not clobber task j's data while the vector still reads it.
The reference's metadata-driven per-core balance becomes a static even
split here (exact for the uniform fast test; cost-based balance TODO).
"""

import os

import tilelang
from tilelang import language as T

from metadata import SAS_META_SIZE

# KV window tile width. ori_win_left (<=127) + 1 <= 128, so the whole
# sliding window fits in one tile. Also sizes api.py's dummy cmp-indices.
DEFAULT_BLOCK_I = 128
# Default number of AI Cube cores (Ascend 910B). The TileLang grid is one
# block per cube core; each core pipelines its share of the tasks.
DEFAULT_CORE_NUM = 24


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
        core_num=core_num,
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
    core_num,
):
    N1 = n_heads  # query heads (= 64)
    N2 = 1  # kv heads
    D = head_dim  # head dim (= 512)
    G = N1 // N2  # GQA group = block_M rows handled per task (= 64)
    BI = DEFAULT_BLOCK_I  # kv window tile (= 128)
    VEC_NUM = 2  # 2 vector cores per cube core
    G2 = G // VEC_NUM  # rows per vector core (= 32)
    BLK = 8  # FP32 elems per 32B block (Brcb fan-out width for row_expand)
    # uint8 shared scratch for SoftmaxFlashV2 (mirrors Ascend C softmaxTmpUb /
    # tmpBuff1 = 32KB; ample for the [G2, BI] softmax block).
    SOFTMAX_TMP_BYTES = 32768

    # ---- DIAGNOSTIC bitmask (0 = FAITHFUL) ----------------------------------
    # The QK-via-gemm_v0_fixp hang was LOCALISED: DBG_MODE=8 (single K-tile, QK =
    # 1 mma + fixpipe like PV) STILL hung -> NOT the multi-K accumulate; the
    # transpose_B fixpipe path itself hung. ROOT CAUSE (faithfulness gap): the
    # fixpipe copied the full nTile columns, but QK's mma only wrote n_actual
    # (window width < nTile); the reference keeps mmadParams.n == fixParams.nSize
    # (== nL1SizeAlign), so the unitFlag fixpipe waited for cL0 columns no mma
    # marked ready -> HANG. Fixed in the compiler: the fixpipe's nSize now = the
    # mma's n (n_actual for transpose_B). Faithful path restored: DBG_MODE=0.
    #   1 = PipeBarrier before each mma (diagnostic; doesn't help).
    #   2 = 0b00 intermediate unitFlag (workaround; abandoned, also hangs).
    #   8 = single K-tile localiser (used to pin the transpose_B fixpipe).
    DBG_MODE = 0
    # DEBUG_SERIAL=True keeps the iteration-boundary barrier_all (current parity
    # structure). The 3-slot KV ring / QP ring / zero-barrier come AFTER the gemm
    # is proven; this flag stays True until then.
    DEBUG_SERIAL = True

    accum_dtype = "float"
    idx_dtype = "int32"
    # Cast rounding modes, faithful to Ascend C: P->KV_T uses CAST_ROUND
    # (swa_block_vector.h:433); output->OUT_T uses CAST_RINT for bf16, else
    # CAST_ROUND (swa_block_vector.h:618-622). CAST_NONE (truncate) would drift.
    out_cast_mode = "CAST_RINT" if dtype == "bfloat16" else "CAST_ROUND"

    # Task space: one task per (batch, query position); mirrors the Ascend C
    # bN2 x gS1 space (kvHeadNum==1, mBaseSize==gSize==G so each gS1 tile is
    # one token's G heads). Statically split across the core_num cube cores.
    block_num = batch * max_seq
    n_iter = (block_num + core_num - 1) // core_num  # tasks per core (ceil)

    # Cross-core pipeline event ids (cube<->vector handshake, reused every
    # iteration as 1-deep counting handshakes, faithful to the reference's
    # fixed syncC1V1 / syncV1C2 / syncC2V2 event ids).
    EV_QK = 0  # cube -> vector: QK result (ws_s) ready
    EV_P = 1  # vector -> cube: softmax P (ws_p) ready
    EV_PV = 2  # cube -> vector: PV result (ws_o) ready

    # Intra-cube MTE2->MTE1 pipe-flag event ids for the 2-buffer KV split
    # (kv_l1[0]=QK, kv_l1[1]=PV). Separate MTE2_MTE1 id namespace; must avoid the
    # gemm template's internal {0,1} (L0AB_EVENT). So {2,3}. These let PV's
    # copy_pa (MTE2 into kv_l1[1]) overlap QK's gemm (M reading kv_l1[0]) instead
    # of a full barrier_all -- faithful to the reference's per-slot KV flags.
    KV_QK_EV = 2  # kv_l1[0] (QK) MTE2 done
    KV_PV_EV = 3  # kv_l1[1] (PV) MTE2 done

    q_shape = [total_tokens, N1, D]
    ori_kv_shape = [ori_block_num, ori_block_size, N2, D]
    cmp_kv_shape = [cmp_block_num, cmp_block_size, N2, D]
    cmp_idx_shape = [total_tokens, N2, DEFAULT_BLOCK_I]

    @tilelang.jit(out_idx=[11, 12], workspace_idx=[13, 14, 15])
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
            # cube<->vector workspaces, per core, double-buffered (dim 1 = j%2):
            workspace_s: T.Tensor([core_num, 2, G, BI], accum_dtype),  # 13 QKᵀ
            workspace_p: T.Tensor([core_num, 2, G, BI], dtype),  # 14 softmax P
            workspace_o: T.Tensor([core_num, 2, G, D], accum_dtype),  # 15 PV
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, vid):
                # ---- Allocations at kernel scope (never inside a conditional).
                # Cube: q_l1, a 2-buffer KV split (kv_l1[0]=QK, kv_l1[1]=PV;
                # kv reloaded for PV, faithful to the reference's DataCopyPA in
                # both Mm1 and Mm2), p_l1. The two KV buffers let PV's copy_pa
                # (MTE2) overlap QK's gemm (M) -- the reference's per-slot KV
                # ring degenerated to depth-2 (only QK(j)/PV(j-1) ever live).
                # 2*[BI,D] + [G,D] + [G,BI] = 256KB+64KB+16KB = 336KB < 512KB L1.
                q_l1 = T.alloc_L1([G, D], dtype)
                kv_l1 = T.alloc_L1([2, BI, D], dtype)
                p_l1 = T.alloc_L1([G, BI], dtype)
                # STEP-1 DIAGNOSTIC: separate cL0 for QK and PV, to isolate the
                # one new variable -- QK's multi-K (K=512, 4 tiles) unitFlag 0b10
                # accumulate inside gemm_v0_fixp (the cube hang). acc_s_l0c = QK's
                # 1-slot score L0C; acc_o_l0c = PV's 2-slot [2,G,BI] ping-pong
                # (008, proven). The faithful SHARED cL0TensorPingPong (one buffer,
                # continuous cl0_base across QK+PV) is layered on AFTER the multi-K
                # gemm is proven -- it adds a QK->PV cross-call slot-reuse hazard
                # that would confound this diagnostic.
                acc_s_l0c = T.alloc_L0C([G, BI], accum_dtype)
                acc_o_l0c = T.alloc_L0C([2, G, BI], accum_dtype)
                # Vector UB scratch.
                s_ub = T.alloc_ub([G2, BI], accum_dtype)
                p_half = T.alloc_ub([G2, BI], dtype)
                o_ub = T.alloc_ub([G2, D], accum_dtype)
                o_half = T.alloc_ub([G2, D], dtype)
                sink_ub = T.alloc_ub([G2, 1], accum_dtype)
                lse_ub = T.alloc_ub([G2, 1], accum_dtype)
                # SoftmaxFlashV2 state: in_sum seed (1.0), the unused flash
                # rescale output (single-block softmax produces but ignores
                # expmax), and the uint8 shared scratch (Ascend C softmaxTmpUb).
                ones_ub = T.alloc_ub([G2, 1], accum_dtype)
                expmax_ub = T.alloc_ub([G2, 1], accum_dtype)
                softmax_tmp = T.alloc_ub([SOFTMAX_TMP_BYTES], "uint8")
                # Contiguous [G2, win_align] score buffer: softmax_flash_v2 compacts
                # s_ub[:, 0:win_align] here so the library runs in its win_align
                # range (sized for the max win_align = BI).
                softmax_cmp = T.alloc_ub([G2, BI], accum_dtype)
                # [G2,8] Brcb scratch for the output row-broadcast div (faithful
                # to Ascend C RowDivs -- no [G2,D] denom broadcast buffer).
                brcb_d = T.alloc_ub([G2, BLK], accum_dtype)
                # Carried softmax state, double-buffered by task parity so the
                # output stage of task j-1 reads the denom/max from its own
                # softmax (computed one iteration earlier).
                m_i = T.alloc_ub([2, G2, 1], accum_dtype)
                denom = T.alloc_ub([2, G2, 1], accum_dtype)

                # ============================ CUBE ============================
                # iter j: QK(task j) -> ws_s[j%2] ; PV(task j-1) -> ws_o[(j-1)%2]
                with T.Scope("C"):
                    for j in T.serial(n_iter + 1):
                        # ---- QK stage for task j ----
                        if j < n_iter:
                            pid = j * core_num + cid
                            buf = j % 2
                            if pid < block_num:
                                b = T.cast(pid // max_seq, "int32")
                                s = T.cast(pid % max_seq, "int32")
                                act_q = act_q_lens[b]
                                if s < act_q:
                                    act_kv = seqused_kv[b]
                                    tok = q_prefix[b] + s
                                    s_global = act_kv - act_q + s
                                    ori_left = T.max(s_global - ori_win_left, 0)
                                    win = s_global + 1 - ori_left
                                    T.copy(Q[tok, :, :], q_l1)
                                    T.copy_pa(
                                        kv_l1[0, :, :],
                                        ori_kv,
                                        ori_block_table,
                                        ori_block_size,
                                        N2,
                                        D,
                                        ori_block_size * N2 * D,
                                        ori_table_len,
                                        D,
                                        win,
                                        BI,
                                        b,
                                        0,
                                        ori_left,
                                        0,
                                    )
                                    # kv_l1[0] loaded (and q_l1, earlier on the
                                    # in-order MTE2 queue) -> tag MTE2 done so the
                                    # gemm's MTE1 L1->L0 load waits on this point-to
                                    # -point flag instead of a full barrier_all,
                                    # letting PV's later copy_pa(kv_l1[1]) MTE2
                                    # overlap this QK gemm's M pipe.
                                    T.set_flag("mte2", "mte1", KV_QK_EV)
                                    # QK = Q @ Kᵀ over the window; N rounds up to 16 to
                                    # match Ascend C ComputeMm1 (nL1SizeAlign =
                                    # SASAlign(window, 16)); copy_pa loads `win` KV
                                    # rows. acc_s_l0c[0:win_align] is the score -- the
                                    # [win:win_align] tail is Q@unloaded-KV, but the
                                    # softmax compacts/processes win_align and the
                                    # reduce/PV use winm, so it is excluded exactly as
                                    # in the reference. [win_align:BI] stays
                                    # uninitialised; the softmax never reads it.
                                    win_align = (win + 15) // 16 * 16
                                    T.wait_flag("mte2", "mte1", KV_QK_EV)
                                    # Faithful QK = ComputeMm1: gemm_v0_fixp K-
                                    # accumulates D=512 over 4 kL0 tiles into
                                    # acc_s_l0c, unitFlag 0b10x3/0b11, then fuses the
                                    # fixpipe straight to workspace_s (no resident
                                    # L0C + separate copy). n_actual=win_align = the
                                    # window width (the score's real columns).
                                    # dbg_mode=0 = faithful. The hang was the fixpipe
                                    # copying nTile cols while the mma wrote only
                                    # n_actual (win_align); fixpipe nSize now = mma n
                                    # (compiler fix), so unitFlag pairs correctly.
                                    T.gemm_v0_fixp(
                                        q_l1,
                                        kv_l1[0, :, :],
                                        acc_s_l0c,
                                        workspace_s[cid, buf, :, :],
                                        k_actual=D,
                                        transpose_B=True,
                                        init=True,
                                        n_actual=win_align,
                                        dbg_mode=DBG_MODE,
                                    )
                            T.set_cross_flag("FIX", EV_QK)
                        # ---- PV stage for task j-1 ----
                        if j >= 1:
                            T.wait_cross_flag(EV_P)
                            # wait_cross_flag(EV_P) already orders the cube after the
                            # vector's ws_p (MTE3) write -- no barrier_all needed.
                            pidm = (j - 1) * core_num + cid
                            bufm = (j - 1) % 2
                            if pidm < block_num:
                                bm = T.cast(pidm // max_seq, "int32")
                                sm = T.cast(pidm % max_seq, "int32")
                                if sm < act_q_lens[bm]:
                                    act_kvm = seqused_kv[bm]
                                    s_globalm = act_kvm - act_q_lens[bm] + sm
                                    ori_leftm = T.max(s_globalm - ori_win_left, 0)
                                    winm = s_globalm + 1 - ori_leftm
                                    T.copy(workspace_p[cid, bufm, :, :], p_l1)
                                    # Reload the task j-1 KV window (faithful to
                                    # the reference reloading V in Mm2).
                                    T.copy_pa(
                                        kv_l1[1, :, :],
                                        ori_kv,
                                        ori_block_table,
                                        ori_block_size,
                                        N2,
                                        D,
                                        ori_block_size * N2 * D,
                                        ori_table_len,
                                        D,
                                        winm,
                                        BI,
                                        bm,
                                        0,
                                        ori_leftm,
                                        0,
                                    )
                                    # kv_l1[1] loaded (and p_l1, earlier on the
                                    # in-order MTE2 queue) -> point-to-point flag
                                    # instead of a full barrier_all.
                                    T.set_flag("mte2", "mte1", KV_PV_EV)
                                    # PV = P @ V, fixpiped per N-tile straight to
                                    # workspace_o (L0C holds one [G,BI] tile).
                                    # k_actual=winm: contract only the real window
                                    # rows (faithful to Ascend C ComputeMm2's
                                    # kSize=window). The pad rows kv_l1[winm:BI]
                                    # are uninitialised L1 -- summing them as
                                    # 0(masked P)*NaN(garbage V) would give NaN;
                                    # contracting only winm excludes them.
                                    T.wait_flag("mte2", "mte1", KV_PV_EV)
                                    # PV = ComputeMm2 (008, proven): gemm_v0_fixp
                                    # fixpipes O[G,D] per N-tile from its own 2-slot
                                    # acc_o_l0c ping-pong. Single K tile (k_actual=
                                    # winm<=128, unitFlag 0b11) -- no multi-K 0b10,
                                    # so it is the unchanged proven baseline (no
                                    # dbg_mode) against which the QK change tests.
                                    T.gemm_v0_fixp(
                                        p_l1,
                                        kv_l1[1, :, :],
                                        acc_o_l0c,
                                        workspace_o[cid, bufm, :, :],
                                        k_actual=winm,
                                        init=True,
                                    )
                                    # Iteration-boundary full drain (kept while
                                    # DEBUG_SERIAL): protects every cross-iteration
                                    # hazard -- q_l1/p_l1 WAR, acc_s WAR, the acc_o
                                    # cL0 ping-pong cross-iteration reuse. The 3-slot
                                    # KV ring / QP ring / fine-grained flags that let
                                    # this be removed come AFTER the gemm is proven.
                                    if DEBUG_SERIAL:
                                        T.barrier_all()
                            T.set_cross_flag("FIX", EV_PV)

                # =========================== VECTOR ===========================
                # iter j: softmax(task j) -> ws_p[j%2] ; output(task j-1)
                with T.Scope("V"):
                    # in_sum seed for SoftmaxFlashV2 = 1.0 (Ascend C R0); filled
                    # once, read each task as the flash running-sum initial value.
                    T.tile.fill(ones_ub, T.float32(1.0))
                    T.barrier_all()
                    for j in T.serial(n_iter + 1):
                        # ---- softmax stage for task j ----
                        if j < n_iter:
                            T.wait_cross_flag(EV_QK)
                            T.barrier_all()
                            pid = j * core_num + cid
                            buf = j % 2
                            if pid < block_num:
                                b = T.cast(pid // max_seq, "int32")
                                s = T.cast(pid % max_seq, "int32")
                                act_q = act_q_lens[b]
                                if s < act_q:
                                    act_kv = seqused_kv[b]
                                    s_global = act_kv - act_q + s
                                    ori_left = T.max(s_global - ori_win_left, 0)
                                    # winm = window length (= Ascend C
                                    # actualSingleProcessSInnerSize); win_align =
                                    # winm rounded up to 16 (= actualSingleProcess
                                    # SInnerSizeAlign / the QK mma N). softmax
                                    # processes win_align columns (columnCount) and
                                    # reduces only winm (actualColumnCount).
                                    winm = s_global + 1 - ori_left
                                    win_align = (winm + 15) // 16 * 16
                                    T.copy(
                                        workspace_s[
                                            cid, buf, vid * G2 : (vid + 1) * G2, :
                                        ],
                                        s_ub,
                                    )
                                    T.barrier_all()
                                    T.tile.mul(s_ub, s_ub, softmax_scale)
                                    T.copy(sinks[vid * G2 : (vid + 1) * G2], sink_ub)
                                    T.barrier_all()
                                    # Faithful Ascend C SoftmaxFlashV2
                                    # (swa_block_vector.h SoftmaxFlashV2Compute):
                                    # sink-seeded single-pass softmax. The primitive
                                    # compacts s_ub[:, 0:win_align] into softmax_cmp,
                                    # runs the library with SoftMaxShapeInfo {G2,
                                    # win_align, G2, winm} (columnCount=win_align in
                                    # its designed range, like Ascend C's win_align-
                                    # strided mmResUb; actualColumnCount=winm reduces
                                    # only the window), then scatters P back -- the
                                    # uninitialised s_ub[win_align:BI] is never read.
                                    # in_max = per-row sink, in_sum = 1.0;
                                    # out_max/out_sum -> carried m_i/denom.
                                    T.tile.softmax_flash_v2(
                                        s_ub,
                                        denom[buf, :, :],
                                        m_i[buf, :, :],
                                        expmax_ub,
                                        s_ub,
                                        ones_ub,
                                        sink_ub,
                                        softmax_tmp,
                                        softmax_cmp,
                                        win_align,
                                        winm,
                                    )
                                    T.barrier_all()
                                    T.tile.cast(p_half, s_ub, "CAST_ROUND", G2 * BI)
                                    T.copy(
                                        p_half,
                                        workspace_p[
                                            cid, buf, vid * G2 : (vid + 1) * G2, :
                                        ],
                                    )
                                    T.barrier_all()
                            T.set_cross_flag("MTE3", EV_P)
                        # ---- output stage for task j-1 ----
                        if j >= 1:
                            T.wait_cross_flag(EV_PV)
                            T.barrier_all()
                            pidm = (j - 1) * core_num + cid
                            bufm = (j - 1) % 2
                            if pidm < block_num:
                                bm = T.cast(pidm // max_seq, "int32")
                                sm = T.cast(pidm % max_seq, "int32")
                                if sm < act_q_lens[bm]:
                                    tokm = q_prefix[bm] + sm
                                    T.copy(
                                        workspace_o[
                                            cid, bufm, vid * G2 : (vid + 1) * G2, :
                                        ],
                                        o_ub,
                                    )
                                    T.barrier_all()
                                    # o = o / denom via row broadcast (Brcb + Div,
                                    # src1RepStride=1) over the full D -- no
                                    # [G2,D] denom buffer, faithful to Ascend C
                                    # RowDivs; processes headDim in one pass.
                                    T.tile.row_expand_div(
                                        o_ub, o_ub, denom[bufm, :, :], brcb_d
                                    )
                                    T.tile.cast(o_half, o_ub, out_cast_mode, G2 * D)
                                    T.barrier_all()
                                    T.copy(
                                        o_half,
                                        Output[tokm, vid * G2 : (vid + 1) * G2, :],
                                    )
                                    # lse = max + ln(sum) (T.tile.ln; scalar
                                    # tir.log is unlowerable on Ascend).
                                    T.tile.ln(lse_ub, denom[bufm, :, :])
                                    T.tile.add(lse_ub, lse_ub, m_i[bufm, :, :])
                                    T.barrier_all()
                                    T.copy(lse_ub, LSE[tokm, vid * G2 : (vid + 1) * G2])
                                    T.barrier_all()

        return sparse_attn_sharedkv_swa

    func = kernel()
    # Debug hook: set SAS_DUMP_SRC=1 to dump the generated Ascend C (for
    # inspecting buffer sizes / sync flags / indexing when localising a runtime
    # fault without a local NPU). No effect unless the env var is set.
    if os.environ.get("SAS_DUMP_SRC"):
        try:
            src = func.get_kernel_source()
            with open("/tmp/swa_gen.cpp", "w") as fh:
                fh.write(src)
            print(f"[SAS_DUMP_SRC] wrote {len(src)} chars to /tmp/swa_gen.cpp")
        except Exception as exc:  # noqa: BLE001
            print(f"[SAS_DUMP_SRC] get_kernel_source failed: {exc!r}")
    return func
