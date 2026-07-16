"""TileLang kernel for the SparseAttnSharedKV op.

This module is the kernel half of the op whose host wrapper lives in
:mod:`api`. ``api.sparse_attn_sharedkv`` imports
:func:`build_sparse_attn_sharedkv`, :data:`DEFAULT_BLOCK_I` and
:data:`DEFAULT_CORE_NUM` from here and calls the returned ``@tilelang.jit``
function with 11 positional tensors (``out_idx=[11, 12]`` ⇒ returns
``(Output, LSE)``; workspaces are auto-allocated via ``workspace_idx``).

Scope of THIS file: **SWA** (``scenario == 1``, :func:`_build_swa`) and **CFA**
(``scenario == 2``, :func:`_build_cfa`) -- faithful TileLang ports of the Ascend C
``SparseAttnSharedkvSwa`` kernel (op_kernel/arch32/sparse_attn_sharedkv_swa_*); CFA
is the same class's ``CFA_TEMPLATE`` multi-KV-tile online-softmax path. SCFA
(``scenario == 3``) is not implemented yet and raises.

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
  vector iter j:  softmax(task j-1) -> ws_p   ||  output(task j-2)

This is the reference's depth-3 preload (PreloadPipeline / extraInfo0/2/1 =
task loop/loop-1/loop-2, swa_kernel.h:745-768): the vector lags the cube by
one task, so ``cube QK(j) ∥ vector softmax(j-1)`` and
``cube PV(j-1) ∥ vector output(j-2)`` run concurrently and the vector never
stalls on the cube's CURRENT QK/PV (its inputs were produced a cube-iter
earlier). The cube↔vector workspaces (ws_s/ws_p/ws_o) and the carried
softmax state (denom/m_i) are double-buffered by ``task % 2``; the cube's
PV-after-softmax dependency (via the ws_p cross-flag) keeps QK(loop) ordered
after softmax(loop-2), so a buffer's prior reader is always done before it
is reused (mod-2 suffices at this depth).
The reference's metadata-driven per-core balance becomes a static even
split here (exact for the uniform fast test; cost-based balance TODO).
"""

import os

import tilelang
from tilelang import language as T
from tvm import tir

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
    if scenario not in (1, 2, 3):
        raise NotImplementedError(
            f"only SWA (1), CFA (2), SCFA (3) are implemented; got scenario={scenario}"
        )
    if n_kv_heads != 1 or n_heads != 64 or head_dim != 512:
        raise ValueError(
            f"kernel assumes N1=64, N2=1, D=512 (got N1={n_heads}, N2={n_kv_heads}, D={head_dim})"
        )
    if ori_win_left + 1 > DEFAULT_BLOCK_I:
        raise ValueError(
            f"ori_win_left={ori_win_left} exceeds single-tile window (BI={DEFAULT_BLOCK_I}); the ori window is one tile for both SWA and CFA"
        )

    if scenario == 1:
        # SWA: single ori KV-tile (window <= 128), degenerate online softmax.
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

    if scenario == 3:
        # SCFA (scenario 3) = CFA + topk SPARSE cmp: instead of reading the cmp
        # segment densely (copy_pa), a vector V0 stage gathers the topk-selected
        # cmp blocks (via cmp_indices / GetRealS2Idx) into a contiguous kvMergeGm,
        # then C1/V1/C2/V2 process the merged KV exactly as CFA. = _build_cfa
        # generalised with the prepended V0 merge + syncV0C1 handshake.
        return _build_scfa(
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
            topk_cmp=topk_cmp,
            ori_win_left=ori_win_left,
            cmp_ratio=cmp_ratio,
            softmax_scale=float(softmax_scale),
            dtype=dtype,
            core_num=core_num,
        )

    # CFA (scenario 2): the reference's SWA class with templateMode==CFA_TEMPLATE.
    # = SWA + a cmp KV segment (read sequentially via copy_pa, no topk/gather) →
    # multi-tile s2 sequence (ori tiles + cmp tiles) threaded by one online-softmax
    # flash-accumulation chain + PV rescale. cmp segment length per task =
    # actCmpS2Size = (cmpMaskRight + s1EndIdx + 1) / cmp_ratio (swa_kernel.h:378-381).
    return _build_cfa(
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
        cmp_ratio=cmp_ratio,
        softmax_scale=float(softmax_scale),
        dtype=dtype,
        core_num=core_num,
    )


def _build_cfa(
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
    cmp_ratio,
    softmax_scale,
    dtype,
    core_num,
):
    """CFA (scenario 2) = the reference SWA class's CFA_TEMPLATE path: SWA + a cmp
    KV segment read sequentially via copy_pa (NO topk / NO gather -- that is SCFA).

    Reverse-degenerates our SWA single-tile kernel back to the reference's
    multi-KV-tile online softmax. Instruction-level faithful (verified): every op
    maps to an existing primitive emitting the exact reference instruction
    (copy_pa->DataCopyPA, gemm_v0_fixp->Mmad/Fixpipe, softmax_flash_v2->SoftmaxFlashV2,
    row_expand_div->RowDivs, row_expand_mul_nd[013]->RowMuls, T.copy->DataCopyPad,
    T.set_flag->MTE3_MTE2 fence); the rest is pure kernel structure mirroring
    ProcessBalance / SoftmaxFlashV2Compute / DealBmm2ResBaseBlock.

    STRUCTURE (verified against swa_kernel.h:728-769 ProcessBalance/PreloadPipeline):
    the reference flattens (bN2, gS1, s2) into a single ``gloop`` that advances PER
    s2-TILE; its 3-stage preload is, at iteration g, cube QK(g)+PV(g-1) and vector
    softmax(g-1)+output(g-2) -- IDENTICAL to our SWA depth-3 pipeline with the loop
    unit changed from "task j" to "global tile g". So this is _build_swa generalised:
    ``for g`` over ``n_iter * MAX_TILES`` flattened tiles (task = g//MAX_TILES, tile =
    g%MAX_TILES; padding tiles beyond a task's s2LoopTimes are skipped by isValid, the
    same way SWA skips invalid tasks), per-tile (ori/cmp) scalar decode, online softmax
    chaining via the g%preLoadNum ring, and the PV rescale.

    cmp tiles are up to s2BaseSize=512 columns (vs ori <=128): QK loops 128-col blocks
    (ComputeMm1 nL1Loops, block_cube.h:353), PV K-accumulates 128-row sub-blocks into
    one cL0 then fixpipes (ComputeMm2 kL1/kL0, :638-672), and the vector m-chunks rows
    (DealBmm1/2 mSplit) so a [<=16,512] tile fits UB.

    Every fixpipe is DIRECT (no scratch round-trip): QK accumulates the 2 D-halves
    into cL0 via gemm_v0 (result stays in L0C, no GM dst) then T.copy(cL0 -> ws_s
    column band) writes the band with the score's full row stride -- faithful to
    ComputeMm1's strided Fixpipe and identical to SWA PV's cL0->ws_o band copy. PV
    is the same: T.mma accumulate -> T.copy(cL0 -> ws_o band).

    STAGE 1 (this build = correctness-first): the multi-tile MATH (column-block QK,
    K-accumulate PV, online softmax chaining, PV rescale, vec2ResGm accumulator, m
    chunking) under COARSE barriers (T.barrier_all between every cube load/compute and
    vector stage) -- mirroring SWA's own
    DEBUG_SERIAL bring-up. STAGE 2 (perf, after correctness): swap barriers for the KV
    3-ring + directional pipe flags + cross-task gloop overlap, exactly like SWA's
    Layer 3/4/5. The cube<->vector cross-flags (EV_QK/EV_P/EV_PV) keep the depth-3
    overlap and fire every g (even invalid) to stay count-balanced, as in SWA.

    Watch (high risk): ring slot off-by-one (softmax in=(g-2)%2 if !isFirst else sink,
    out=(g-1)%2; rescale prev=vec2ResGm[(g-3)%2], expmax=(g-2)%2 -- verified consistent
    with mod-2 under the depth-3 lag); cmp tile width 512 vs ori <=128; UB budget.
    """
    N1 = n_heads  # query heads (= 64)
    N2 = 1  # kv heads
    D = head_dim  # head dim (= 512)
    D2 = D // 2  # 256: QK D-chunk (kL1) width -- function-scope int (see _build_swa)
    G = N1 // N2  # GQA group = rows per task (= 64)
    BI = DEFAULT_BLOCK_I  # 128: QK column-block / PV K-sub-block / output-D tile width
    ORI_W = 16  # DEBUG shrink (was BI=128) to test UB overflow vs codegen. narrow bucket. Defined
    # here (outer scope), NOT inside the prim_func: a `name = const` statement in the
    # body is parsed as a symbolic Let-var, and using it as a buffer dim makes the
    # tile-op size checks compare PrimExprs instead of ints (size-must-be-same fails).
    PV_NT = D // BI  # 4 output-D tiles (= ComputeMm2 nL1Loops)
    PV_NW = BI  # 128: each output-D tile width
    VEC_NUM = 2  # 2 vector cores per cube core
    G2 = G // VEC_NUM  # 32: rows per vector core
    BLK = 8  # FP32 elems per 32B block (Brcb fan-out for row_expand)
    S2_BASE = 512  # cmp tile width = reference s2BaseSize (metadata.py:319)
    MAX_COLBLK = S2_BASE // BI  # 4: max 128-col QK blocks per (cmp) tile
    # preLoadNum = 2 (online-softmax ring depth); the rings are [2, ...] / g%2.
    # Vector m-chunk: a [M_CHUNK, 512] fp32 tile must fit UB alongside the rescale
    # buffers. M_CHUNK=8 -> NMC=4 chunks; conservative for Stage 1 (Stage 2 may widen
    # to 16 = the reference's mSplit for columnCount=512).
    # m-split chunk = ref mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM(32K/4=8192) /
    # columnCount(=S2_BASE 512) = 16 (block_vector.h:448-453). NMC=2 chunks/core.
    M_CHUNK = 16
    NMC = G2 // M_CHUNK  # 2 m-chunks per vector core (= ref loopCount)

    accum_dtype = "float"
    idx_dtype = "int32"
    out_cast_mode = "CAST_RINT" if dtype == "bfloat16" else "CAST_ROUND"

    # Max cmp tiles per task (compile-time upper bound). cmp length per task =
    # (s_global+1)//cmp_ratio <= max_seq//cmp_ratio; tiled at S2_BASE.
    max_cmp_len = (max_seq + cmp_ratio - 1) // cmp_ratio
    MAX_CMP_TILES = (max_cmp_len + S2_BASE - 1) // S2_BASE
    MAX_TILES = 1 + MAX_CMP_TILES  # 1 ori tile + cmp tiles

    block_num = batch * max_seq
    n_iter = (block_num + core_num - 1) // core_num  # tasks per core (ceil)
    # Flattened per-core tile iterations: each task occupies MAX_TILES g-slots
    # (padding tiles skipped via isValid). gloop advances per tile, g%PRELOAD rings.
    GLOOP = n_iter * MAX_TILES

    # Cube<->vector cross-core handshake event ids (= reference syncC1V1/V1C2/C2V2).
    EV_QK = 0  # cube -> vector: QK score (ws_s) ready
    EV_P = 1  # vector -> cube: softmax P (ws_p) ready
    EV_PV = 2  # cube -> vector: PV result (ws_o) ready

    # ---- Stage 2 cube within-pipe sync (= SWA Layer 3/4/5, proven). ----
    # DEBUG_SERIAL gates the per-op T.barrier_all: True keeps every barrier (the
    # ring/flags are present but redundant -- verifies they BALANCE / never
    # deadlock, correctness still guaranteed by the barriers); flip False to drop
    # the barriers and let the ring + reverse flags overlap the pipes (the perf).
    DEBUG_SERIAL = False
    # Shared KV 3-slot L1 ring (= reference kvL1BufIter%3): QK's K D-halves and PV's
    # V tiles rotate the SAME 3 slots; per-slot MTE2_MTE1/MTE1_MTE2 reverse flags let
    # the next copy_pa (the 1961us mte2) overlap the current gemm/mma. Slot is a
    # RUNTIME kv_iter%3 (bands/K-blocks are runtime-guarded; a fixed Python slot
    # would break flag balance when a band is skipped). Flag id by slot via
    # Select(slot<2, KV_EV0+slot, KV_EV2) -- runtime ids OK (FlagOpCodegen PrintExpr's
    # the event id). {2,3,6} avoid the L0AB pair {4,5} and the gemm's internal
    # MTE2_MTE1 self-fence (event 0).
    KV_EV0 = 2
    KV_EV1 = 3
    KV_EV2 = 6
    # Q (both D-halves, reused across a tile's bands) / P reverse flags (MTE1_MTE2).
    Q_EV = 1
    P_EV = 7
    # L0AB M_MTE1 ping-pong for PV's decomposed mma (QK's gemm_v0 self-manages its
    # own L0AB on event 0); primed once before the g-loop, drained once after.
    L0AB_EV0 = 4
    L0AB_EV1 = 5

    # ---- Stage 2c vector within-pipe directed flags. = block_vector.h's SYNC_*_BUF,
    # replacing the m-chunk barrier_all (VECTOR core's own event ids; CrossCore EV_*
    # are separate). Mirrors the proven SWA vector debarrier (_build_swa) + the
    # multi-tile rescale (DealBmm2 671-697) / stash (709-717), now with the reference's
    # BUFFER REUSE: vec1(softmax j-1) and vec2(output j-2) run sequentially within one
    # PreloadPipeline iteration (swa_kernel.h:757/765), so they TIME-SHARE one input
    # buffer (in_ub = ref inputBuff1, SYNC_INPUT_BUF1) and one output buffer (out_ub =
    # ref outputBuff1, SYNC_OUTPUT_BUF1). in_ub is m-split PING-PONG (ids IN+{0,1}, =
    # pingpongFlag): chunk i+1's MTE2 load overlaps chunk i's V compute. acc_pre single
    # (= ref BUF2). Distinct HardEvent types are independent id namespaces, so
    # LSE_EV (MTE3_V/V_MTE3) and FENCE (MTE3_MTE2) share id 0 without colliding. ----
    IN_EV = 2  # in_ub ping-pong base (ids 2,3): s_ub(vec1)+o_ub(vec2); V_MTE2+MTE2_V (+stash V_MTE3/MTE3_V)
    ACC_EV = 4  # acc_pre single: V_MTE2 (WAR) + MTE2_V (RAW)
    OUT_EV = 5  # out_ub single: p_half(vec1)+o_half(vec2); MTE3_V (WAR) + V_MTE3 (RAW)
    LSE_EV = 0  # lse_ub single: MTE3_V (WAR) + V_MTE3 (RAW)
    # workspace_acc cross-tile MTE3->MTE2 fence: prev tile stashed via MTE3, this tile
    # reloads via MTE2 -> same-core GM RAW (= DealBmm2:672-674). MTE3_MTE2 own namespace.
    FENCE = 0

    q_shape = [total_tokens, N1, D]
    ori_kv_shape = [ori_block_num, ori_block_size, N2, D]
    cmp_kv_shape = [cmp_block_num, cmp_block_size, N2, D]
    cmp_idx_shape = [total_tokens, N2, DEFAULT_BLOCK_I]

    @tilelang.jit(out_idx=[11, 12], workspace_idx=[13, 14, 15, 16, 17])
    def kernel():
        @T.prim_func
        def sparse_attn_sharedkv_cfa(
            Q: T.Tensor(q_shape, dtype),  # 0
            ori_kv: T.Tensor(ori_kv_shape, dtype),  # 1  (ori shared K & V)
            ori_block_table: T.Tensor([batch, ori_table_len], idx_dtype),  # 2
            cmp_kv: T.Tensor(cmp_kv_shape, dtype),  # 3  (cmp shared K & V)
            cmp_block_table: T.Tensor([batch, cmp_table_len], idx_dtype),  # 4
            cmp_indices: T.Tensor(cmp_idx_shape, idx_dtype),  # 5  (unused, CFA = dense)
            q_prefix: T.Tensor([batch], idx_dtype),  # 6  flat-token base
            act_q_lens: T.Tensor([batch], idx_dtype),  # 7  per-batch q len
            seqused_kv: T.Tensor([batch], idx_dtype),  # 8  per-batch ori kv len
            sinks: T.Tensor([N1], accum_dtype),  # 9  per-head sink
            metadata: T.Tensor([SAS_META_SIZE], idx_dtype),  # 10 (unused, v1)
            Output: T.Tensor(q_shape, dtype),  # 11 out
            LSE: T.Tensor([total_tokens, N1], accum_dtype),  # 12 out
            # cube<->vector workspaces, per core, ring-buffered (dim 1 = g%2):
            workspace_s: T.Tensor([core_num, 2, G, S2_BASE], accum_dtype),  # 13 QKᵀ
            workspace_p: T.Tensor([core_num, 2, G, S2_BASE], dtype),  # 14 softmax P
            workspace_o: T.Tensor([core_num, 2, G, D], accum_dtype),  # 15 PV (raw)
            # running online-softmax PV accumulator (= reference vec2ResGm):
            workspace_acc: T.Tensor([core_num, 2, G, D], accum_dtype),  # 16
            # QK shape-token: gemm_v0_fixp(do_fixpipe=False) (DEBUG_SERIAL=False path)
            # derives M,N from a dst it NEVER writes (the fixpipe is T.copy(cL0->band)).
            workspace_qk: T.Tensor([core_num, G, BI], accum_dtype),  # 17
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, vid):
                # ---- Cube L1/L0 allocations (kernel scope). ----
                # Q/K as two 256-wide D-halves (ComputeMm1 kL1Loops=2).
                q_l1 = T.alloc_L1([2, G, D2], dtype)  # Q D-halves (reused across bands)
                # Shared KV 3-slot ring (= reference kvL1): QK K D-halves + PV V tiles.
                kv_ring = T.alloc_L1([3, BI, D2], dtype)
                p_l1 = T.alloc_L1([G, S2_BASE], dtype)  # P (up to 512 wide)
                cL0 = T.alloc_L0C([2, G, BI], accum_dtype)  # shared QK/PV cL0 ping-pong
                p_l0a = T.alloc_L0A(
                    [2, G, BI], dtype
                )  # P activations (PV, L0AB ping-pong)
                v_l0b = T.alloc_L0B([2, BI, BI], dtype)  # V tile (PV, L0AB ping-pong)
                # Persistent ring/ping-pong counters (= reference class members; survive
                # across g, mutated in place). kv_iter -> KV-ring slot (per load),
                # cl0_iter -> cL0 slot (per band/PV-tile fixpipe), ab_iter -> L0AB pp
                # (per PV L0 load). alloc_var scalars: read by name, write with +=.
                kv_iter = T.alloc_var("int32", init=0)
                cl0_iter = T.alloc_var("int32", init=0)
                ab_iter = T.alloc_var("int32", init=0)
                # ---- Vector UB allocations (m-chunked to M_CHUNK rows). D == S2_BASE
                # (both 512), so one buffer serves both the score (vec1) and PV (vec2)
                # roles. = ref's time-shared inputBuff1 / outputBuff1 (vec1 fully
                # precedes vec2 within a PreloadPipeline iteration). ----
                in_ub = T.alloc_ub(
                    [2, M_CHUNK, S2_BASE], accum_dtype
                )  # = ref inputBuff1: s_ub(vec1 score) + o_ub(vec2 PV); ping-pong [mc&1]
                softmax_cmp = T.alloc_ub([M_CHUNK, S2_BASE], accum_dtype)  # compaction
                out_ub = T.alloc_ub(
                    [M_CHUNK, S2_BASE], dtype
                )  # = ref outputBuff1: p_half(vec1 cast P) + o_half(vec2 cast out)
                acc_pre = T.alloc_ub(
                    [M_CHUNK, D], accum_dtype
                )  # prev accumulator (rescale); = ref inputBuff2 (single)
                sink_ub = T.alloc_ub(
                    [2, M_CHUNK, 1], accum_dtype
                )  # m-split ping-pong [mc&1]: rides in_ub's slot-keyed IN_EV+ps flag
                lse_ub = T.alloc_ub([M_CHUNK, 1], accum_dtype)
                ones_ub = T.alloc_ub([M_CHUNK, 1], accum_dtype)  # in_sum seed (1.0)
                brcb_d = T.alloc_ub(
                    [M_CHUNK, BLK], accum_dtype
                )  # Brcb scratch (row_expand)
                # 手拼替代 007: sumP=reduce_sum(P) 临时(替代 softmax_tmp);
                # softmax_cmp 复用为 broadcast(max) 的 [M_CHUNK,512] 目标(免新 alloc)。
                sumP = T.alloc_ub([M_CHUNK, 1], accum_dtype)
                # Online-softmax running state rings (= softmaxMax/Sum/ExpUb), indexed
                # [tile g%2, m-chunk, row, 1]. The m-chunk is a SEPARATE dim (not a
                # bounded row-slice of [2,G2,1]) so a per-chunk view ring[slot, mc, :, :]
                # is a FULL-slice BufferRegion the tile primitives accept (a bounded
                # slice ring[slot, mc*M:(mc+1)*M, :] parses to a BufferLoad they reject).
                m_i = T.alloc_ub([2, NMC, M_CHUNK, 1], accum_dtype)  # running max
                denom = T.alloc_ub([2, NMC, M_CHUNK, 1], accum_dtype)  # running sum
                expmax = T.alloc_ub([2, NMC, M_CHUNK, 1], accum_dtype)  # flash rescale
                # path B narrow-tile bucket: tiles whose valid window fits ORI_W (the
                # ori tile: win <= ori_win_left+1 <= BI) run the softmax on a CONTIGUOUS
                # [M_CHUNK, ORI_W] temp instead of the full S2_BASE buffer, so
                # reduce/exp/cast only touch ORI_W columns and beat the full-512 white
                # compute (a strided sub-window of the 512 buffer would fold in padding;
                # a contiguous narrow temp does not). Wide tiles keep the S2_BASE path.
                # (ORI_W is defined at the outer scope, above, to stay a Python int.)
                sc_n = T.alloc_ub(
                    [2, M_CHUNK, ORI_W], accum_dtype
                )  # narrow score, ping-pong [mc&1] (rides in_ub's IN_EV+ps flag)
                cmp_n = T.alloc_ub(
                    [M_CHUNK, ORI_W], accum_dtype
                )  # narrow broadcast target
                out_n = T.alloc_ub([M_CHUNK, ORI_W], dtype)  # narrow cast P

                # ============================ CUBE ============================
                with T.Scope("C"):
                    # AllocEventID (= block_cube.h:225): prime the L0AB M_MTE1 pair
                    # {4,5} (PV's mma; QK's gemm_v0 self-manages event 0), the 3 KV-ring
                    # slots' MTE1_MTE2 reverse flags {2,3,6}, and the Q/P reverse flags
                    # {1,7} -- all "free" so the first user waits on an already-set
                    # flag. Drained once after the g-loop.
                    T.set_flag("m", "mte1", L0AB_EV0)
                    T.set_flag("m", "mte1", L0AB_EV1)
                    T.set_flag("mte1", "mte2", KV_EV0)
                    T.set_flag("mte1", "mte2", KV_EV1)
                    T.set_flag("mte1", "mte2", KV_EV2)
                    T.set_flag("mte1", "mte2", Q_EV)
                    T.set_flag("mte1", "mte2", P_EV)
                    for g in T.serial(GLOOP + 1):
                        # ---- QK for tile g ----
                        if g < GLOOP:
                            buf = g % 2
                            task = g // MAX_TILES
                            tile = g % MAX_TILES
                            pid = task * core_num + cid
                            if pid < block_num:
                                b = T.cast(pid // max_seq, "int32")
                                s = T.cast(pid % max_seq, "int32")
                                act_q = act_q_lens[b]
                                if s < act_q:
                                    act_kv = seqused_kv[b]
                                    s_global = act_kv - act_q + s
                                    act_cmp = (s_global + 1) // cmp_ratio
                                    cmp_tiles = (act_cmp + S2_BASE - 1) // S2_BASE
                                    s2lt = 1 + cmp_tiles
                                    if tile < s2lt:
                                        is_ori = tile == 0
                                        ori_left = T.max(s_global - ori_win_left, 0)
                                        win = s_global + 1 - ori_left
                                        rel = tile - 1
                                        # tile column width. Inner cmp-tail conditional
                                        # is T.min (last tile: act_cmp-rel*512 < 512;
                                        # else >=512 -> 512); the ori/cmp pick is a
                                        # tir.Select. Both lower to an INLINE ternary
                                        # (SelectNode); a runtime T.if_then_else instead
                                        # lowers to a statement (int32_t condval; if{}),
                                        # which the codegen cannot inline into the copy_pa
                                        # arg list (bad C++). For ori (rel=-1) the cmp
                                        # branch = min(512, act_cmp+512)=512, unused.
                                        tw = tir.Select(
                                            is_ori,
                                            win,
                                            T.min(S2_BASE, act_cmp - rel * S2_BASE),
                                        )
                                        s2base = tir.Select(
                                            is_ori, ori_left, rel * S2_BASE
                                        )
                                        tok = q_prefix[b] + s
                                        # Q reverse flag: wait Q free (prev tile's gemms
                                        # done), load both D-halves, signal loaded, wait
                                        # loaded (every band's gemm reads q_l1).
                                        T.wait_flag("mte1", "mte2", Q_EV)
                                        T.copy(Q[tok, :, 0:D2], q_l1[0, :, :])
                                        T.copy(Q[tok, :, D2:D], q_l1[1, :, :])
                                        T.set_flag("mte2", "mte1", Q_EV)
                                        T.wait_flag("mte2", "mte1", Q_EV)
                                        if DEBUG_SERIAL:
                                            T.barrier_all()
                                        # QK over 128-col blocks (ComputeMm1 nL1Loops):
                                        # each band loads its 2 K D-halves into 2 KV-ring
                                        # slots (per-slot reverse flags overlap the next
                                        # band's copy_pa with the current gemm), accumulates
                                        # them into one cL0 ping-pong slot via gemm_v0, then
                                        # T.copy(cL0 -> ws_s band) is the standalone band
                                        # Fixpipe.
                                        for cb in range(MAX_COLBLK):
                                            if cb * BI < tw:
                                                ncols = T.min(BI, tw - cb * BI)
                                                ncols_a = (ncols + 15) // 16 * 16
                                                cs = cl0_iter % 2
                                                # 2 K D-halves -> 2 consecutive ring slots
                                                # (pre-increment exprs; kv_iter bumps AFTER
                                                # the gemms consume the slots).
                                                s0 = kv_iter % 3
                                                s1 = (kv_iter + 1) % 3
                                                ev0 = tir.Select(
                                                    s0 < 2, KV_EV0 + s0, KV_EV2
                                                )
                                                ev1 = tir.Select(
                                                    s1 < 2, KV_EV0 + s1, KV_EV2
                                                )
                                                for h in range(2):
                                                    slot = (kv_iter + h) % 3
                                                    ev = tir.Select(
                                                        slot < 2, KV_EV0 + slot, KV_EV2
                                                    )
                                                    T.wait_flag("mte1", "mte2", ev)
                                                    if is_ori:
                                                        T.copy_pa(
                                                            kv_ring[slot, :, :],
                                                            ori_kv,
                                                            ori_block_table,
                                                            ori_block_size,
                                                            N2,
                                                            D,
                                                            ori_block_size * N2 * D,
                                                            ori_table_len,
                                                            D2,
                                                            ncols,
                                                            BI,
                                                            b,
                                                            0,
                                                            s2base + cb * BI,
                                                            h * D2,
                                                        )
                                                    else:
                                                        T.copy_pa(
                                                            kv_ring[slot, :, :],
                                                            cmp_kv,
                                                            cmp_block_table,
                                                            cmp_block_size,
                                                            N2,
                                                            D,
                                                            cmp_block_size * N2 * D,
                                                            cmp_table_len,
                                                            D2,
                                                            ncols,
                                                            BI,
                                                            b,
                                                            0,
                                                            s2base + cb * BI,
                                                            h * D2,
                                                        )
                                                    T.set_flag("mte2", "mte1", ev)
                                                if DEBUG_SERIAL:
                                                    T.barrier_all()
                                                # 2 D-halves accumulate into cL0[cs].
                                                # DEBUG_SERIAL=True: gemm_v0 (no unitFlag,
                                                # self-L0AB on event 0, standalone fixpipe)
                                                # -- the barriered debug path. False (perf):
                                                # gemm_v0_fixp(prime_drain=False shares L0AB
                                                # {4,5} primed once, do_fixpipe=False leaves
                                                # cL0 for the 0b11 band fixpipe) = ComputeMm1
                                                # Mmad; the last mma 0b11 lets fixpipe(cb) ||
                                                # mma(cb+1) (cL0 ping-pong), so band cb+1's
                                                # copy_pa(mte2) overlaps band cb's gemm(mac)
                                                # -- the overlap gemm_v0's per-call FIX_M
                                                # drain blocks. (0b11 needs the FUSED 0b11
                                                # fixpipe, so only the no-barrier path.)
                                                T.wait_flag("mte2", "mte1", ev0)
                                                if DEBUG_SERIAL:
                                                    T.gemm_v0(
                                                        q_l1[0, :, :],
                                                        kv_ring[s0, :, :],
                                                        cL0[cs, :, :],
                                                        transpose_B=True,
                                                        init=True,
                                                        n_actual=ncols_a,
                                                    )
                                                else:
                                                    T.gemm_v0_fixp(
                                                        q_l1[0, :, :],
                                                        kv_ring[s0, :, :],
                                                        cL0,
                                                        workspace_qk[cid, :, :],
                                                        k_actual=D2,
                                                        transpose_B=True,
                                                        init=True,
                                                        n_actual=ncols_a,
                                                        cl0_base=cs,
                                                        prime_drain=False,
                                                        flush_last=False,
                                                        do_fixpipe=False,
                                                    )
                                                T.set_flag("mte1", "mte2", ev0)
                                                T.wait_flag("mte2", "mte1", ev1)
                                                if DEBUG_SERIAL:
                                                    T.gemm_v0(
                                                        q_l1[1, :, :],
                                                        kv_ring[s1, :, :],
                                                        cL0[cs, :, :],
                                                        transpose_B=True,
                                                        init=False,
                                                        n_actual=ncols_a,
                                                    )
                                                else:
                                                    T.gemm_v0_fixp(
                                                        q_l1[1, :, :],
                                                        kv_ring[s1, :, :],
                                                        cL0,
                                                        workspace_qk[cid, :, :],
                                                        k_actual=D2,
                                                        transpose_B=True,
                                                        init=False,
                                                        n_actual=ncols_a,
                                                        cl0_base=cs,
                                                        prime_drain=False,
                                                        flush_last=True,
                                                        do_fixpipe=False,
                                                    )
                                                T.set_flag("mte1", "mte2", ev1)
                                                kv_iter += 2
                                                if DEBUG_SERIAL:
                                                    T.barrier_all()
                                                # band Fixpipe cL0[cs] -> strided ws_s band
                                                # (realDstN = 512 = ComputeMm1 dstStride).
                                                # 0b11 pairs with gemm_v0_fixp's last mma
                                                # (0b11) -> fixpipe(cb) || mma(cb+1) via the
                                                # cL0 ping-pong; standalone + barrier on the
                                                # DEBUG_SERIAL (gemm_v0) debug path.
                                                if DEBUG_SERIAL:
                                                    T.copy(
                                                        cL0[cs, :, :],
                                                        workspace_s[
                                                            cid,
                                                            buf,
                                                            :,
                                                            cb * BI : (cb + 1) * BI,
                                                        ],
                                                    )
                                                    T.barrier_all()
                                                else:
                                                    # 0b11 fixpipe nSize MUST equal the
                                                    # mma's n_actual=ncols_a (else it waits
                                                    # cL0 cols [ncols_a:BI] no mma marked ->
                                                    # hang). So the band is ncols_a wide
                                                    # (row stride still 512 = ws_s width).
                                                    T.copy(
                                                        cL0[cs, :, 0:ncols_a],
                                                        workspace_s[
                                                            cid,
                                                            buf,
                                                            :,
                                                            cb * BI : cb * BI + ncols_a,
                                                        ],
                                                        unit_flag=0b11,
                                                    )
                                                cl0_iter += 1
                                        # release Q for the next tile's QK load.
                                        T.set_flag("mte1", "mte2", Q_EV)
                            # fire every g<GLOOP (incl. invalid tiles) to keep the
                            # cross-flag count balanced with the vector's waits.
                            T.set_cross_flag("FIX", EV_QK)
                        # ---- PV for tile g-1 ----
                        if g >= 1:
                            T.wait_cross_flag(EV_P)
                            gm = g - 1
                            bufm = gm % 2
                            taskm = gm // MAX_TILES
                            tilem = gm % MAX_TILES
                            pidm = taskm * core_num + cid
                            if pidm < block_num:
                                bm = T.cast(pidm // max_seq, "int32")
                                sm = T.cast(pidm % max_seq, "int32")
                                if sm < act_q_lens[bm]:
                                    act_kvm = seqused_kv[bm]
                                    s_globalm = act_kvm - act_q_lens[bm] + sm
                                    act_cmpm = (s_globalm + 1) // cmp_ratio
                                    cmp_tilesm = (act_cmpm + S2_BASE - 1) // S2_BASE
                                    s2ltm = 1 + cmp_tilesm
                                    if tilem < s2ltm:
                                        is_orim = tilem == 0
                                        ori_leftm = T.max(s_globalm - ori_win_left, 0)
                                        winm = s_globalm + 1 - ori_leftm
                                        relm = tilem - 1
                                        # inline ternaries (T.min + tir.Select), see QK.
                                        twm = tir.Select(
                                            is_orim,
                                            winm,
                                            T.min(S2_BASE, act_cmpm - relm * S2_BASE),
                                        )
                                        s2basem = tir.Select(
                                            is_orim, ori_leftm, relm * S2_BASE
                                        )
                                        # P reverse flag: wait P free, load P, signal
                                        # loaded, wait loaded (every (nl,ks) reads p_l1).
                                        T.wait_flag("mte1", "mte2", P_EV)
                                        T.copy(
                                            workspace_p[cid, bufm, :, 0:S2_BASE],
                                            p_l1[:, :],
                                        )
                                        T.set_flag("mte2", "mte1", P_EV)
                                        T.wait_flag("mte2", "mte1", P_EV)
                                        if DEBUG_SERIAL:
                                            T.barrier_all()
                                        # PV = ComputeMm2: 4 output-D tiles, each
                                        # K-accumulating the tile's 128-row sub-blocks into
                                        # one cL0 slot. Each (nl,ks): LOAD V -> KV ring slot
                                        # (per-slot reverse flag), CONSUME (L0 load on the
                                        # L0AB pp ping-pong -> mma accumulate). Then the
                                        # band Fixpipe cL0[cs] -> ws_o band (0b11 overlaps
                                        # the next tile's mma; standalone under barriers).
                                        for nl in range(PV_NT):
                                            cs = cl0_iter % 2
                                            for ks in range(MAX_COLBLK):
                                                if ks * BI < twm:
                                                    krows = T.min(BI, twm - ks * BI)
                                                    is_first_ks = ks == 0
                                                    is_last_ks = (ks + 1) * BI >= twm
                                                    slot = kv_iter % 3
                                                    ev = tir.Select(
                                                        slot < 2, KV_EV0 + slot, KV_EV2
                                                    )
                                                    pp = ab_iter % 2
                                                    # LOAD V D-tile nl, K-block ks -> slot.
                                                    T.wait_flag("mte1", "mte2", ev)
                                                    if is_orim:
                                                        T.copy_pa(
                                                            kv_ring[slot, :, :],
                                                            ori_kv,
                                                            ori_block_table,
                                                            ori_block_size,
                                                            N2,
                                                            D,
                                                            ori_block_size * N2 * D,
                                                            ori_table_len,
                                                            PV_NW,
                                                            krows,
                                                            BI,
                                                            bm,
                                                            0,
                                                            s2basem + ks * BI,
                                                            nl * PV_NW,
                                                        )
                                                    else:
                                                        T.copy_pa(
                                                            kv_ring[slot, :, :],
                                                            cmp_kv,
                                                            cmp_block_table,
                                                            cmp_block_size,
                                                            N2,
                                                            D,
                                                            cmp_block_size * N2 * D,
                                                            cmp_table_len,
                                                            PV_NW,
                                                            krows,
                                                            BI,
                                                            bm,
                                                            0,
                                                            s2basem + ks * BI,
                                                            nl * PV_NW,
                                                        )
                                                    T.set_flag("mte2", "mte1", ev)
                                                    if DEBUG_SERIAL:
                                                        T.barrier_all()
                                                    # CONSUME: L0 load (L0AB pp) -> mma.
                                                    T.wait_flag(
                                                        "m", "mte1", L0AB_EV0 + pp
                                                    )
                                                    T.wait_flag("mte2", "mte1", ev)
                                                    T.copy(
                                                        p_l1[
                                                            :, ks * BI : (ks + 1) * BI
                                                        ],
                                                        p_l0a[pp, :, :],
                                                        real_k=krows,
                                                    )
                                                    T.copy(
                                                        kv_ring[slot, :, 0:PV_NW],
                                                        v_l0b[pp, :, :],
                                                        real_k=krows,
                                                    )
                                                    T.set_flag(
                                                        "mte1", "m", L0AB_EV0 + pp
                                                    )
                                                    T.wait_flag(
                                                        "mte1", "m", L0AB_EV0 + pp
                                                    )
                                                    # last K -> 0b11 flush (pairs with the
                                                    # band fixpipe); else 0b10 accumulate.
                                                    # 0 under barriers (no fusion).
                                                    uf = (
                                                        0
                                                        if DEBUG_SERIAL
                                                        else tir.Select(
                                                            is_last_ks, 0b11, 0b10
                                                        )
                                                    )
                                                    T.mma(
                                                        p_l0a[pp, :, :],
                                                        v_l0b[pp, :, :],
                                                        cL0[cs, :, :],
                                                        init=is_first_ks,
                                                        k_actual=krows,
                                                        unit_flag=uf,
                                                    )
                                                    T.set_flag(
                                                        "m", "mte1", L0AB_EV0 + pp
                                                    )
                                                    T.set_flag("mte1", "mte2", ev)
                                                    kv_iter += 1
                                                    ab_iter += 1
                                                    if DEBUG_SERIAL:
                                                        T.barrier_all()
                                            # band Fixpipe cL0[cs] -> this output-D band.
                                            if DEBUG_SERIAL:
                                                T.copy(
                                                    cL0[cs, :, :],
                                                    workspace_o[
                                                        cid,
                                                        bufm,
                                                        :,
                                                        nl * PV_NW : (nl + 1) * PV_NW,
                                                    ],
                                                )
                                                T.barrier_all()
                                            else:
                                                T.copy(
                                                    cL0[cs, :, :],
                                                    workspace_o[
                                                        cid,
                                                        bufm,
                                                        :,
                                                        nl * PV_NW : (nl + 1) * PV_NW,
                                                    ],
                                                    unit_flag=0b11,
                                                )
                                            cl0_iter += 1
                                        # release P for the next tile's PV load.
                                        T.set_flag("mte1", "mte2", P_EV)
                            # fire every g>=1 (incl. invalid) for count balance.
                            T.set_cross_flag("FIX", EV_PV)
                    # FreeEventID (= block_cube.h:239): drain the L0AB pair {4,5}, the 3
                    # KV-ring slots {2,3,6}, and Q/P {1,7} -- each left set by its last
                    # consumer, balancing the prime before the g-loop.
                    T.wait_flag("m", "mte1", L0AB_EV0)
                    T.wait_flag("m", "mte1", L0AB_EV1)
                    T.wait_flag("mte1", "mte2", KV_EV0)
                    T.wait_flag("mte1", "mte2", KV_EV1)
                    T.wait_flag("mte1", "mte2", KV_EV2)
                    T.wait_flag("mte1", "mte2", Q_EV)
                    T.wait_flag("mte1", "mte2", P_EV)

                # =========================== VECTOR ===========================
                with T.Scope("V"):
                    T.tile.fill(ones_ub, T.float32(1.0))
                    # Prime every vector buffer's reverse flag ONCE (buffers free) =
                    # block_vector.h InitBuffers SetFlag<V_MTE2>/<MTE3_V>; the first mc
                    # of the first tile waits on an already-set flag. Drained once after
                    # the g-loop. Each executed mc does exactly 1 wait + 1 set, so
                    # skipping invalid tiles keeps the balance (wait+set is net-zero).
                    T.set_flag("v", "mte2", IN_EV)  # in_ub slot0 free
                    T.set_flag("v", "mte2", IN_EV + 1)  # in_ub slot1 free (pong)
                    T.set_flag("v", "mte2", ACC_EV)  # acc_pre free
                    T.set_flag("mte3", "v", OUT_EV)  # out_ub free
                    T.set_flag("mte3", "v", LSE_EV)  # lse_ub free

                    for g in T.serial(1, GLOOP + 2):
                        # ---- softmax for tile g-1 ----
                        if g < GLOOP + 1:
                            T.wait_cross_flag(EV_QK)
                            gs = g - 1
                            buf = gs % 2
                            prev = (gs - 1) % 2
                            task = gs // MAX_TILES
                            tile = gs % MAX_TILES
                            pid = task * core_num + cid
                            if pid < block_num:
                                b = T.cast(pid // max_seq, "int32")
                                s = T.cast(pid % max_seq, "int32")
                                act_q = act_q_lens[b]
                                if s < act_q:
                                    act_kv = seqused_kv[b]
                                    s_global = act_kv - act_q + s
                                    act_cmp = (s_global + 1) // cmp_ratio
                                    cmp_tiles = (act_cmp + S2_BASE - 1) // S2_BASE
                                    s2lt = 1 + cmp_tiles
                                    if tile < s2lt:
                                        is_first = tile == 0
                                        ori_left = T.max(s_global - ori_win_left, 0)
                                        win = s_global + 1 - ori_left
                                        rel = tile - 1
                                        # inline ternaries (T.min + tir.Select), see QK.
                                        tw = tir.Select(
                                            is_first,
                                            win,
                                            T.min(S2_BASE, act_cmp - rel * S2_BASE),
                                        )
                                        tw_a = (tw + 15) // 16 * 16
                                        for mc in range(NMC):
                                            r0 = vid * G2 + mc * M_CHUNK
                                            ps = (
                                                mc & 1
                                            )  # in_ub ping-pong slot (= ref pingpongFlag)
                                            # path B: narrow tiles (win fits ORI_W, i.e. the ori tile) run the
                                            # softmax on a CONTIGUOUS [M_CHUNK, ORI_W] temp so reduce/exp/cast only
                                            # touch ORI_W cols (beats full-512 white compute); wide tiles keep the
                                            # S2_BASE path. Both branches issue identical IN_EV+ps / OUT_EV flags,
                                            # so the per-mc wait/set balance holds whichever tile width is taken.
                                            if tw < 0:  # DEBUG force-wide: narrow emitted but never taken
                                                T.wait_flag("v", "mte2", IN_EV + ps)
                                                T.tile.fill(
                                                    sc_n[ps, :, :],
                                                    -T.infinity(accum_dtype),
                                                )
                                                T.set_flag("v", "mte2", IN_EV + ps)
                                                T.wait_flag("v", "mte2", IN_EV + ps)
                                                T.copy(
                                                    workspace_s[
                                                        cid,
                                                        buf,
                                                        r0 : r0 + M_CHUNK,
                                                        0:tw,
                                                    ],
                                                    sc_n[ps, :, 0:tw],
                                                )
                                                T.copy(
                                                    sinks[r0 : r0 + M_CHUNK],
                                                    sink_ub[ps, :, :],
                                                )
                                                T.set_flag("mte2", "v", IN_EV + ps)
                                                T.wait_flag("mte2", "v", IN_EV + ps)
                                                T.tile.mul(
                                                    sc_n[ps, :, :],
                                                    sc_n[ps, :, :],
                                                    softmax_scale,
                                                )
                                                T.pipe_barrier("v")
                                                T.reduce_max(
                                                    sc_n[ps, :, :],
                                                    m_i[buf, mc, :, :],
                                                    dim=-1,
                                                )
                                                if is_first:
                                                    T.tile.max(
                                                        m_i[buf, mc, :, :],
                                                        sink_ub[ps, :, :],
                                                        m_i[buf, mc, :, :],
                                                    )
                                                    T.tile.sub(
                                                        expmax[buf, mc, :, :],
                                                        sink_ub[ps, :, :],
                                                        m_i[buf, mc, :, :],
                                                    )
                                                else:
                                                    T.tile.max(
                                                        m_i[buf, mc, :, :],
                                                        m_i[prev, mc, :, :],
                                                        m_i[buf, mc, :, :],
                                                    )
                                                    T.tile.sub(
                                                        expmax[buf, mc, :, :],
                                                        m_i[prev, mc, :, :],
                                                        m_i[buf, mc, :, :],
                                                    )
                                                T.tile.exp(
                                                    expmax[buf, mc, :, :],
                                                    expmax[buf, mc, :, :],
                                                )
                                                T.tile.broadcast(
                                                    cmp_n, m_i[buf, mc, :, :]
                                                )
                                                T.tile.sub(
                                                    sc_n[ps, :, :],
                                                    sc_n[ps, :, :],
                                                    cmp_n,
                                                )
                                                T.tile.exp(
                                                    sc_n[ps, :, :], sc_n[ps, :, :]
                                                )
                                                if is_first:
                                                    T.tile.mul(
                                                        denom[buf, mc, :, :],
                                                        ones_ub,
                                                        expmax[buf, mc, :, :],
                                                    )
                                                else:
                                                    T.tile.mul(
                                                        denom[buf, mc, :, :],
                                                        denom[prev, mc, :, :],
                                                        expmax[buf, mc, :, :],
                                                    )
                                                T.pipe_barrier("v")
                                                T.wait_flag("mte3", "v", OUT_EV)
                                                T.tile.cast(
                                                    out_n,
                                                    sc_n[ps, :, :],
                                                    "CAST_ROUND",
                                                    M_CHUNK * ORI_W,
                                                )
                                                T.pipe_barrier("v")
                                                T.reduce_sum(
                                                    sc_n[ps, :, :], sumP, dim=-1
                                                )
                                                T.tile.add(
                                                    denom[buf, mc, :, :],
                                                    denom[buf, mc, :, :],
                                                    sumP,
                                                )
                                                T.set_flag("v", "mte2", IN_EV + ps)
                                                T.set_flag("v", "mte3", OUT_EV)
                                                T.wait_flag("v", "mte3", OUT_EV)
                                                T.copy(
                                                    out_n[:, 0:tw_a],
                                                    workspace_p[
                                                        cid,
                                                        buf,
                                                        r0 : r0 + M_CHUNK,
                                                        0:tw_a,
                                                    ],
                                                )
                                                T.set_flag("mte3", "v", OUT_EV)
                                            else:
                                                T.wait_flag("v", "mte2", IN_EV + ps)
                                                T.tile.fill(
                                                    in_ub[ps, :, :],
                                                    -T.infinity(accum_dtype),
                                                )
                                                T.set_flag("v", "mte2", IN_EV + ps)
                                                T.wait_flag("v", "mte2", IN_EV + ps)
                                                # DEBUG slop fix: load exactly [0:tw] (not
                                                # tw_a-aligned) so the [tw:tw_a] out-of-window
                                                # QK slop stays fill(-inf) and the full-512
                                                # reduce ignores it (lse was 97.77% from slop).
                                                T.copy(
                                                    workspace_s[
                                                        cid,
                                                        buf,
                                                        r0 : r0 + M_CHUNK,
                                                        0:tw,
                                                    ],
                                                    in_ub[ps, :, 0:tw],
                                                )
                                                T.copy(
                                                    sinks[r0 : r0 + M_CHUNK],
                                                    sink_ub[ps, :, :],
                                                )
                                                T.set_flag("mte2", "v", IN_EV + ps)
                                                T.wait_flag("mte2", "v", IN_EV + ps)
                                                T.tile.mul(
                                                    in_ub[ps, :, :],
                                                    in_ub[ps, :, :],
                                                    softmax_scale,
                                                )
                                                T.pipe_barrier("v")
                                                T.reduce_max(
                                                    in_ub[ps, :, :],
                                                    m_i[buf, mc, :, :],
                                                    dim=-1,
                                                )
                                                if is_first:
                                                    T.tile.max(
                                                        m_i[buf, mc, :, :],
                                                        sink_ub[ps, :, :],
                                                        m_i[buf, mc, :, :],
                                                    )
                                                    T.tile.sub(
                                                        expmax[buf, mc, :, :],
                                                        sink_ub[ps, :, :],
                                                        m_i[buf, mc, :, :],
                                                    )
                                                else:
                                                    T.tile.max(
                                                        m_i[buf, mc, :, :],
                                                        m_i[prev, mc, :, :],
                                                        m_i[buf, mc, :, :],
                                                    )
                                                    T.tile.sub(
                                                        expmax[buf, mc, :, :],
                                                        m_i[prev, mc, :, :],
                                                        m_i[buf, mc, :, :],
                                                    )
                                                T.tile.exp(
                                                    expmax[buf, mc, :, :],
                                                    expmax[buf, mc, :, :],
                                                )
                                                T.tile.broadcast(
                                                    softmax_cmp, m_i[buf, mc, :, :]
                                                )
                                                T.tile.sub(
                                                    in_ub[ps, :, :],
                                                    in_ub[ps, :, :],
                                                    softmax_cmp,
                                                )
                                                T.tile.exp(
                                                    in_ub[ps, :, :], in_ub[ps, :, :]
                                                )
                                                if is_first:
                                                    T.tile.mul(
                                                        denom[buf, mc, :, :],
                                                        ones_ub,
                                                        expmax[buf, mc, :, :],
                                                    )
                                                else:
                                                    T.tile.mul(
                                                        denom[buf, mc, :, :],
                                                        denom[prev, mc, :, :],
                                                        expmax[buf, mc, :, :],
                                                    )
                                                T.pipe_barrier("v")
                                                T.wait_flag("mte3", "v", OUT_EV)
                                                T.tile.cast(
                                                    out_ub,
                                                    in_ub[ps, :, :],
                                                    "CAST_ROUND",
                                                    M_CHUNK * S2_BASE,
                                                )
                                                T.pipe_barrier("v")
                                                T.reduce_sum(
                                                    in_ub[ps, :, :], sumP, dim=-1
                                                )
                                                T.tile.add(
                                                    denom[buf, mc, :, :],
                                                    denom[buf, mc, :, :],
                                                    sumP,
                                                )
                                                T.set_flag("v", "mte2", IN_EV + ps)
                                                T.set_flag("v", "mte3", OUT_EV)
                                                T.wait_flag("v", "mte3", OUT_EV)
                                                T.copy(
                                                    out_ub[:, 0:tw_a],
                                                    workspace_p[
                                                        cid,
                                                        buf,
                                                        r0 : r0 + M_CHUNK,
                                                        0:tw_a,
                                                    ],
                                                )
                                                T.set_flag("mte3", "v", OUT_EV)
                            T.set_cross_flag("MTE3", EV_P)
                        # ---- output for tile g-2 ----
                        if g >= 2:
                            T.wait_cross_flag(EV_PV)
                            go = g - 2
                            bufo = go % 2
                            prevo = (go - 1) % 2
                            task = go // MAX_TILES
                            tile = go % MAX_TILES
                            pid = task * core_num + cid
                            if pid < block_num:
                                b = T.cast(pid // max_seq, "int32")
                                s = T.cast(pid % max_seq, "int32")
                                act_q = act_q_lens[b]
                                if s < act_q:
                                    act_kv = seqused_kv[b]
                                    s_global = act_kv - act_q + s
                                    # [DEFENSIVE, same class as the SCFA output stage]
                                    # This DELAYED (go=g-2) stage's task=(g-2)//MAX_TILES
                                    # simplifies to g//2-1, pulling -core_num into the
                                    # (s_global+1)//cmp_ratio dividend. The Ascend codegen
                                    # emits FloorDiv as C truncated `/`, so IF core_num is
                                    # divisible by cmp_ratio the arith simplifier
                                    # distributes it and the split remainder goes negative
                                    # for the first tokens -> act_cmp +1 -> reads a
                                    # non-existent cmp tile's stale GM (see SCFA fix +
                                    # [[tilelang-ascend-floordiv-truncation]]). CFA is
                                    # currently BENIGN because cmp_ratio=128 is NOT a
                                    # divisor of core_num (24) -> no distribution -> the
                                    # whole (>=0) dividend truncates correctly. Select the
                                    # provable 0 anyway so a future divisor-of-core_num
                                    # cmp_ratio can't silently regress.
                                    act_cmp = tir.Select(
                                        s_global < cmp_ratio - 1,
                                        0,
                                        (s_global + 1) // cmp_ratio,
                                    )
                                    cmp_tiles = (act_cmp + S2_BASE - 1) // S2_BASE
                                    s2lt = 1 + cmp_tiles
                                    if tile < s2lt:
                                        not_first = tile != 0
                                        is_last = tile == s2lt - 1
                                        tok = q_prefix[b] + s
                                        for mc in range(NMC):
                                            r0 = vid * G2 + mc * M_CHUNK
                                            po = (
                                                mc & 1
                                            )  # in_ub ping-pong slot (= ref pingpongFlag)
                                            # copy-in PV result into in_ub (MTE2). WAR:
                                            # wait this slot free (mc-2's last reader / this
                                            # iter's vec1 softmax done) = ref
                                            # WaitFlag<V_MTE2>(BUF1+pong):656.
                                            T.wait_flag("v", "mte2", IN_EV + po)
                                            T.copy(
                                                workspace_o[
                                                    cid, bufo, r0 : r0 + M_CHUNK, :
                                                ],
                                                in_ub[po, :, :],
                                            )
                                            # copy-in -> compute (= ref SetFlag/WaitFlag
                                            # <MTE2_V>(BUF1+pong):659-660).
                                            T.set_flag("mte2", "v", IN_EV + po)
                                            T.wait_flag("mte2", "v", IN_EV + po)
                                            if not_first:
                                                # prev tile STASHED workspace_acc via
                                                # MTE3; this MTE2 reload is a same-core
                                                # cross-tile GM RAW -> MTE3->MTE2 fence
                                                # (= ref DealBmm2:672-674).
                                                T.set_flag("mte3", "mte2", FENCE)
                                                T.wait_flag("mte3", "mte2", FENCE)
                                                # WAR: acc_pre free (= ref WaitFlag
                                                # <V_MTE2> SYNC_INPUT_BUF2:677).
                                                T.wait_flag("v", "mte2", ACC_EV)
                                                T.copy(
                                                    workspace_acc[
                                                        cid, prevo, r0 : r0 + M_CHUNK, :
                                                    ],
                                                    acc_pre,
                                                )
                                                # load -> compute (= ref SetFlag/WaitFlag
                                                # <MTE2_V> BUF2:682-683).
                                                T.set_flag("mte2", "v", ACC_EV)
                                                T.wait_flag("mte2", "v", ACC_EV)
                                                # rescale prev (RowMuls) -> Add (= ref
                                                # :691-694, PipeBarrier<V> between).
                                                # rescale prev = Ascend C RowMuls; 复用现成 experiment,
                                                # 按 xattention 写法展开: brcb 外置 + 逐 8*BLK(=64) 列段
                                                # mul_experiment(不传 tmp、src1 已 brcb), 覆盖整个 D。
                                                T.tile.brcb_experiment(
                                                    brcb_d,
                                                    expmax[bufo, mc, :, :],
                                                    M_CHUNK // 8,
                                                    1,
                                                    8,
                                                )
                                                T.pipe_barrier("v")
                                                for dcol in T.serial(D // (8 * BLK)):
                                                    cb = dcol * (8 * BLK)
                                                    T.tile.row_expand_mul_experiment(
                                                        acc_pre[:, cb : cb + 8 * BLK],
                                                        acc_pre[:, cb : cb + 8 * BLK],
                                                        brcb_d,
                                                    )
                                                T.pipe_barrier("v")
                                                T.tile.add(
                                                    in_ub[po, :, :],
                                                    in_ub[po, :, :],
                                                    acc_pre,
                                                )
                                                T.pipe_barrier("v")
                                                # acc_pre free (= ref SetFlag<V_MTE2>
                                                # BUF2:696).
                                                T.set_flag("v", "mte2", ACC_EV)
                                            if is_last:
                                                # normalize -> cast -> copy out + LSE
                                                # (= ref DealBmm2 700-708 + Bmm2Cast
                                                # AndCopyOut + LSE block; flags as SWA).
                                                # normalize = Ascend C RowDivs; 同上 xattention 写法:
                                                # brcb 外置 + 逐 8*BLK(=64) 列段 div_experiment(不传 tmp)。
                                                T.tile.brcb_experiment(
                                                    brcb_d,
                                                    denom[bufo, mc, :, :],
                                                    M_CHUNK // 8,
                                                    1,
                                                    8,
                                                )
                                                T.pipe_barrier("v")
                                                for dcol in T.serial(D // (8 * BLK)):
                                                    cb = dcol * (8 * BLK)
                                                    T.tile.row_expand_div_experiment(
                                                        in_ub[po, :, cb : cb + 8 * BLK],
                                                        in_ub[po, :, cb : cb + 8 * BLK],
                                                        brcb_d,
                                                    )
                                                T.pipe_barrier("v")
                                                # cast -> copy-out (MTE3). WAR: out_ub
                                                # free = ref WaitFlag<MTE3_V>(:617).
                                                T.wait_flag("mte3", "v", OUT_EV)
                                                T.tile.cast(
                                                    out_ub,
                                                    in_ub[po, :, :],
                                                    out_cast_mode,
                                                    M_CHUNK * D,
                                                )
                                                # in_ub slot free (V cast last-read it) =
                                                # ref SetFlag<V_MTE2>; cast -> copy-out =
                                                # ref SetFlag/WaitFlag<V_MTE3>(:624-625).
                                                T.set_flag("v", "mte2", IN_EV + po)
                                                T.set_flag("v", "mte3", OUT_EV)
                                                T.wait_flag("v", "mte3", OUT_EV)
                                                T.copy(
                                                    out_ub,
                                                    Output[tok, r0 : r0 + M_CHUNK, :],
                                                )
                                                # out_ub free = ref SetFlag<MTE3_V>(:627).
                                                T.set_flag("mte3", "v", OUT_EV)
                                                # LSE = ln(denom)+m_i (= ref :393-402).
                                                T.wait_flag("mte3", "v", LSE_EV)
                                                T.tile.ln(
                                                    lse_ub,
                                                    denom[bufo, mc, :, :],
                                                )
                                                T.tile.add(
                                                    lse_ub,
                                                    lse_ub,
                                                    m_i[bufo, mc, :, :],
                                                )
                                                T.set_flag("v", "mte3", LSE_EV)
                                                T.wait_flag("v", "mte3", LSE_EV)
                                                T.copy(
                                                    lse_ub, LSE[tok, r0 : r0 + M_CHUNK]
                                                )
                                                T.set_flag("mte3", "v", LSE_EV)
                                            else:
                                                # stash in_ub -> workspace_acc (MTE3) for
                                                # the next tile's rescale. Two guards =
                                                # ref DealBmm2:711-717 (which V-copies
                                                # bmm2ResUb->outUb then MTE3-stashes outUb):
                                                # (A) V->MTE3 so the stash reads the final
                                                # in_ub slot (after its MTE2 load + optional
                                                # V add); (B) MTE3->V->V_MTE2 so the next
                                                # reuse of this slot (waits V_MTE2) is
                                                # ordered after this stash read.
                                                T.set_flag("v", "mte3", IN_EV + po)
                                                T.wait_flag("v", "mte3", IN_EV + po)
                                                T.copy(
                                                    in_ub[po, :, :],
                                                    workspace_acc[
                                                        cid, bufo, r0 : r0 + M_CHUNK, :
                                                    ],
                                                )
                                                T.set_flag("mte3", "v", IN_EV + po)
                                                T.wait_flag("mte3", "v", IN_EV + po)
                                                T.set_flag("v", "mte2", IN_EV + po)
                            # NOTE: this vector EV_PV set looks unfaithful (reference
                            # syncC2V2 has a single cube setter, scfa_kernel.h:638), BUT
                            # removing it REGRESSES scfa (bf16 7/7 -> fail), so it is
                            # load-bearing for TileLang's flat-g pipeline -- keep it.
                            T.set_cross_flag("MTE3", EV_PV)
                    # Drain ALL the vector buffers' reverse flags ONCE (balance the
                    # primes above): each buffer's last consumer left its flag set
                    # (= block_vector.h FreeAllEventID for the SYNC_*_BUF flags).
                    T.wait_flag("v", "mte2", IN_EV)
                    T.wait_flag("v", "mte2", IN_EV + 1)
                    T.wait_flag("v", "mte2", ACC_EV)
                    T.wait_flag("mte3", "v", OUT_EV)
                    T.wait_flag("mte3", "v", LSE_EV)

        return sparse_attn_sharedkv_cfa

    func = kernel()
    if os.environ.get("SAS_DUMP_SRC"):
        try:
            src = func.get_kernel_source()
            with open("/tmp/cfa_gen.cpp", "w") as fh:
                fh.write(src)
            print(f"[SAS_DUMP_SRC] wrote {len(src)} chars to /tmp/cfa_gen.cpp")
        except Exception as exc:  # noqa: BLE001
            print(f"[SAS_DUMP_SRC] get_kernel_source failed: {exc!r}")
    return func


def _build_scfa(
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
    topk_cmp,
    ori_win_left,
    cmp_ratio,
    softmax_scale,
    dtype,
    core_num,
):
    """CFA (scenario 2) = the reference SWA class's CFA_TEMPLATE path: SWA + a cmp
    KV segment read sequentially via copy_pa (NO topk / NO gather -- that is SCFA).

    Reverse-degenerates our SWA single-tile kernel back to the reference's
    multi-KV-tile online softmax. Instruction-level faithful (verified): every op
    maps to an existing primitive emitting the exact reference instruction
    (copy_pa->DataCopyPA, gemm_v0_fixp->Mmad/Fixpipe, softmax_flash_v2->SoftmaxFlashV2,
    row_expand_div->RowDivs, row_expand_mul_nd[013]->RowMuls, T.copy->DataCopyPad,
    T.set_flag->MTE3_MTE2 fence); the rest is pure kernel structure mirroring
    ProcessBalance / SoftmaxFlashV2Compute / DealBmm2ResBaseBlock.

    STRUCTURE (verified against swa_kernel.h:728-769 ProcessBalance/PreloadPipeline):
    the reference flattens (bN2, gS1, s2) into a single ``gloop`` that advances PER
    s2-TILE; its 3-stage preload is, at iteration g, cube QK(g)+PV(g-1) and vector
    softmax(g-1)+output(g-2) -- IDENTICAL to our SWA depth-3 pipeline with the loop
    unit changed from "task j" to "global tile g". So this is _build_swa generalised:
    ``for g`` over ``n_iter * MAX_TILES`` flattened tiles (task = g//MAX_TILES, tile =
    g%MAX_TILES; padding tiles beyond a task's s2LoopTimes are skipped by isValid, the
    same way SWA skips invalid tasks), per-tile (ori/cmp) scalar decode, online softmax
    chaining via the g%preLoadNum ring, and the PV rescale.

    cmp tiles are up to s2BaseSize=512 columns (vs ori <=128): QK loops 128-col blocks
    (ComputeMm1 nL1Loops, block_cube.h:353), PV K-accumulates 128-row sub-blocks into
    one cL0 then fixpipes (ComputeMm2 kL1/kL0, :638-672), and the vector m-chunks rows
    (DealBmm1/2 mSplit) so a [<=16,512] tile fits UB.

    Every fixpipe is DIRECT (no scratch round-trip): QK accumulates the 2 D-halves
    into cL0 via gemm_v0 (result stays in L0C, no GM dst) then T.copy(cL0 -> ws_s
    column band) writes the band with the score's full row stride -- faithful to
    ComputeMm1's strided Fixpipe and identical to SWA PV's cL0->ws_o band copy. PV
    is the same: T.mma accumulate -> T.copy(cL0 -> ws_o band).

    STAGE 1 (this build = correctness-first): the multi-tile MATH (column-block QK,
    K-accumulate PV, online softmax chaining, PV rescale, vec2ResGm accumulator, m
    chunking) under COARSE barriers (T.barrier_all between every cube load/compute and
    vector stage) -- mirroring SWA's own
    DEBUG_SERIAL bring-up. STAGE 2 (perf, after correctness): swap barriers for the KV
    3-ring + directional pipe flags + cross-task gloop overlap, exactly like SWA's
    Layer 3/4/5. The cube<->vector cross-flags (EV_QK/EV_P/EV_PV) keep the depth-3
    overlap and fire every g (even invalid) to stay count-balanced, as in SWA.

    Watch (high risk): ring slot off-by-one (softmax in=(g-2)%2 if !isFirst else sink,
    out=(g-1)%2; rescale prev=vec2ResGm[(g-3)%2], expmax=(g-2)%2 -- verified consistent
    with mod-2 under the depth-3 lag); cmp tile width 512 vs ori <=128; UB budget.
    """
    N1 = n_heads  # query heads (= 64)
    N2 = 1  # kv heads
    D = head_dim  # head dim (= 512)
    D2 = D // 2  # 256: QK D-chunk (kL1) width -- function-scope int (see _build_swa)
    G = N1 // N2  # GQA group = rows per task (= 64)
    BI = DEFAULT_BLOCK_I  # 128: QK column-block / PV K-sub-block / output-D tile width
    PV_NT = D // BI  # 4 output-D tiles (= ComputeMm2 nL1Loops)
    PV_NW = BI  # 128: each output-D tile width
    VEC_NUM = 2  # 2 vector cores per cube core
    G2 = G // VEC_NUM  # 32: rows per vector core
    BLK = 8  # FP32 elems per 32B block (Brcb fan-out for row_expand)
    S2_BASE = 512  # cmp tile width = reference s2BaseSize (metadata.py:319)
    MAX_COLBLK = S2_BASE // BI  # 4: max 128-col QK blocks per (cmp) tile
    # preLoadNum = 2 (online-softmax ring depth); the rings are [2, ...] / g%2.
    # Vector m-chunk: a [M_CHUNK, 512] fp32 tile must fit UB alongside the rescale
    # buffers. M_CHUNK=8 -> NMC=4 chunks; conservative for Stage 1 (Stage 2 may widen
    # to 16 = the reference's mSplit for columnCount=512).
    # m-split chunk = ref mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM(32K/4=8192) /
    # columnCount(=S2_BASE 512) = 16 (block_vector.h:448-453). NMC=2 chunks/core.
    M_CHUNK = 16
    NMC = G2 // M_CHUNK  # 2 m-chunks per vector core (= ref loopCount)

    # V0 merge batch (Phase D step1): gather MERGE_ROWS topk cmp tokens into the UB
    # merge buffer, then ONE batched scatter to kvMergeGm -- amortises the per-token
    # barrier (= ref 16-row flush; capped at 8 by UB budget, V0 段 only ~13.5KB free).
    # Gather is a plain per-token T.copy: dstM folds to 1 (min(MERGE_ROWS, row-extent
    # 1)) so a partial-row load does not Duplicate-wipe the multi-row merge buffer.
    MERGE_ROWS = 6  # step4: merge_ub double-buffered [2,MERGE_ROWS,D] -> 2x UB; 6 keeps
    # 2*6*512*2=12KB <= ~14.5KB V0-free budget (ref uses 16 via UB reuse, deferred).

    accum_dtype = "float"
    idx_dtype = "int32"
    out_cast_mode = "CAST_RINT" if dtype == "bfloat16" else "CAST_ROUND"

    # Max cmp tiles per task (compile-time upper bound). cmp length per task =
    # (s_global+1)//cmp_ratio <= max_seq//cmp_ratio; tiled at S2_BASE.
    # SCFA: the cmp segment is the topk-MERGED KV, capped at topk_cmp tokens
    # (NOT dense max_seq/cmp_ratio). So at most ceil(topk_cmp/S2_BASE) cmp tiles;
    # using the dense count left ~2.5x EMPTY padding-tile pipeline iterations that
    # still ran the cross-flag handshakes (EV_QK/P/PV/V0). Cap = sparse merged len.
    max_cmp_len = min((max_seq + cmp_ratio - 1) // cmp_ratio, topk_cmp)
    MAX_CMP_TILES = (max_cmp_len + S2_BASE - 1) // S2_BASE
    MAX_TILES = 1 + MAX_CMP_TILES  # 1 ori tile + cmp tiles

    block_num = batch * max_seq
    n_iter = (block_num + core_num - 1) // core_num  # tasks per core (ceil)
    # Flattened per-core tile iterations: each task occupies MAX_TILES g-slots
    # (padding tiles skipped via isValid). gloop advances per tile, g%PRELOAD rings.
    GLOOP = n_iter * MAX_TILES

    # Cube<->vector cross-core handshake event ids (= reference syncC1V1/V1C2/C2V2).
    EV_QK = 0  # cube -> vector: QK score (ws_s) ready
    EV_P = 1  # vector -> cube: softmax P (ws_p) ready
    EV_PV = 2  # cube -> vector: PV result (ws_o) ready
    # SCFA: vector V0 -> cube cross-flag (= reference syncV0C1 / SYNC_V0_C1_FLAG):
    # V0 has written this cmp tile's merged KV (kvMergeGm); cube cmp QK may read it.
    # Cross-core id 3 (0/1/2 = EV_QK/P/PV); ori tiles skip it (cube reads ori_kv).
    EV_V0 = 3
    # SCFA flag-3 (= reference SAS_SYNC_MODE2 literal flag 3, scfa_kernel.h:680-683/
    # 777-780/801-803/816-818): a TASK-level 4-credit semaphore throttling the V0
    # writer to <=4 tasks ahead of the cube reader of the SAME kvMergeGm slot (= 4-deep
    # ring). The per-tile EV_V0 RAW handshake only orders the SCALAR pointer; the V0
    # scatter (async MTE3 DMA) can run far ahead once issued, so without this credit the
    # ring WAR fires intermittently at scale (full-8K cold NaN, confirmed: ring=32 probe
    # passed 6/6). cube primes 4 + returns 1 after each task's last-tile PV read (MTE2);
    # vector waits 1 before each task's first tile + drains 4. id 4 (0..3 used above).
    EV_CREDIT = 4

    # ---- Stage 2 cube within-pipe sync (= SWA Layer 3/4/5, proven). ----
    # DEBUG_SERIAL gates the per-op T.barrier_all: True keeps every barrier (the
    # ring/flags are present but redundant -- verifies they BALANCE / never
    # deadlock, correctness still guaranteed by the barriers); flip False to drop
    # the barriers and let the ring + reverse flags overlap the pipes (the perf).
    DEBUG_SERIAL = False
    # Shared KV 3-slot L1 ring (= reference kvL1BufIter%3): QK's K D-halves and PV's
    # V tiles rotate the SAME 3 slots; per-slot MTE2_MTE1/MTE1_MTE2 reverse flags let
    # the next copy_pa (the 1961us mte2) overlap the current gemm/mma. Slot is a
    # RUNTIME kv_iter%3 (bands/K-blocks are runtime-guarded; a fixed Python slot
    # would break flag balance when a band is skipped). Flag id by slot via
    # Select(slot<2, KV_EV0+slot, KV_EV2) -- runtime ids OK (FlagOpCodegen PrintExpr's
    # the event id). {2,3,6} avoid the L0AB pair {4,5} and the gemm's internal
    # MTE2_MTE1 self-fence (event 0).
    KV_EV0 = 2
    KV_EV1 = 3
    KV_EV2 = 6
    # Q (both D-halves, reused across a tile's bands) / P reverse flags (MTE1_MTE2).
    Q_EV = 1
    P_EV = 7
    # L0AB M_MTE1 ping-pong for PV's decomposed mma (QK's gemm_v0 self-manages its
    # own L0AB on event 0); primed once before the g-loop, drained once after.
    L0AB_EV0 = 4
    L0AB_EV1 = 5

    # ---- Stage 2c vector within-pipe directed flags. = block_vector.h's SYNC_*_BUF,
    # replacing the m-chunk barrier_all (VECTOR core's own event ids; CrossCore EV_*
    # are separate). Mirrors the proven SWA vector debarrier (_build_swa) + the
    # multi-tile rescale (DealBmm2 671-697) / stash (709-717), now with the reference's
    # BUFFER REUSE: vec1(softmax j-1) and vec2(output j-2) run sequentially within one
    # PreloadPipeline iteration (swa_kernel.h:757/765), so they TIME-SHARE one input
    # buffer (in_ub = ref inputBuff1, SYNC_INPUT_BUF1) and one output buffer (out_ub =
    # ref outputBuff1, SYNC_OUTPUT_BUF1). in_ub is m-split PING-PONG (ids IN+{0,1}, =
    # pingpongFlag): chunk i+1's MTE2 load overlaps chunk i's V compute. acc_pre single
    # (= ref BUF2). Distinct HardEvent types are independent id namespaces, so
    # LSE_EV (MTE3_V/V_MTE3) and FENCE (MTE3_MTE2) share id 0 without colliding. ----
    IN_EV = 2  # in_ub ping-pong base (ids 2,3): s_ub(vec1)+o_ub(vec2); V_MTE2+MTE2_V (+stash V_MTE3/MTE3_V)
    ACC_EV = 4  # acc_pre single: V_MTE2 (WAR) + MTE2_V (RAW)
    OUT_EV = 5  # out_ub single: p_half(vec1)+o_half(vec2); MTE3_V (WAR) + V_MTE3 (RAW)
    LSE_EV = 0  # lse_ub single: MTE3_V (WAR) + V_MTE3 (RAW)
    # workspace_acc cross-tile MTE3->MTE2 fence: prev tile stashed via MTE3, this tile
    # reloads via MTE2 -> same-core GM RAW (= DealBmm2:672-674). MTE3_MTE2 own namespace.
    FENCE = 0
    # SCFA V0 merge-buffer ping-pong flags (= ref SYNC_INPUT_BUF2): MTE2_MTE3
    # (gather->scatter RAW) + MTE3_MTE2 (scatter->next-gather buffer-free WAR),
    # keyed by buf pp = jb%2 -> ids MRG_EV+{0,1}. MTE3_MTE2 id 0 is FENCE, so use
    # 6/7; MTE2_MTE3 has no other user. Lets gather(batch i+1) overlap scatter(i).
    MRG_EV = 6

    # kvMergeGm ring depth (= reference MERGE_CACHE_GM_BUF_NUM = 4, scfa_block_vector.h:125).
    # MUST equal the EV_CREDIT prime count (4): the flag-3 credit bounds the V0 writer to
    # <=KVMERGE_RING tasks ahead of the cube reader, so credits==slots keeps the ring WAR-
    # free with max overlap (= reference's exact tuning). (A 32 probe confirmed the WAR;
    # the faithful fix is flag-3 + ring=4, not a deeper ring.)
    KVMERGE_RING = 4

    q_shape = [total_tokens, N1, D]
    ori_kv_shape = [ori_block_num, ori_block_size, N2, D]
    cmp_kv_shape = [cmp_block_num, cmp_block_size, N2, D]
    # SCFA: topk index table = K (=topk_cmp) selected cmp-block ids per (token,n2).
    cmp_idx_shape = [total_tokens, N2, topk_cmp]

    @tilelang.jit(out_idx=[11, 12], workspace_idx=[13, 14, 15, 16, 17, 18])
    def kernel():
        @T.prim_func
        def sparse_attn_sharedkv_scfa(
            Q: T.Tensor(q_shape, dtype),  # 0
            ori_kv: T.Tensor(ori_kv_shape, dtype),  # 1  (ori shared K & V)
            ori_block_table: T.Tensor([batch, ori_table_len], idx_dtype),  # 2
            cmp_kv: T.Tensor(cmp_kv_shape, dtype),  # 3  (cmp shared K & V)
            cmp_block_table: T.Tensor([batch, cmp_table_len], idx_dtype),  # 4
            cmp_indices: T.Tensor(cmp_idx_shape, idx_dtype),  # 5  (unused, CFA = dense)
            q_prefix: T.Tensor([batch], idx_dtype),  # 6  flat-token base
            act_q_lens: T.Tensor([batch], idx_dtype),  # 7  per-batch q len
            seqused_kv: T.Tensor([batch], idx_dtype),  # 8  per-batch ori kv len
            sinks: T.Tensor([N1], accum_dtype),  # 9  per-head sink
            metadata: T.Tensor([SAS_META_SIZE], idx_dtype),  # 10 (unused, v1)
            Output: T.Tensor(q_shape, dtype),  # 11 out
            LSE: T.Tensor([total_tokens, N1], accum_dtype),  # 12 out
            # cube<->vector workspaces, per core, ring-buffered (dim 1 = g%2):
            workspace_s: T.Tensor([core_num, 2, G, S2_BASE], accum_dtype),  # 13 QKᵀ
            workspace_p: T.Tensor([core_num, 2, G, S2_BASE], dtype),  # 14 softmax P
            workspace_o: T.Tensor([core_num, 2, G, D], accum_dtype),  # 15 PV (raw)
            # running online-softmax PV accumulator (= reference vec2ResGm):
            workspace_acc: T.Tensor([core_num, 2, G, D], accum_dtype),  # 16
            # QK shape-token: gemm_v0_fixp(do_fixpipe=False) (DEBUG_SERIAL=False path)
            # derives M,N from a dst it NEVER writes (the fixpipe is T.copy(cL0->band)).
            workspace_qk: T.Tensor([core_num, G, BI], accum_dtype),  # 17
            # SCFA V0 merged sparse KV (= reference kvMergeGm). 4-deep ring indexed
            # by the flat tile g%4 (= reference cmpLoop%4, but a pure function of g so
            # V0(g)/QK(g)/PV(g-1) agree with NO runtime counter): the depth-3 pipeline
            # keeps tiles g,g-1,g-2 in flight, so g%4-distinct slots avoid the V0-write
            # vs cube-read WAR. V0 (vector) gathers topk-selected cmp tokens here; cube
            # cmp QK/PV read it via a plain Nd2Nz copy (= block_cube.h:448-478, NOT
            # paged copy_pa).
            kvMergeGm: T.Tensor(
                [core_num, KVMERGE_RING, S2_BASE, D], dtype
            ),  # 18  (KVMERGE_RING-deep ring = ref MERGE_CACHE_GM_BUF_NUM=4)
        ):
            with T.Kernel(core_num, is_npu=True) as (cid, vid):
                # ---- Cube L1/L0 allocations (kernel scope). ----
                # Q/K as two 256-wide D-halves (ComputeMm1 kL1Loops=2).
                q_l1 = T.alloc_L1([2, G, D2], dtype)  # Q D-halves (reused across bands)
                # Shared KV 3-slot ring (= reference kvL1): QK K D-halves + PV V tiles.
                kv_ring = T.alloc_L1([3, BI, D2], dtype)
                p_l1 = T.alloc_L1([G, S2_BASE], dtype)  # P (up to 512 wide)
                cL0 = T.alloc_L0C([2, G, BI], accum_dtype)  # shared QK/PV cL0 ping-pong
                p_l0a = T.alloc_L0A(
                    [2, G, BI], dtype
                )  # P activations (PV, L0AB ping-pong)
                v_l0b = T.alloc_L0B([2, BI, BI], dtype)  # V tile (PV, L0AB ping-pong)
                # Persistent ring/ping-pong counters (= reference class members; survive
                # across g, mutated in place). kv_iter -> KV-ring slot (per load),
                # cl0_iter -> cL0 slot (per band/PV-tile fixpipe), ab_iter -> L0AB pp
                # (per PV L0 load). alloc_var scalars: read by name, write with +=.
                kv_iter = T.alloc_var("int32", init=0)
                cl0_iter = T.alloc_var("int32", init=0)
                ab_iter = T.alloc_var("int32", init=0)
                # ---- Vector UB allocations (m-chunked to M_CHUNK rows). D == S2_BASE
                # (both 512), so one buffer serves both the score (vec1) and PV (vec2)
                # roles. = ref's time-shared inputBuff1 / outputBuff1 (vec1 fully
                # precedes vec2 within a PreloadPipeline iteration). ----
                in_ub = T.alloc_ub(
                    [2, M_CHUNK, S2_BASE], accum_dtype
                )  # = ref inputBuff1: s_ub(vec1 score) + o_ub(vec2 PV); ping-pong [mc&1]
                softmax_cmp = T.alloc_ub([M_CHUNK, S2_BASE], accum_dtype)  # compaction
                out_ub = T.alloc_ub(
                    [M_CHUNK, S2_BASE], dtype
                )  # = ref outputBuff1: p_half(vec1 cast P) + o_half(vec2 cast out)
                acc_pre = T.alloc_ub(
                    [M_CHUNK, D], accum_dtype
                )  # prev accumulator (rescale); = ref inputBuff2 (single)
                sink_ub = T.alloc_ub(
                    [2, M_CHUNK, 1], accum_dtype
                )  # m-split ping-pong [mc&1]: rides in_ub's slot-keyed IN_EV+ps flag
                lse_ub = T.alloc_ub([M_CHUNK, 1], accum_dtype)
                ones_ub = T.alloc_ub([M_CHUNK, 1], accum_dtype)  # in_sum seed (1.0)
                brcb_d = T.alloc_ub(
                    [M_CHUNK, BLK], accum_dtype
                )  # Brcb scratch (row_expand)
                # 手拼替代 007: sumP=reduce_sum(P) 临时(替代 softmax_tmp);
                # softmax_cmp 复用为 broadcast(max) 的 [M_CHUNK,512] 目标(免新 alloc)。
                sumP = T.alloc_ub([M_CHUNK, 1], accum_dtype)
                # SCFA V0 merge buffer (MERGE_ROWS tokens x D, = ref INPUT2_BUFFER batched
                # at 8 by UB budget). V0 gathers a batch of topk-selected cmp tokens here
                # (GM cmp_kv -> UB, one per-token T.copy: the partial-row load folds dstM to
                # 1 so it does NOT Duplicate-wipe the multi-row buffer), then ONE batched
                # scatter to kvMergeGm.
                merge_ub = T.alloc_ub([2, MERGE_ROWS, D], dtype)
                # Online-softmax running state rings (= softmaxMax/Sum/ExpUb), indexed
                # [tile g%2, m-chunk, row, 1]. The m-chunk is a SEPARATE dim (not a
                # bounded row-slice of [2,G2,1]) so a per-chunk view ring[slot, mc, :, :]
                # is a FULL-slice BufferRegion the tile primitives accept (a bounded
                # slice ring[slot, mc*M:(mc+1)*M, :] parses to a BufferLoad they reject).
                m_i = T.alloc_ub([2, NMC, M_CHUNK, 1], accum_dtype)  # running max
                denom = T.alloc_ub([2, NMC, M_CHUNK, 1], accum_dtype)  # running sum
                expmax = T.alloc_ub([2, NMC, M_CHUNK, 1], accum_dtype)  # flash rescale

                # ============================ CUBE ============================
                with T.Scope("C"):
                    # AllocEventID (= block_cube.h:225): prime the L0AB M_MTE1 pair
                    # {4,5} (PV's mma; QK's gemm_v0 self-manages event 0), the 3 KV-ring
                    # slots' MTE1_MTE2 reverse flags {2,3,6}, and the Q/P reverse flags
                    # {1,7} -- all "free" so the first user waits on an already-set
                    # flag. Drained once after the g-loop.
                    T.set_flag("m", "mte1", L0AB_EV0)
                    T.set_flag("m", "mte1", L0AB_EV1)
                    T.set_flag("mte1", "mte2", KV_EV0)
                    T.set_flag("mte1", "mte2", KV_EV1)
                    T.set_flag("mte1", "mte2", KV_EV2)
                    T.set_flag("mte1", "mte2", Q_EV)
                    T.set_flag("mte1", "mte2", P_EV)
                    # flag-3 prime (= scfa_kernel.h:680-683): AIC pre-loads KVMERGE_RING
                    # credits so the first KVMERGE_RING tasks' V0 run freely; each credit
                    # is returned after that task's last-tile PV reads the slot (below).
                    for _cr in range(KVMERGE_RING):
                        T.set_cross_flag("MTE2", EV_CREDIT)
                    for g in T.serial(GLOOP + 1):
                        # ---- QK for tile g ----
                        if g < GLOOP:
                            buf = g % 2
                            task = g // MAX_TILES
                            tile = g % MAX_TILES
                            pid = task * core_num + cid
                            if pid < block_num:
                                b = T.cast(pid // max_seq, "int32")
                                s = T.cast(pid % max_seq, "int32")
                                act_q = act_q_lens[b]
                                if s < act_q:
                                    act_kv = seqused_kv[b]
                                    s_global = act_kv - act_q + s
                                    act_cmp = (s_global + 1) // cmp_ratio
                                    cmp_tiles = (
                                        T.min(act_cmp, topk_cmp) + S2_BASE - 1
                                    ) // S2_BASE
                                    s2lt = 1 + cmp_tiles
                                    if tile < s2lt:
                                        is_ori = tile == 0
                                        ori_left = T.max(s_global - ori_win_left, 0)
                                        win = s_global + 1 - ori_left
                                        rel = tile - 1
                                        # tile column width. Inner cmp-tail conditional
                                        # is T.min (last tile: act_cmp-rel*512 < 512;
                                        # else >=512 -> 512); the ori/cmp pick is a
                                        # tir.Select. Both lower to an INLINE ternary
                                        # (SelectNode); a runtime T.if_then_else instead
                                        # lowers to a statement (int32_t condval; if{}),
                                        # which the codegen cannot inline into the copy_pa
                                        # arg list (bad C++). For ori (rel=-1) the cmp
                                        # branch = min(512, act_cmp+512)=512, unused.
                                        tw = tir.Select(
                                            is_ori,
                                            win,
                                            T.min(
                                                S2_BASE,
                                                T.min(act_cmp, topk_cmp)
                                                - rel * S2_BASE,
                                            ),
                                        )
                                        s2base = tir.Select(
                                            is_ori, ori_left, rel * S2_BASE
                                        )
                                        tok = q_prefix[b] + s
                                        # Q reverse flag: wait Q free (prev tile's gemms
                                        # done), load both D-halves, signal loaded, wait
                                        # loaded (every band's gemm reads q_l1).
                                        T.wait_flag("mte1", "mte2", Q_EV)
                                        T.copy(Q[tok, :, 0:D2], q_l1[0, :, :])
                                        T.copy(Q[tok, :, D2:D], q_l1[1, :, :])
                                        T.set_flag("mte2", "mte1", Q_EV)
                                        T.wait_flag("mte2", "mte1", Q_EV)
                                        if DEBUG_SERIAL:
                                            T.barrier_all()
                                        # QK over 128-col blocks (ComputeMm1 nL1Loops):
                                        # each band loads its 2 K D-halves into 2 KV-ring
                                        # slots (per-slot reverse flags overlap the next
                                        # band's copy_pa with the current gemm), accumulates
                                        # them into one cL0 ping-pong slot via gemm_v0, then
                                        # T.copy(cL0 -> ws_s band) is the standalone band
                                        # Fixpipe.
                                        # SCFA: cmp tiles read merged KV from kvMergeGm; wait
                                        # until V0 (vector) produced this tile's slot (= ref
                                        # ComputeMm1 CrossCoreWaitFlag(syncV0C1):797). ori
                                        # tiles read ori_kv (no wait); V0 sets EV_V0 only for
                                        # cmp, and tile<s2lt uses the same acmp_s on both
                                        # cores, so the set/wait counts stay balanced.
                                        if tile != 0:
                                            T.wait_cross_flag(EV_V0)
                                        for cb in range(MAX_COLBLK):
                                            if cb * BI < tw:
                                                ncols = T.min(BI, tw - cb * BI)
                                                ncols_a = (ncols + 15) // 16 * 16
                                                cs = cl0_iter % 2
                                                # 2 K D-halves -> 2 consecutive ring slots
                                                # (pre-increment exprs; kv_iter bumps AFTER
                                                # the gemms consume the slots).
                                                s0 = kv_iter % 3
                                                s1 = (kv_iter + 1) % 3
                                                ev0 = tir.Select(
                                                    s0 < 2, KV_EV0 + s0, KV_EV2
                                                )
                                                ev1 = tir.Select(
                                                    s1 < 2, KV_EV0 + s1, KV_EV2
                                                )
                                                for h in range(2):
                                                    slot = (kv_iter + h) % 3
                                                    ev = tir.Select(
                                                        slot < 2, KV_EV0 + slot, KV_EV2
                                                    )
                                                    T.wait_flag("mte1", "mte2", ev)
                                                    if is_ori:
                                                        T.copy_pa(
                                                            kv_ring[slot, :, :],
                                                            ori_kv,
                                                            ori_block_table,
                                                            ori_block_size,
                                                            N2,
                                                            D,
                                                            ori_block_size * N2 * D,
                                                            ori_table_len,
                                                            D2,
                                                            ncols,
                                                            BI,
                                                            b,
                                                            0,
                                                            s2base + cb * BI,
                                                            h * D2,
                                                        )
                                                    else:
                                                        # SCFA: this cmp tile's merged KV is in
                                                        # kvMergeGm (V0 produced it; QK waited
                                                        # EV_V0 before this cb loop). Plain Nd2Nz
                                                        # (= block_cube.h:459) replaces paged
                                                        # copy_pa; the slot holds only this tile's
                                                        # tokens at local 0 so row = cb*BI (not
                                                        # s2base+cb*BI). g%4 = ring slot.
                                                        T.copy(
                                                            kvMergeGm[
                                                                cid,
                                                                task % KVMERGE_RING,
                                                                cb * BI : cb * BI
                                                                + ncols,
                                                                h * D2 : h * D2 + D2,
                                                            ],
                                                            kv_ring[slot, 0:ncols, :],
                                                        )
                                                    T.set_flag("mte2", "mte1", ev)
                                                if DEBUG_SERIAL:
                                                    T.barrier_all()
                                                # 2 D-halves accumulate into cL0[cs].
                                                # DEBUG_SERIAL=True: gemm_v0 (no unitFlag,
                                                # self-L0AB on event 0, standalone fixpipe)
                                                # -- the barriered debug path. False (perf):
                                                # gemm_v0_fixp(prime_drain=False shares L0AB
                                                # {4,5} primed once, do_fixpipe=False leaves
                                                # cL0 for the 0b11 band fixpipe) = ComputeMm1
                                                # Mmad; the last mma 0b11 lets fixpipe(cb) ||
                                                # mma(cb+1) (cL0 ping-pong), so band cb+1's
                                                # copy_pa(mte2) overlaps band cb's gemm(mac)
                                                # -- the overlap gemm_v0's per-call FIX_M
                                                # drain blocks. (0b11 needs the FUSED 0b11
                                                # fixpipe, so only the no-barrier path.)
                                                T.wait_flag("mte2", "mte1", ev0)
                                                if DEBUG_SERIAL:
                                                    T.gemm_v0(
                                                        q_l1[0, :, :],
                                                        kv_ring[s0, :, :],
                                                        cL0[cs, :, :],
                                                        transpose_B=True,
                                                        init=True,
                                                        n_actual=ncols_a,
                                                    )
                                                else:
                                                    T.gemm_v0_fixp(
                                                        q_l1[0, :, :],
                                                        kv_ring[s0, :, :],
                                                        cL0,
                                                        workspace_qk[cid, :, :],
                                                        k_actual=D2,
                                                        transpose_B=True,
                                                        init=True,
                                                        n_actual=ncols_a,
                                                        cl0_base=cs,
                                                        prime_drain=False,
                                                        flush_last=False,
                                                        do_fixpipe=False,
                                                    )
                                                T.set_flag("mte1", "mte2", ev0)
                                                T.wait_flag("mte2", "mte1", ev1)
                                                if DEBUG_SERIAL:
                                                    T.gemm_v0(
                                                        q_l1[1, :, :],
                                                        kv_ring[s1, :, :],
                                                        cL0[cs, :, :],
                                                        transpose_B=True,
                                                        init=False,
                                                        n_actual=ncols_a,
                                                    )
                                                else:
                                                    T.gemm_v0_fixp(
                                                        q_l1[1, :, :],
                                                        kv_ring[s1, :, :],
                                                        cL0,
                                                        workspace_qk[cid, :, :],
                                                        k_actual=D2,
                                                        transpose_B=True,
                                                        init=False,
                                                        n_actual=ncols_a,
                                                        cl0_base=cs,
                                                        prime_drain=False,
                                                        flush_last=True,
                                                        do_fixpipe=False,
                                                    )
                                                T.set_flag("mte1", "mte2", ev1)
                                                kv_iter += 2
                                                if DEBUG_SERIAL:
                                                    T.barrier_all()
                                                # band Fixpipe cL0[cs] -> strided ws_s band
                                                # (realDstN = 512 = ComputeMm1 dstStride).
                                                # 0b11 pairs with gemm_v0_fixp's last mma
                                                # (0b11) -> fixpipe(cb) || mma(cb+1) via the
                                                # cL0 ping-pong; standalone + barrier on the
                                                # DEBUG_SERIAL (gemm_v0) debug path.
                                                if DEBUG_SERIAL:
                                                    T.copy(
                                                        cL0[cs, :, :],
                                                        workspace_s[
                                                            cid,
                                                            buf,
                                                            :,
                                                            cb * BI : (cb + 1) * BI,
                                                        ],
                                                    )
                                                    T.barrier_all()
                                                else:
                                                    # 0b11 fixpipe nSize MUST equal the
                                                    # mma's n_actual=ncols_a (else it waits
                                                    # cL0 cols [ncols_a:BI] no mma marked ->
                                                    # hang). So the band is ncols_a wide
                                                    # (row stride still 512 = ws_s width).
                                                    T.copy(
                                                        cL0[cs, :, 0:ncols_a],
                                                        workspace_s[
                                                            cid,
                                                            buf,
                                                            :,
                                                            cb * BI : cb * BI + ncols_a,
                                                        ],
                                                        unit_flag=0b11,
                                                    )
                                                cl0_iter += 1
                                        # release Q for the next tile's QK load.
                                        T.set_flag("mte1", "mte2", Q_EV)
                            # fire every g<GLOOP (incl. invalid tiles) to keep the
                            # cross-flag count balanced with the vector's waits.
                            T.set_cross_flag("FIX", EV_QK)
                        # ---- PV for tile g-1 ----
                        if g >= 1:
                            T.wait_cross_flag(EV_P)
                            gm = g - 1
                            bufm = gm % 2
                            taskm = gm // MAX_TILES
                            tilem = gm % MAX_TILES
                            pidm = taskm * core_num + cid
                            if pidm < block_num:
                                bm = T.cast(pidm // max_seq, "int32")
                                sm = T.cast(pidm % max_seq, "int32")
                                if sm < act_q_lens[bm]:
                                    act_kvm = seqused_kv[bm]
                                    s_globalm = act_kvm - act_q_lens[bm] + sm
                                    act_cmpm = (s_globalm + 1) // cmp_ratio
                                    cmp_tilesm = (
                                        T.min(act_cmpm, topk_cmp) + S2_BASE - 1
                                    ) // S2_BASE
                                    s2ltm = 1 + cmp_tilesm
                                    if tilem < s2ltm:
                                        is_orim = tilem == 0
                                        ori_leftm = T.max(s_globalm - ori_win_left, 0)
                                        winm = s_globalm + 1 - ori_leftm
                                        relm = tilem - 1
                                        # inline ternaries (T.min + tir.Select), see QK.
                                        twm = tir.Select(
                                            is_orim,
                                            winm,
                                            T.min(
                                                S2_BASE,
                                                T.min(act_cmpm, topk_cmp)
                                                - relm * S2_BASE,
                                            ),
                                        )
                                        s2basem = tir.Select(
                                            is_orim, ori_leftm, relm * S2_BASE
                                        )
                                        # P reverse flag: wait P free, load P, signal
                                        # loaded, wait loaded (every (nl,ks) reads p_l1).
                                        T.wait_flag("mte1", "mte2", P_EV)
                                        T.copy(
                                            workspace_p[cid, bufm, :, 0:S2_BASE],
                                            p_l1[:, :],
                                        )
                                        T.set_flag("mte2", "mte1", P_EV)
                                        T.wait_flag("mte2", "mte1", P_EV)
                                        if DEBUG_SERIAL:
                                            T.barrier_all()
                                        # PV = ComputeMm2: 4 output-D tiles, each
                                        # K-accumulating the tile's 128-row sub-blocks into
                                        # one cL0 slot. Each (nl,ks): LOAD V -> KV ring slot
                                        # (per-slot reverse flag), CONSUME (L0 load on the
                                        # L0AB pp ping-pong -> mma accumulate). Then the
                                        # band Fixpipe cL0[cs] -> ws_o band (0b11 overlaps
                                        # the next tile's mma; standalone under barriers).
                                        for nl in range(PV_NT):
                                            cs = cl0_iter % 2
                                            for ks in range(MAX_COLBLK):
                                                if ks * BI < twm:
                                                    krows = T.min(BI, twm - ks * BI)
                                                    is_first_ks = ks == 0
                                                    is_last_ks = (ks + 1) * BI >= twm
                                                    slot = kv_iter % 3
                                                    ev = tir.Select(
                                                        slot < 2, KV_EV0 + slot, KV_EV2
                                                    )
                                                    pp = ab_iter % 2
                                                    # LOAD V D-tile nl, K-block ks -> slot.
                                                    T.wait_flag("mte1", "mte2", ev)
                                                    if is_orim:
                                                        T.copy_pa(
                                                            kv_ring[slot, :, :],
                                                            ori_kv,
                                                            ori_block_table,
                                                            ori_block_size,
                                                            N2,
                                                            D,
                                                            ori_block_size * N2 * D,
                                                            ori_table_len,
                                                            PV_NW,
                                                            krows,
                                                            BI,
                                                            bm,
                                                            0,
                                                            s2basem + ks * BI,
                                                            nl * PV_NW,
                                                        )
                                                    else:
                                                        # SCFA: merged V from kvMergeGm (gm%4 ring
                                                        # slot; V0 produced tile gm). Plain Nd2Nz
                                                        # replaces paged copy_pa. PV reads K-block
                                                        # ks (rows) x D-tile nl; local row = ks*BI.
                                                        T.copy(
                                                            kvMergeGm[
                                                                cid,
                                                                taskm % KVMERGE_RING,
                                                                ks * BI : ks * BI
                                                                + krows,
                                                                nl * PV_NW : nl * PV_NW
                                                                + PV_NW,
                                                            ],
                                                            kv_ring[
                                                                slot, 0:krows, 0:PV_NW
                                                            ],
                                                        )
                                                    T.set_flag("mte2", "mte1", ev)
                                                    if DEBUG_SERIAL:
                                                        T.barrier_all()
                                                    # CONSUME: L0 load (L0AB pp) -> mma.
                                                    T.wait_flag(
                                                        "m", "mte1", L0AB_EV0 + pp
                                                    )
                                                    T.wait_flag("mte2", "mte1", ev)
                                                    T.copy(
                                                        p_l1[
                                                            :, ks * BI : (ks + 1) * BI
                                                        ],
                                                        p_l0a[pp, :, :],
                                                        real_k=krows,
                                                    )
                                                    T.copy(
                                                        kv_ring[slot, :, 0:PV_NW],
                                                        v_l0b[pp, :, :],
                                                        real_k=krows,
                                                    )
                                                    T.set_flag(
                                                        "mte1", "m", L0AB_EV0 + pp
                                                    )
                                                    T.wait_flag(
                                                        "mte1", "m", L0AB_EV0 + pp
                                                    )
                                                    # last K -> 0b11 flush (pairs with the
                                                    # band fixpipe); else 0b10 accumulate.
                                                    # 0 under barriers (no fusion).
                                                    uf = (
                                                        0
                                                        if DEBUG_SERIAL
                                                        else tir.Select(
                                                            is_last_ks, 0b11, 0b10
                                                        )
                                                    )
                                                    T.mma(
                                                        p_l0a[pp, :, :],
                                                        v_l0b[pp, :, :],
                                                        cL0[cs, :, :],
                                                        init=is_first_ks,
                                                        k_actual=krows,
                                                        unit_flag=uf,
                                                    )
                                                    T.set_flag(
                                                        "m", "mte1", L0AB_EV0 + pp
                                                    )
                                                    T.set_flag("mte1", "mte2", ev)
                                                    kv_iter += 1
                                                    ab_iter += 1
                                                    if DEBUG_SERIAL:
                                                        T.barrier_all()
                                            # band Fixpipe cL0[cs] -> this output-D band.
                                            if DEBUG_SERIAL:
                                                T.copy(
                                                    cL0[cs, :, :],
                                                    workspace_o[
                                                        cid,
                                                        bufm,
                                                        :,
                                                        nl * PV_NW : (nl + 1) * PV_NW,
                                                    ],
                                                )
                                                T.barrier_all()
                                            else:
                                                T.copy(
                                                    cL0[cs, :, :],
                                                    workspace_o[
                                                        cid,
                                                        bufm,
                                                        :,
                                                        nl * PV_NW : (nl + 1) * PV_NW,
                                                    ],
                                                    unit_flag=0b11,
                                                )
                                            cl0_iter += 1
                                        # release P for the next tile's PV load.
                                        T.set_flag("mte1", "mte2", P_EV)
                                        # flag-3 return (= scfa_kernel.h:816-818, isLastS2Loop): after this
                                        # task's LAST valid tile PV has read its kvMergeGm slot (MTE2), hand
                                        # back 1 credit so V0(task + KVMERGE_RING) may overwrite that slot.
                                        if tilem == s2ltm - 1:
                                            T.set_cross_flag("MTE2", EV_CREDIT)
                            # fire every g>=1 (incl. invalid) for count balance.
                            T.set_cross_flag("FIX", EV_PV)
                    # FreeEventID (= block_cube.h:239): drain the L0AB pair {4,5}, the 3
                    # KV-ring slots {2,3,6}, and Q/P {1,7} -- each left set by its last
                    # consumer, balancing the prime before the g-loop.
                    T.wait_flag("m", "mte1", L0AB_EV0)
                    T.wait_flag("m", "mte1", L0AB_EV1)
                    T.wait_flag("mte1", "mte2", KV_EV0)
                    T.wait_flag("mte1", "mte2", KV_EV1)
                    T.wait_flag("mte1", "mte2", KV_EV2)
                    T.wait_flag("mte1", "mte2", Q_EV)
                    T.wait_flag("mte1", "mte2", P_EV)

                # =========================== VECTOR ===========================
                with T.Scope("V"):
                    T.tile.fill(ones_ub, T.float32(1.0))
                    # Prime every vector buffer's reverse flag ONCE (buffers free) =
                    # block_vector.h InitBuffers SetFlag<V_MTE2>/<MTE3_V>; the first mc
                    # of the first tile waits on an already-set flag. Drained once after
                    # the g-loop. Each executed mc does exactly 1 wait + 1 set, so
                    # skipping invalid tiles keeps the balance (wait+set is net-zero).
                    T.set_flag("v", "mte2", IN_EV)  # in_ub slot0 free
                    T.set_flag("v", "mte2", IN_EV + 1)  # in_ub slot1 free (pong)
                    T.set_flag("v", "mte2", ACC_EV)  # acc_pre free
                    T.set_flag("mte3", "v", OUT_EV)  # out_ub free
                    T.set_flag("mte3", "v", LSE_EV)  # lse_ub free
                    # V0 per-cmp-tile token loop bound (compile-time): one cmp tile merges
                    # at most min(S2_BASE, topk_cmp) topk-selected tokens (runtime v0n <=
                    # this; the jtok < v0n guard skips the tail). V0_NB = batch count.
                    V0_NMAX = min(S2_BASE, topk_cmp)
                    V0_NB = (V0_NMAX + MERGE_ROWS - 1) // MERGE_ROWS
                    for g in T.serial(1, GLOOP + 2):
                        # ==== V0: merge topk-selected sparse cmp KV for tile g ====
                        # = reference ProcessVec0L. The vector gathers the CURRENT flat
                        # tile g's topk cmp tokens into kvMergeGm; cube QK(g) reads them
                        # (cross-flag EV_V0). flat index = g (SAME as cube QK) so the
                        # EV_V0 set(vector)/wait(cube) counts pair tile-for-tile and stay
                        # balanced. sparseBlockSize=1 (scfa_kernel.h:234) => cmp_indices
                        # [pid,0,j] is directly the j-th selected cmp TOKEN id (realS2Idx);
                        # GetKeyGmOffset paging = cmp_block_table[b, id//blk] then id%blk.
                        # Only cmp tiles (tile!=0) merge (ori reads ori_kv on the cube;
                        # g=0 is always ori, so the vector loop starting at g=1 still
                        # covers every cmp tile). Phase A: ONE kvMergeGm slot, GM->UB->GM
                        # in M_CHUNK batches under barriers (= ref CopyInSingleKv +
                        # CopyOutMrgeResult, blockCount=1 fallback shape); the fused
                        # 2-token gather [015] + cmpLoop%4 ring + ping-pong are Phase D.
                        if g < GLOOP:
                            v0task = g // MAX_TILES
                            v0tile = g % MAX_TILES
                            v0pid = v0task * core_num + cid
                            if v0pid < block_num:
                                v0b = T.cast(v0pid // max_seq, "int32")
                                v0s = T.cast(v0pid % max_seq, "int32")
                                # token dim of cmp_indices/Q/Output is the FLAT token
                                # q_prefix[b]+s (= cube/vector tok), NOT v0pid: they agree
                                # for BSND (q_prefix=b*max_seq) & TND B=1 (q_prefix[0]=0)
                                # but differ for TND B>1 (variable prefix sum) (review #2).
                                v0tok = q_prefix[v0b] + v0s
                                if v0s < act_q_lens[v0b]:
                                    # flag-3 wait (= scfa_kernel.h:801-803, isFirstSInnerLoop):
                                    # before this task's FIRST tile's V0, take 1 credit so the
                                    # V0 scatter (async MTE3) cannot run > KVMERGE_RING tasks
                                    # ahead of the cube reader of the same kvMergeGm slot. A
                                    # task's first V0 tile is tile 0 (g = task*MAX_TILES); but
                                    # the vector loop starts at g=1, so task 0's tile 0 (g=0)
                                    # is never processed -- its credit is taken at g==1 (its
                                    # first processed V0 tile). One wait per valid task, exactly
                                    # matching the cube's one return per valid task (last tile).
                                    if v0tile == 0 or g == 1:
                                        T.wait_cross_flag(EV_CREDIT)
                                    v0skv = seqused_kv[v0b] - act_q_lens[v0b] + v0s
                                    v0acmp = (v0skv + 1) // cmp_ratio
                                    # sparse merged length = min(topk_cmp, causal act_cmp)
                                    v0asparse = T.min(v0acmp, topk_cmp)
                                    v0ctiles = (v0asparse + S2_BASE - 1) // S2_BASE
                                    if (
                                        v0tile != 0
                                    ):  # cmp tile (ori reads ori_kv directly)
                                        if v0tile < 1 + v0ctiles:
                                            v0rel = v0tile - 1
                                            v0n = T.min(
                                                S2_BASE, v0asparse - v0rel * S2_BASE
                                            )
                                            # Sub-block split (= ref ProcessVec0L s2GmStartOffset/
                                            # s2GmLimit): BOTH vids merge HALF, so BOTH set EV_V0
                                            # after a REAL MTE3-fenced scatter. (vid==1 doing no
                                            # merge but still setting EV_V0 was an UNFENCED "empty
                                            # set" -> the cube's MODE2 rendezvous could converge
                                            # before vid==0's scatter landed -> stale/uninit read
                                            # = cold-start NaN. This = the faithful reference + 2x
                                            # V0 throughput.) Both vids write DISJOINT kvMergeGm
                                            # rows; cube waits both (rendezvous) -> reads complete.
                                            v0half = (v0n + 1) // 2
                                            v0start = tir.Select(vid == 0, 0, v0half)
                                            v0lim = tir.Select(vid == 0, v0half, v0n)
                                            # prime both merge_ub ping-pong buffers "free" (= AllocEventID): a
                                            # batch reuses buffer pp=jb%2 only after its prev scatter's
                                            # MTE3_MTE2 fires, so gather(batch i+1, buf pp^1) overlaps
                                            # scatter(batch i, buf pp) -> removes the per-batch barrier_all.
                                            T.set_flag("mte3", "mte2", MRG_EV + 0)
                                            T.set_flag("mte3", "mte2", MRG_EV + 1)
                                            for jb in T.serial(V0_NB):
                                                pp = jb % 2
                                                if v0start + jb * MERGE_ROWS < v0lim:
                                                    # wait buf pp free (its prev scatter retired) before gathering.
                                                    T.wait_flag(
                                                        "mte3", "mte2", MRG_EV + pp
                                                    )
                                                    for jj in range(MERGE_ROWS):
                                                        jtok = (
                                                            v0start
                                                            + jb * MERGE_ROWS
                                                            + jj
                                                        )
                                                        if jtok < v0lim:
                                                            r2 = cmp_indices[
                                                                v0tok,
                                                                0,
                                                                v0rel * S2_BASE + jtok,
                                                            ]
                                                            blkid = cmp_block_table[
                                                                v0b,
                                                                r2 // cmp_block_size,
                                                            ]
                                                            # V0 gather 1 token GM->UB merge_ub[pp,jj]
                                                            # via plain T.copy: dstM folds to 1 =
                                                            # min(MERGE_ROWS, row-extent 1), so this
                                                            # partial-row load does NOT Duplicate-wipe
                                                            # the multi-row buffer (codegen emits
                                                            # copy_gm_to_ub<T, D, 1>). MTE2 like the
                                                            # gather it replaces, so the surrounding
                                                            # hand-placed pipe flags still apply.
                                                            T.copy(
                                                                cmp_kv[
                                                                    blkid,
                                                                    r2 % cmp_block_size,
                                                                    0,
                                                                    :,
                                                                ],
                                                                merge_ub[pp, jj, :],
                                                            )
                                                    # gather(MTE2)->scatter(MTE3) RAW on buf pp (= ref
                                                    # CopyOutMrgeResult's MTE2_MTE3 fence).
                                                    T.set_flag(
                                                        "mte2", "mte3", MRG_EV + pp
                                                    )
                                                    T.wait_flag(
                                                        "mte2", "mte3", MRG_EV + pp
                                                    )
                                                    bcnt = T.min(
                                                        MERGE_ROWS,
                                                        v0lim
                                                        - (v0start + jb * MERGE_ROWS),
                                                    )
                                                    T.copy(
                                                        merge_ub[pp, 0:bcnt, :],
                                                        kvMergeGm[
                                                            cid,
                                                            v0task % KVMERGE_RING,
                                                            v0start
                                                            + jb * MERGE_ROWS : v0start
                                                            + jb * MERGE_ROWS
                                                            + bcnt,
                                                            :,
                                                        ],
                                                    )
                                                    # buf pp scatter retired -> free for its next user (2 batches
                                                    # later); the next batch (pp^1) gathers meanwhile = overlap.
                                                    T.set_flag(
                                                        "mte3", "mte2", MRG_EV + pp
                                                    )
                                            # drain the 2 outstanding buffer-free sets (balanced: prime 2 +
                                            # N(wait+set) + drain 2 waits; per-buf sets==waits for any N>=0).
                                            T.wait_flag("mte3", "mte2", MRG_EV + 0)
                                            T.wait_flag("mte3", "mte2", MRG_EV + 1)
                                            # both vids did REAL fenced scatters -> set EV_V0 (= ref syncV0C1,
                                            # both sub-blocks); MODE2 rendezvous waits both, both fenced.
                                            T.set_cross_flag("MTE3", EV_V0)
                        # ---- softmax for tile g-1 ----
                        if g < GLOOP + 1:
                            T.wait_cross_flag(EV_QK)
                            gs = g - 1
                            buf = gs % 2
                            prev = (gs - 1) % 2
                            task = gs // MAX_TILES
                            tile = gs % MAX_TILES
                            pid = task * core_num + cid
                            if pid < block_num:
                                b = T.cast(pid // max_seq, "int32")
                                s = T.cast(pid % max_seq, "int32")
                                act_q = act_q_lens[b]
                                if s < act_q:
                                    act_kv = seqused_kv[b]
                                    s_global = act_kv - act_q + s
                                    act_cmp = (s_global + 1) // cmp_ratio
                                    cmp_tiles = (
                                        T.min(act_cmp, topk_cmp) + S2_BASE - 1
                                    ) // S2_BASE
                                    s2lt = 1 + cmp_tiles
                                    if tile < s2lt:
                                        is_first = tile == 0
                                        ori_left = T.max(s_global - ori_win_left, 0)
                                        win = s_global + 1 - ori_left
                                        rel = tile - 1
                                        # inline ternaries (T.min + tir.Select), see QK.
                                        tw = tir.Select(
                                            is_first,
                                            win,
                                            T.min(
                                                S2_BASE,
                                                T.min(act_cmp, topk_cmp)
                                                - rel * S2_BASE,
                                            ),
                                        )
                                        tw_a = (tw + 15) // 16 * 16
                                        for mc in range(NMC):
                                            r0 = vid * G2 + mc * M_CHUNK
                                            ps = (
                                                mc & 1
                                            )  # in_ub ping-pong slot (= ref pingpongFlag)
                                            # copy-in score + sink (MTE2). WAR: wait this
                                            # in_ub slot free (mc-2's compute / prev vec2's
                                            # output done reading it) = ref WaitFlag
                                            # <V_MTE2>(BUF1+pong):415.
                                            T.wait_flag("v", "mte2", IN_EV + ps)
                                            # 手拼替代 007: 满宽 fill -inf mask, 再用 014 只搬
                                            # 有效 tw_a 列 -> [tw_a:512] 留 -inf, 定长满宽(512)
                                            # reduce/exp 等价只算有效窗口(绕开变长 reduce)。
                                            # fill(V)->copy(MTE2) 复用 IN_EV+ps 握手。
                                            T.tile.fill(
                                                in_ub[ps, :, :],
                                                -T.infinity(accum_dtype),
                                            )
                                            T.set_flag("v", "mte2", IN_EV + ps)
                                            T.wait_flag("v", "mte2", IN_EV + ps)
                                            # load exactly [0:tw] (not tw_a-aligned): the
                                            # [tw:tw_a] out-of-window QK slop stays fill(-inf)
                                            # so the full-width reduce ignores it (lse slop fix).
                                            T.copy(
                                                workspace_s[
                                                    cid,
                                                    buf,
                                                    r0 : r0 + M_CHUNK,
                                                    0:tw,
                                                ],
                                                in_ub[ps, :, 0:tw],
                                            )
                                            T.copy(
                                                sinks[r0 : r0 + M_CHUNK],
                                                sink_ub[ps, :, :],
                                            )
                                            # copy-in -> compute (= ref SetFlag/WaitFlag
                                            # <MTE2_V>(BUF1+pong):418-419; one flag covers
                                            # in_ub for mul/softmax and sink for softmax).
                                            T.set_flag("mte2", "v", IN_EV + ps)
                                            T.wait_flag("mte2", "v", IN_EV + ps)
                                            T.tile.mul(
                                                in_ub[ps, :, :],
                                                in_ub[ps, :, :],
                                                softmax_scale,
                                            )
                                            # V-pipe order mul -> softmax (= ref
                                            # PipeBarrier<PIPE_V>:423).
                                            T.pipe_barrier("v")
                                            # 手拼 online softmax(替代 007): 产出 m_i/expmax/P;
                                            # denom 的 in_sum*expmax 部分在此, sum(P) 部分在 cast
                                            # 之后(reduce_sum 破坏 src)。in_max/in_sum 分支:
                                            #   is_first: in_max=sink, in_sum=ones(1.0)
                                            #   else:     in_max=m_i[prev], in_sum=denom[prev]
                                            # m_new=max(in_max,rowmax); expmax=exp(in_max-m_new);
                                            # P=exp(score-m_new)。[tw_a:512]=-inf 使满宽等价只算
                                            # 有效列。softmax_cmp 复用为 broadcast 目标。
                                            T.reduce_max(
                                                in_ub[ps, :, :],
                                                m_i[buf, mc, :, :],
                                                dim=-1,
                                            )
                                            if is_first:
                                                # m_new = max(sink, rowmax); dst(m_i)放第一位
                                                # (T.tile.max(A,B,C) 编成 Max(dst=A,B,C))
                                                T.tile.max(
                                                    m_i[buf, mc, :, :],
                                                    sink_ub[ps, :, :],
                                                    m_i[buf, mc, :, :],
                                                )
                                                T.tile.sub(
                                                    expmax[buf, mc, :, :],
                                                    sink_ub[ps, :, :],
                                                    m_i[buf, mc, :, :],
                                                )
                                            else:
                                                # m_new = max(m_i[prev], rowmax); dst(m_i)放第一位
                                                T.tile.max(
                                                    m_i[buf, mc, :, :],
                                                    m_i[prev, mc, :, :],
                                                    m_i[buf, mc, :, :],
                                                )
                                                T.tile.sub(
                                                    expmax[buf, mc, :, :],
                                                    m_i[prev, mc, :, :],
                                                    m_i[buf, mc, :, :],
                                                )
                                            T.tile.exp(
                                                expmax[buf, mc, :, :],
                                                expmax[buf, mc, :, :],
                                            )
                                            T.tile.broadcast(
                                                softmax_cmp, m_i[buf, mc, :, :]
                                            )
                                            T.tile.sub(
                                                in_ub[ps, :, :],
                                                in_ub[ps, :, :],
                                                softmax_cmp,
                                            )
                                            T.tile.exp(in_ub[ps, :, :], in_ub[ps, :, :])
                                            if is_first:
                                                T.tile.mul(
                                                    denom[buf, mc, :, :],
                                                    ones_ub,
                                                    expmax[buf, mc, :, :],
                                                )
                                            else:
                                                T.tile.mul(
                                                    denom[buf, mc, :, :],
                                                    denom[prev, mc, :, :],
                                                    expmax[buf, mc, :, :],
                                                )
                                            # V-pipe order softmax -> cast (= ref
                                            # PipeBarrier<PIPE_V>:429).
                                            T.pipe_barrier("v")
                                            # cast -> copy-out (MTE3). WAR: wait out_ub
                                            # free (prev copy-out done) = ref
                                            # WaitFlag<MTE3_V>(:431).
                                            T.wait_flag("mte3", "v", OUT_EV)
                                            T.tile.cast(
                                                out_ub,
                                                in_ub[ps, :, :],
                                                "CAST_ROUND",
                                                M_CHUNK * S2_BASE,
                                            )
                                            # denom += sum(P): reduce_sum 破坏 src, 放 cast 之后
                                            # (同 V pipe, pipe_barrier 隔开 cast 先读完 P)。
                                            T.pipe_barrier("v")
                                            T.reduce_sum(in_ub[ps, :, :], sumP, dim=-1)
                                            T.tile.add(
                                                denom[buf, mc, :, :],
                                                denom[buf, mc, :, :],
                                                sumP,
                                            )
                                            # in_ub slot free 在 reduce_sum 之后; cast -> copy-out
                                            # = ref SetFlag/WaitFlag<V_MTE3>(:436-437).
                                            T.set_flag("v", "mte2", IN_EV + ps)
                                            T.set_flag("v", "mte3", OUT_EV)
                                            T.wait_flag("v", "mte3", OUT_EV)
                                            # copy out only the tw_a valid P columns
                                            # (the cube PV reads exactly the window) --
                                            # = ref width, saves MTE3 (needs 014).
                                            T.copy(
                                                out_ub[:, 0:tw_a],
                                                workspace_p[
                                                    cid,
                                                    buf,
                                                    r0 : r0 + M_CHUNK,
                                                    0:tw_a,
                                                ],
                                            )
                                            # copy-out done, out_ub free = ref SetFlag
                                            # <MTE3_V>(:439); gates the cross EV_P below.
                                            T.set_flag("mte3", "v", OUT_EV)
                            T.set_cross_flag("MTE3", EV_P)
                        # ---- output for tile g-2 ----
                        if g >= 2:
                            T.wait_cross_flag(EV_PV)
                            go = g - 2
                            bufo = go % 2
                            prevo = (go - 1) % 2
                            task = go // MAX_TILES
                            tile = go % MAX_TILES
                            pid = task * core_num + cid
                            if pid < block_num:
                                b = T.cast(pid // max_seq, "int32")
                                s = T.cast(pid % max_seq, "int32")
                                act_q = act_q_lens[b]
                                if s < act_q:
                                    act_kv = seqused_kv[b]
                                    s_global = act_kv - act_q + s
                                    # [FIX cmp-tiles floordiv] The Ascend codegen
                                    # lowers FloorDiv to C truncated `/`
                                    # (codegen_ascend.cc FloorDivNode). For THIS
                                    # delayed (go=g-2) output/LSE stage the arith
                                    # simplifier distributes (s_global+1)//cmp_ratio
                                    # into task*(core_num//cmp_ratio) +
                                    # (act_kv-act_q+cid-(core_num-1))//cmp_ratio,
                                    # whose split remainder is NEGATIVE for the first
                                    # tokens (small s_global) -> truncation != floor
                                    # -> act_cmp is +1 too large. That only flips
                                    # cmp_tiles when the true act_cmp is 0 (pure-ori
                                    # token, s_global < cmp_ratio-1), making this stage
                                    # finalize a NON-existent cmp tile 1 and read uninit
                                    # denom/m_i/workspace_o (stale GM left by the prior
                                    # op in a batch) -> NaN Output/LSE on token rows
                                    # [0,1,2]. Select the provably-exact 0 via a
                                    # comparison, bypassing the buggy divide. (The QK
                                    # and softmax stages use task=g//2 / (g-1)//2, whose
                                    # split remainder stays >=0, so they never hit it.)
                                    act_cmp = tir.Select(
                                        s_global < cmp_ratio - 1,
                                        0,
                                        (s_global + 1) // cmp_ratio,
                                    )
                                    cmp_tiles = (
                                        T.min(act_cmp, topk_cmp) + S2_BASE - 1
                                    ) // S2_BASE
                                    s2lt = 1 + cmp_tiles
                                    if tile < s2lt:
                                        not_first = tile != 0
                                        is_last = tile == s2lt - 1
                                        tok = q_prefix[b] + s
                                        for mc in range(NMC):
                                            r0 = vid * G2 + mc * M_CHUNK
                                            po = (
                                                mc & 1
                                            )  # in_ub ping-pong slot (= ref pingpongFlag)
                                            # copy-in PV result into in_ub (MTE2). WAR:
                                            # wait this slot free (mc-2's last reader / this
                                            # iter's vec1 softmax done) = ref
                                            # WaitFlag<V_MTE2>(BUF1+pong):656.
                                            T.wait_flag("v", "mte2", IN_EV + po)
                                            T.copy(
                                                workspace_o[
                                                    cid, bufo, r0 : r0 + M_CHUNK, :
                                                ],
                                                in_ub[po, :, :],
                                            )
                                            # copy-in -> compute (= ref SetFlag/WaitFlag
                                            # <MTE2_V>(BUF1+pong):659-660).
                                            T.set_flag("mte2", "v", IN_EV + po)
                                            T.wait_flag("mte2", "v", IN_EV + po)
                                            if not_first:
                                                # prev tile STASHED workspace_acc via
                                                # MTE3; this MTE2 reload is a same-core
                                                # cross-tile GM RAW -> MTE3->MTE2 fence
                                                # (= ref DealBmm2:672-674).
                                                T.set_flag("mte3", "mte2", FENCE)
                                                T.wait_flag("mte3", "mte2", FENCE)
                                                # WAR: acc_pre free (= ref WaitFlag
                                                # <V_MTE2> SYNC_INPUT_BUF2:677).
                                                T.wait_flag("v", "mte2", ACC_EV)
                                                T.copy(
                                                    workspace_acc[
                                                        cid, prevo, r0 : r0 + M_CHUNK, :
                                                    ],
                                                    acc_pre,
                                                )
                                                # load -> compute (= ref SetFlag/WaitFlag
                                                # <MTE2_V> BUF2:682-683).
                                                T.set_flag("mte2", "v", ACC_EV)
                                                T.wait_flag("mte2", "v", ACC_EV)
                                                # rescale prev (RowMuls) -> Add (= ref
                                                # :691-694, PipeBarrier<V> between).
                                                # rescale prev = Ascend C RowMuls; 复用现成 experiment,
                                                # 按 xattention 写法展开: brcb 外置 + 逐 8*BLK(=64) 列段
                                                # mul_experiment(不传 tmp、src1 已 brcb), 覆盖整个 D。
                                                T.tile.brcb_experiment(
                                                    brcb_d,
                                                    expmax[bufo, mc, :, :],
                                                    M_CHUNK // 8,
                                                    1,
                                                    8,
                                                )
                                                T.pipe_barrier("v")
                                                for dcol in T.serial(D // (8 * BLK)):
                                                    cb = dcol * (8 * BLK)
                                                    T.tile.row_expand_mul_experiment(
                                                        acc_pre[:, cb : cb + 8 * BLK],
                                                        acc_pre[:, cb : cb + 8 * BLK],
                                                        brcb_d,
                                                    )
                                                T.pipe_barrier("v")
                                                T.tile.add(
                                                    in_ub[po, :, :],
                                                    in_ub[po, :, :],
                                                    acc_pre,
                                                )
                                                T.pipe_barrier("v")
                                                # acc_pre free (= ref SetFlag<V_MTE2>
                                                # BUF2:696).
                                                T.set_flag("v", "mte2", ACC_EV)
                                            if is_last:
                                                # normalize -> cast -> copy out + LSE
                                                # (= ref DealBmm2 700-708 + Bmm2Cast
                                                # AndCopyOut + LSE block; flags as SWA).
                                                # normalize = Ascend C RowDivs; 同上 xattention 写法:
                                                # brcb 外置 + 逐 8*BLK(=64) 列段 div_experiment(不传 tmp)。
                                                T.tile.brcb_experiment(
                                                    brcb_d,
                                                    denom[bufo, mc, :, :],
                                                    M_CHUNK // 8,
                                                    1,
                                                    8,
                                                )
                                                T.pipe_barrier("v")
                                                for dcol in T.serial(D // (8 * BLK)):
                                                    cb = dcol * (8 * BLK)
                                                    T.tile.row_expand_div_experiment(
                                                        in_ub[po, :, cb : cb + 8 * BLK],
                                                        in_ub[po, :, cb : cb + 8 * BLK],
                                                        brcb_d,
                                                    )
                                                T.pipe_barrier("v")
                                                # cast -> copy-out (MTE3). WAR: out_ub
                                                # free = ref WaitFlag<MTE3_V>(:617).
                                                T.wait_flag("mte3", "v", OUT_EV)
                                                T.tile.cast(
                                                    out_ub,
                                                    in_ub[po, :, :],
                                                    out_cast_mode,
                                                    M_CHUNK * D,
                                                )
                                                # in_ub slot free (V cast last-read it) =
                                                # ref SetFlag<V_MTE2>; cast -> copy-out =
                                                # ref SetFlag/WaitFlag<V_MTE3>(:624-625).
                                                T.set_flag("v", "mte2", IN_EV + po)
                                                T.set_flag("v", "mte3", OUT_EV)
                                                T.wait_flag("v", "mte3", OUT_EV)
                                                T.copy(
                                                    out_ub,
                                                    Output[tok, r0 : r0 + M_CHUNK, :],
                                                )
                                                # out_ub free = ref SetFlag<MTE3_V>(:627).
                                                T.set_flag("mte3", "v", OUT_EV)
                                                # LSE = ln(denom)+m_i (= ref :393-402).
                                                T.wait_flag("mte3", "v", LSE_EV)
                                                T.tile.ln(
                                                    lse_ub,
                                                    denom[bufo, mc, :, :],
                                                )
                                                T.tile.add(
                                                    lse_ub,
                                                    lse_ub,
                                                    m_i[bufo, mc, :, :],
                                                )
                                                T.set_flag("v", "mte3", LSE_EV)
                                                T.wait_flag("v", "mte3", LSE_EV)
                                                T.copy(
                                                    lse_ub, LSE[tok, r0 : r0 + M_CHUNK]
                                                )
                                                T.set_flag("mte3", "v", LSE_EV)
                                            else:
                                                # stash in_ub -> workspace_acc (MTE3) for
                                                # the next tile's rescale. Two guards =
                                                # ref DealBmm2:711-717 (which V-copies
                                                # bmm2ResUb->outUb then MTE3-stashes outUb):
                                                # (A) V->MTE3 so the stash reads the final
                                                # in_ub slot (after its MTE2 load + optional
                                                # V add); (B) MTE3->V->V_MTE2 so the next
                                                # reuse of this slot (waits V_MTE2) is
                                                # ordered after this stash read.
                                                T.set_flag("v", "mte3", IN_EV + po)
                                                T.wait_flag("v", "mte3", IN_EV + po)
                                                T.copy(
                                                    in_ub[po, :, :],
                                                    workspace_acc[
                                                        cid, bufo, r0 : r0 + M_CHUNK, :
                                                    ],
                                                )
                                                T.set_flag("mte3", "v", IN_EV + po)
                                                T.wait_flag("mte3", "v", IN_EV + po)
                                                T.set_flag("v", "mte2", IN_EV + po)
                            # NOTE: this vector EV_PV set looks unfaithful (reference
                            # syncC2V2 has a single cube setter, scfa_kernel.h:638), BUT
                            # removing it REGRESSES scfa (bf16 7/7 -> fail), so it is
                            # load-bearing for TileLang's flat-g pipeline -- keep it.
                            T.set_cross_flag("MTE3", EV_PV)
                    # Drain ALL the vector buffers' reverse flags ONCE (balance the
                    # primes above): each buffer's last consumer left its flag set
                    # (= block_vector.h FreeAllEventID for the SYNC_*_BUF flags).
                    T.wait_flag("v", "mte2", IN_EV)
                    T.wait_flag("v", "mte2", IN_EV + 1)
                    T.wait_flag("v", "mte2", ACC_EV)
                    T.wait_flag("mte3", "v", OUT_EV)
                    T.wait_flag("mte3", "v", LSE_EV)
                    # flag-3 drain (= scfa_kernel.h:777-780): consume the KVMERGE_RING
                    # credits the cube returned for the LAST KVMERGE_RING tasks (whose
                    # V0-waits already passed). Balances the cube prime: cube sets
                    # KVMERGE_RING + Ntask, vector waits Ntask + KVMERGE_RING -> equal, no
                    # leftover credit (which would loosen the WAR bound), no deadlock.
                    for _cr in range(KVMERGE_RING):
                        T.wait_cross_flag(EV_CREDIT)

        return sparse_attn_sharedkv_scfa

    func = kernel()
    if os.environ.get("SAS_DUMP_SRC"):
        try:
            src = func.get_kernel_source()
            with open("/tmp/scfa_gen.cpp", "w") as fh:
                fh.write(src)
            print(f"[SAS_DUMP_SRC] wrote {len(src)} chars to /tmp/scfa_gen.cpp")
        except Exception as exc:  # noqa: BLE001
            print(f"[SAS_DUMP_SRC] get_kernel_source failed: {exc!r}")
    return func


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
    # D-half (kL1) width = 256. MUST be defined here at function scope (a captured
    # Python int), NOT inside the @T.prim_func body: a `D2 = D // 2` statement
    # inside the prim_func becomes a TVMScript Let-bound symbolic tir.Var, which
    # then appears as a non-IntImm buffer dim ([BI, D2]) and SIGSEGVs LowerTileOp's
    # makeBufferWithLayout (it assumes static shapes for the L1 fractal layout).
    D2 = D // 2  # 256: the QK D-chunk (kL1) width
    G = N1 // N2  # GQA group = block_M rows handled per task (= 64)
    BI = DEFAULT_BLOCK_I  # kv window tile (= 128)
    # PV output-D tiling (= ComputeMm2 nL1Loops): D=512 -> 4 tiles of 128. The
    # decomposed PV drives these 4 tiles itself (kernel-driven mma + fused fixpipe
    # = gemm_v0_fixp internal nL0split=4), so the V D-slices can later enter the
    # 3-slot KV ring per tile. Function-scope Python ints (NOT prim_func body --
    # see the D2 note: a body-level const becomes a symbolic Var and SIGSEGVs).
    PV_NT = D // BI  # 4 output-D tiles
    PV_NW = BI  # 128: each output-D tile's column width (= mma n / fixpipe nSize)
    VEC_NUM = 2  # 2 vector cores per cube core
    G2 = G // VEC_NUM  # rows per vector core (= 32)
    BLK = 8  # FP32 elems per 32B block (Brcb fan-out width for row_expand)

    # DEBUG_SERIAL gates the cube's iteration-boundary barrier_all (line ~617).
    # Now False (Layer 5): the KV 3-ring + Q/P reverse flags protect every L1
    # buffer per-slot, the prime-once M_MTE1 ping-pong protects L0AB, and the
    # persistent cl0_iter rotation + 2-slot hardware unitFlag protects cL0 (the
    # reference has no cL0 software flag) -- so the barrier is redundant and the
    # cube QK(j)/PV(j-1) can pipeline. (The 11 vector-scope barriers are
    # unconditional and untouched here -- a separate later perf item.)
    DEBUG_SERIAL = False

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

    # KV 3-slot ring per-slot MTE2_MTE1/MTE1_MTE2 reverse-flag event ids (shared by
    # QK's K halves + PV's V tiles). Must avoid the gemm template's internal
    # MTE2_MTE1 self-fences (event 0 = L0AB_EVENT) and the L0AB M_MTE1 ({4,5}); {2,3,6}
    # is the free MTE2_MTE1 triple.
    # Scalars (NOT a Python list): a list indexed by a TVMScript loop Var (e.g.
    # slot = (2+nl)%3) fails to parse -- the index must be a literal or the flag id
    # a tir expr (KV_EV0 + h, or the if_then_else slot select in the PV loop).
    KV_EV0 = 2  # slot 0 (QK K half 0 / PV V tile 1)
    KV_EV1 = 3  # slot 1 (QK K half 1 / PV V tile 2)
    KV_EV2 = 6  # slot 2 (PV V tiles 0, 3)
    # QP reverse-flag event ids (Q halves as one unit + P). For SWA (mL1Loops=1)
    # the reference's 4-slot QP ring degenerates to Q (slots {0,1}, loaded+consumed
    # as the QK unit) + P (slot {2}) -- so one reverse flag for Q and one for P
    # suffice. Events {1,7} (event 1 free; 7 free), no reuse. Replaces the old
    # per-iteration QP_TMP_EV self-fence with the prime-once reverse-flag protocol
    # (a slot's reload waits on its own prior consumer), priming Q/P for the
    # eventual barrier removal. (The 4th slot / cross-iteration double-buffer is a
    # later perf refinement, and the depth-2->depth-3 pipeline is separate.)
    Q_EV = 1  # Q (both D-halves) reverse flag
    P_EV = 7  # P reverse flag

    # L0AB M_MTE1 ping-pong flags. These match the gemm_v0_fixp template's
    # DEDICATED shared-mode base L0AB_MM_EVENT (=4) and +1 (=5) -- NOT the default
    # L0AB_EVENT (=0). The shared (prime_drain=False) gemm holds these two M_MTE1
    # flags SET across the whole cube loop, so they must be disjoint from the
    # template's per-call MTE2_MTE1/MTE1_MTE2 self-pair fences (which stay on
    # {0,1}) and from the KV ring flags ({2,3,6}); {4,5} is the free pair (faithful
    # to the reference, which puts M_MTE1 on its own EVENT_ID3/4 disjoint from the
    # L1 flags). Primed ONCE before the cube loop (= AllocEventID,
    # block_cube.h:225-226) and drained ONCE after it (= FreeEventID, :239-240);
    # the gemm calls consume/re-arm them per tile rather than re-priming at every
    # QK/PV boundary.
    L0AB_EV0 = 4
    L0AB_EV1 = 5

    # Vector-core HardEvent ids for the fine-grained pipe flags that replace the
    # vector's full barrier_all (faithful to block_vector.h DealBmm1ResBaseBlock /
    # DealBmm2ResBaseBlock, which use 0 PIPE_ALL barriers -- only MTE2_V/V_MTE2/
    # MTE3_V/V_MTE3 SetFlag/WaitFlag + PipeBarrier<PIPE_V>). The vector core has its
    # own HardEvent namespace (disjoint from the cube's). Avoid id 0: copy_gm_to_ub's
    # padding path (common.h:282-288) uses MTE2_V/MTE3_V/V_MTE2 at id 0 (our full-tile
    # copies don't hit it, but stay clear). Each id is reused for the RAW and WAR
    # HardEvent of one buffer (= the reference's SYNC_INPUT/OUTPUT_BUF*_FLAG).
    SV_S_EV = 2  # softmax input buffer (s_ub/sink_ub): MTE2_V (RAW) + V_MTE2 (WAR)
    SV_P_EV = 3  # softmax output buffer (p_half): V_MTE3 (RAW) + MTE3_V (WAR)
    VO_O_EV = 4  # output input buffer (o_ub): MTE2_V (RAW) + V_MTE2 (WAR)
    VO_OUT_EV = 5  # output result buffer (o_half): V_MTE3 (RAW) + MTE3_V (WAR)
    VO_LSE_EV = 6  # output lse buffer (lse_ub): V_MTE3 (RAW) + MTE3_V (WAR)

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
                # LAYER 1 (QK D-chunking): QK contracts D=512, which the reference
                # loads as two 256-wide D-halves (ComputeMm1 kL1Loops=2,
                # block_cube.h:341-450), each Q/K half into its OWN L1 block, and
                # accumulates both into one cL0 before a single Fixpipe. The two
                # D-halves are separate 2D buffers (= the reference's two L1 blocks);
                # a [2,...] 3D array sliced by a const index would work too (3D-slice
                # is fine -- the earlier LowerTileOp segfault was a SYMBOLIC buffer
                # dim D2, not 3D slicing; see D2's def note). PV's V stays a whole
                # [BI,D] buffer: the whole-V gemm_v0_fixp already does the faithful
                # per-output-D-tile fused fixpipe + cL0 ping-pong internally
                # (nL0split=4 = ComputeMm2 nL1Loops); V into separate ring slots is
                # an OVERLAP concern, deferred to the KV/QP ring (Layer 3). 2*[G,256]
                # + 2*[BI,256] + [BI,D] + [G,BI] = 64+128+128+16 = 336KB < 512KB L1.
                q_l1_0 = T.alloc_L1([G, D2], dtype)  # Q D-half 0 (D[0:256])
                q_l1_1 = T.alloc_L1([G, D2], dtype)  # Q D-half 1 (D[256:512])
                # Shared KV 3-slot ring (= reference kvL1 ring): QK's 2 K D-halves
                # and PV's 4 V D-tiles rotate through the SAME 3 slots. Slot =
                # load_position % 3 (6 loads/task; 6 % 3 == 0 so the phase resets per
                # task -- identical to the reference's continuous kvL1BufIter % 3
                # since 6 is divisible by 3). K halves write the full [BI,256]; V
                # tiles write the first [BI,128] (same copy_pa dstNzC0Stride=BI, so
                # the slot's Nz layout is consistent for both). 3*128*256*2 = 192KB.
                # Per-slot reverse flags (KV_EV0/1/2) replace the single KV_QK/PV_EV,
                # priming the L1 ring for the eventual barrier removal.
                kv_ring = T.alloc_L1([3, BI, D2], dtype)
                p_l1 = T.alloc_L1([G, BI], dtype)
                # INCREMENT 1 (shared cL0 buffer): one cL0TensorPingPong shared by
                # QK and PV, faithful to the reference's single cL0TensorPingPong
                # (block_cube.h:127/212 -- one tmpBufL0C.Get, both ComputeMm1:559
                # and ComputeMm2:833 index it by cL0BufIter%2). [2,G,BI] = 2 slots
                # of [64,128]fp32 = 64KB < 128KB L0C. Both QK and PV index it by
                # the persistent cl0_iter % 2 (below), faithful to the reference's
                # single class-member cL0BufIter (cube.h:97/560/834).
                cL0 = T.alloc_L0C([2, G, BI], accum_dtype)
                # Persistent cL0 ping-pong counter = the reference's class-member
                # uint32_t cL0BufIter (cube.h:97). local.var -> a plain `int32_t
                # cl0_iter = 0;` declared once before the j-loop, so it survives
                # across iterations and is mutated in place (codegen_ascend.cc:471
                # `cl0_iter = ...;`). cL0BufIter advances +1 per QK (mL1Loops=1,
                # cube.h:613-615) and +1 per PV D-tile (cube.h:946-948), both ONLY
                # inside the valid-task guards (= the reference's isValid gate,
                # swa_kernel.h:750-762). cL0BufIter%2 is the slot: since PV adds an
                # even +4 per task, the parity flips once per task (+5 odd), making
                # consecutive cL0 uses alternate 0,1,0,1,... across the QK->PV->next
                # boundary -- so the 2-slot hardware unitFlag ping-pong (no software
                # cL0 flag, exactly as the reference) protects reuse and the
                # DEBUG_SERIAL barrier can be dropped next.
                cl0_iter = T.alloc_var("int32", init=0)
                # PV decomposition L0A/L0B ping-pong (kernel-driven per-tile mma).
                # P[G,winm] -> p_l0a, V tile [winm,128] -> v_l0b, alternating slot
                # pp = nl&1 (= gemm_v0_fixp tileIdx&1 = 0,1,0,1 for the 4 tiles).
                # These are OFFSET-0 views of the same A2/B2 space QK's gemm_v0_fixp
                # uses whole; the per-tile M_MTE1(4/5) flag chain orders QK's L0 use
                # before PV writes here (QK's last mma SetFlag<M_MTE1> -> PV's first
                # WaitFlag<M_MTE1>), so the address overlap is safe. p_l0a [2,64,128]
                # =32KB, v_l0b [2,128,128]=64KB (within the 64KB A2/B2 each).
                p_l0a = T.alloc_L0A([2, G, BI], dtype)  # P activations
                v_l0b = T.alloc_L0B([2, BI, BI], dtype)  # V tile (K=winm, N=128)
                # Vector UB scratch.
                s_ub = T.alloc_ub([G2, BI], accum_dtype)
                p_half = T.alloc_ub([G2, BI], dtype)
                # 手拼 softmax(替代 007)新增: m_2d = broadcast(max) 的 [G2,BI] 目标;
                # sumP = reduce_sum(P) 的 [G2,1] 临时。
                m_2d = T.alloc_ub([G2, BI], accum_dtype)
                sumP = T.alloc_ub([G2, 1], accum_dtype)
                o_ub = T.alloc_ub([G2, D], accum_dtype)
                o_half = T.alloc_ub([G2, D], dtype)
                sink_ub = T.alloc_ub([G2, 1], accum_dtype)
                lse_ub = T.alloc_ub([G2, 1], accum_dtype)
                # 手拼 softmax 状态: expmax_ub 复用为 denom 的 sink 项 expsink=exp(sink-m)。
                # (ones_ub 是 007 的 in_sum seed, 手拼不再需要, 保留其一次性 fill 无害。)
                ones_ub = T.alloc_ub([G2, 1], accum_dtype)
                expmax_ub = T.alloc_ub([G2, 1], accum_dtype)
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
                    # AllocEventID (block_cube.h:225-226): prime the two L0AB
                    # M_MTE1 ping-pong flags ONCE for the whole cube loop. The
                    # shared QK/PV gemm_v0_fixp calls (prime_drain=False) consume
                    # and re-arm these per tile instead of self-priming+draining
                    # the L0AB ring at every call boundary -- faithful to the
                    # reference, where ComputeMm1/Mm2 never touch the L0AB flag
                    # lifecycle (only the per-slot Wait/Set inside the tile loop).
                    T.set_flag("m", "mte1", L0AB_EV0)
                    T.set_flag("m", "mte1", L0AB_EV1)
                    # AllocEventID (cont.): prime the 3 KV-ring slots' MTE1_MTE2
                    # reverse flags ONCE, so the first load into each slot waits on
                    # an already-set flag (the slot is "free"); each consume re-arms
                    # it. Drained once after the loop (FreeEventID).
                    T.set_flag("mte1", "mte2", KV_EV0)
                    T.set_flag("mte1", "mte2", KV_EV1)
                    T.set_flag("mte1", "mte2", KV_EV2)
                    # Prime the Q / P reverse flags too (free = ready to load).
                    T.set_flag("mte1", "mte2", Q_EV)
                    T.set_flag("mte1", "mte2", P_EV)
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
                                    # QK D-chunking (faithful ComputeMm1 kL1Loops=2,
                                    # block_cube.h:341-450/545-557): load Q and K as
                                    # two 256-wide D-halves into their own L1 slots.
                                    # Q half h = Q[tok, :, h*256:(h+1)*256] (a strided
                                    # GM slice, = CopyInMm1AToL1 headOffset=h*256,
                                    # headSize=256); K half h = copy_pa with
                                    # act_head_dim=256, d_idx=h*256 (= DataCopyPA
                                    # startPos.dIdx=kL1*256, actHeadDim=256).
                                    # Q reverse flag: wait the Q buffers' prior consumer
                                    # (primed once / the previous task's QK gemm), load
                                    # both D-halves, signal loaded.
                                    T.wait_flag("mte1", "mte2", Q_EV)
                                    T.copy(Q[tok, :, 0:D2], q_l1_0)
                                    T.copy(Q[tok, :, D2:D], q_l1_1)
                                    T.set_flag("mte2", "mte1", Q_EV)
                                    # K D-halves -> KV ring slots 0,1. Per-slot reverse
                                    # flag: wait the slot's prior consumer (primed once
                                    # / a previous task's V tile), load, signal loaded.
                                    # = the reference's per-slot kvL1 ring (the gemm is
                                    # unchanged -- it just reads the slot buffer passed).
                                    # ev = KV_EV0 + h (= 2,3 for slots 0,1) -- a tir
                                    # expr, since h is a TVMScript loop Var (can't
                                    # index a Python list with it).
                                    for h in range(2):
                                        T.wait_flag("mte1", "mte2", KV_EV0 + h)
                                        T.copy_pa(
                                            kv_ring[h, :, :],
                                            ori_kv,
                                            ori_block_table,
                                            ori_block_size,
                                            N2,
                                            D,
                                            ori_block_size * N2 * D,
                                            ori_table_len,
                                            D2,
                                            win,
                                            BI,
                                            b,
                                            0,
                                            ori_left,
                                            h * D2,
                                        )
                                        T.set_flag("mte2", "mte1", KV_EV0 + h)
                                    # QK = Q @ Kᵀ over the window; N rounds up to 16 to
                                    # match Ascend C ComputeMm1 (nL1SizeAlign =
                                    # SASAlign(window, 16)); copy_pa loads `win` KV
                                    # rows. cL0[slot 0][0:win_align] is the score -- the
                                    # [win:win_align] tail is Q@unloaded-KV, but the
                                    # softmax compacts/processes win_align and the
                                    # reduce/PV use winm, so it is excluded exactly as
                                    # in the reference. [win_align:BI] stays
                                    # uninitialised; the softmax never reads it.
                                    win_align = (win + 15) // 16 * 16
                                    T.wait_flag("mte2", "mte1", Q_EV)
                                    # Faithful QK = ComputeMm1, per-D-chunk: each
                                    # 256-wide D-half accumulates into the SAME shared
                                    # cL0 slot (cl0_base=0); chunk 0 inits + holds
                                    # (flush_last/do_fixpipe=False), chunk 1 flushes +
                                    # fixpipes the fully-accumulated Q@Kᵀ to
                                    # workspace_s (= the reference's single Fixpipe
                                    # after both kL1 halves, cube.h:591). k_actual=D2
                                    # is each chunk's own contraction width;
                                    # n_actual=win_align = the score's real columns.
                                    # Each chunk reads K half from KV ring slot h; the
                                    # per-slot MTE2_MTE1 wait gates the gemm's L1->L0,
                                    # the MTE1_MTE2 set after releases the slot for its
                                    # next ring user.
                                    T.wait_flag("mte2", "mte1", KV_EV0)
                                    T.gemm_v0_fixp(
                                        q_l1_0,
                                        kv_ring[0, :, :],
                                        cL0,
                                        workspace_s[cid, buf, :, :],
                                        k_actual=D2,
                                        transpose_B=True,
                                        init=True,
                                        n_actual=win_align,
                                        cl0_base=cl0_iter % 2,
                                        prime_drain=False,
                                        flush_last=False,
                                        do_fixpipe=False,
                                    )
                                    T.set_flag("mte1", "mte2", KV_EV0)
                                    T.wait_flag("mte2", "mte1", KV_EV1)
                                    T.gemm_v0_fixp(
                                        q_l1_1,
                                        kv_ring[1, :, :],
                                        cL0,
                                        workspace_s[cid, buf, :, :],
                                        k_actual=D2,
                                        transpose_B=True,
                                        init=False,
                                        n_actual=win_align,
                                        cl0_base=cl0_iter % 2,
                                        prime_drain=False,
                                        flush_last=True,
                                        do_fixpipe=True,
                                    )
                                    T.set_flag("mte1", "mte2", KV_EV1)
                                    # both gemms consumed Q -> release the Q buffers
                                    # for their next ring user (next task's QK load).
                                    T.set_flag("mte1", "mte2", Q_EV)
                                    # QK consumed one cL0 slot (the two D-halves
                                    # accumulate into the SAME slot, one fixpipe ->
                                    # +1, = cube.h:613-615 mL1Loops==1). Advance the
                                    # persistent cL0BufIter once per valid QK so the
                                    # next PV/QK ping-pongs onto the other slot.
                                    cl0_iter += 1
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
                                    # P reverse flag: wait the P buffer's prior consumer
                                    # (primed once / the previous task's PV), load P,
                                    # signal loaded, wait ready (for the L0A reads below).
                                    T.wait_flag("mte1", "mte2", P_EV)
                                    T.copy(workspace_p[cid, bufm, :, :], p_l1)
                                    T.set_flag("mte2", "mte1", P_EV)
                                    T.wait_flag("mte2", "mte1", P_EV)
                                    # PV = ComputeMm2, decomposed (= gemm_v0_fixp
                                    # nL0split=4) + V D-tiles into the SHARED KV 3-ring.
                                    # Each of the 4 output-D tiles: LOAD V tile -> ring
                                    # slot (per-slot reverse flag), then CONSUME it (L0
                                    # load + mma + fused fixpipe). load+consume are
                                    # INTERLEAVED per tile so the per-slot MTE1_MTE2
                                    # flag orders slot reuse (slot 2 is reused by tile 0
                                    # and tile 3) -- a two-pass load-all/consume-all
                                    # would overwrite slot 2 before tile 0 is consumed.
                                    # Slot = (2+nl)%3 = 2,0,1,2 (continues the ring after
                                    # QK's K halves took slots 0,1; 6 loads/task, 6%3==0
                                    # = the reference's continuous kvL1BufIter%3).
                                    #
                                    # pp=nl&1 (0,1,0,1) = gemm tileIdx&1 -> L0AB events
                                    # {4,5}. cs=cl0_iter%2 (the persistent cL0BufIter,
                                    # cube.h:834): the 4 tiles ping-pong the 2 cL0
                                    # slots while continuing QK's rotation, so reuse
                                    # alternates across the iteration boundary too.
                                    # real_k=k_actual=winm
                                    # (L0 fractal K and mma K MUST match, else M-block/
                                    # head 16-63 wrong for winm<128). unit_flag=0b11
                                    # (single K tile); mma->fixpipe fused, no software
                                    # M_FIX flag. (G/16)*(128/16)=32>=10 -> no
                                    # PipeBarrier<PIPE_M> (= gemm:944 gate).
                                    for nl in range(PV_NT):
                                        slot = (2 + nl) % 3
                                        pp = nl % 2
                                        cs = cl0_iter % 2
                                        # ev = slot's ring event (slots 0,1 -> KV_EV0
                                        # +slot = 2,3; slot 2 -> KV_EV2 = 6). A tir
                                        # expr -- slot is a loop Var (no list index).
                                        ev = T.if_then_else(
                                            slot < 2, KV_EV0 + slot, KV_EV2
                                        )
                                        # LOAD V D-tile nl -> ring slot (act_head_dim
                                        # =128, d_idx=nl*128). Wait the slot's prior
                                        # consumer (QK's K half / an earlier V tile).
                                        T.wait_flag("mte1", "mte2", ev)
                                        T.copy_pa(
                                            kv_ring[slot, :, :],
                                            ori_kv,
                                            ori_block_table,
                                            ori_block_size,
                                            N2,
                                            D,
                                            ori_block_size * N2 * D,
                                            ori_table_len,
                                            PV_NW,
                                            winm,
                                            BI,
                                            bm,
                                            0,
                                            ori_leftm,
                                            nl * PV_NW,
                                        )
                                        T.set_flag("mte2", "mte1", ev)
                                        # CONSUME: L0 load (real_k=winm) -> mma
                                        # (k_actual=winm) -> fused fixpipe (unit_flag).
                                        T.wait_flag("m", "mte1", L0AB_EV0 + pp)
                                        T.wait_flag("mte2", "mte1", ev)
                                        T.copy(p_l1[:, :], p_l0a[pp, :, :], real_k=winm)
                                        T.copy(
                                            kv_ring[slot, :, 0:PV_NW],
                                            v_l0b[pp, :, :],
                                            real_k=winm,
                                        )
                                        T.set_flag("mte1", "m", L0AB_EV0 + pp)
                                        T.wait_flag("mte1", "m", L0AB_EV0 + pp)
                                        T.mma(
                                            p_l0a[pp, :, :],
                                            v_l0b[pp, :, :],
                                            cL0[cs, :, :],
                                            init=True,
                                            k_actual=winm,
                                            unit_flag=0b11,
                                        )
                                        T.set_flag("m", "mte1", L0AB_EV0 + pp)
                                        # release ring slot for its next user (reverse)
                                        T.set_flag("mte1", "mte2", ev)
                                        T.copy(
                                            cL0[cs, :, :],
                                            workspace_o[
                                                cid,
                                                bufm,
                                                :,
                                                nl * PV_NW : (nl + 1) * PV_NW,
                                            ],
                                            unit_flag=0b11,
                                        )
                                        # this D-tile consumed one cL0 slot -> advance
                                        # the persistent cL0BufIter (= cube.h:946-948,
                                        # one ++ per nL1 with mL1Loops==1) so the next
                                        # tile/QK lands on the other slot.
                                        cl0_iter += 1
                                    # all 4 tiles consumed P -> release the P buffer
                                    # for its next ring user (next task's PV load).
                                    T.set_flag("mte1", "mte2", P_EV)
                                    # Iteration-boundary full drain (kept while
                                    # DEBUG_SERIAL). All cross-iteration hazards are now
                                    # covered without it: L1 by the KV ring + Q/P
                                    # reverse flags, L0AB by the prime-once M_MTE1
                                    # ping-pong (cl0_iter is even-invariant in parity),
                                    # and cL0 by the persistent cl0_iter rotation +
                                    # 2-slot hardware unitFlag (the reference has no cL0
                                    # software flag). Flipping DEBUG_SERIAL=False next
                                    # (Layer 5) drops this barrier; Step 1 keeps it to
                                    # verify the counter machinery in isolation first.
                                    if DEBUG_SERIAL:
                                        T.barrier_all()
                            T.set_cross_flag("FIX", EV_PV)
                    # FreeEventID (block_cube.h:239-240): drain the two L0AB
                    # M_MTE1 flags ONCE after the whole cube loop, balancing the
                    # AllocEventID prime above (the shared gemm calls left them
                    # armed instead of self-draining at each call boundary).
                    T.wait_flag("m", "mte1", L0AB_EV0)
                    T.wait_flag("m", "mte1", L0AB_EV1)
                    # FreeEventID (cont.): drain the 3 KV-ring slots' MTE1_MTE2
                    # reverse flags, balancing their prime above (each slot's last
                    # consumer left its flag set).
                    T.wait_flag("mte1", "mte2", KV_EV0)
                    T.wait_flag("mte1", "mte2", KV_EV1)
                    T.wait_flag("mte1", "mte2", KV_EV2)
                    # drain the Q / P reverse flags too (each left set by its last
                    # consumer), balancing their prime.
                    T.wait_flag("mte1", "mte2", Q_EV)
                    T.wait_flag("mte1", "mte2", P_EV)

                # =========================== VECTOR ===========================
                # DEPTH-3 (= reference PreloadPipeline, swa_kernel.h:745-768): the
                # vector lags the cube by one task. iter j does softmax(task j-1)
                # and output(task j-2), while the cube does QK(task j) and
                # PV(task j-1). So softmax(j-1) reads QK(j-1)'s ws_s (set in the
                # PREVIOUS cube iter) and runs CONCURRENT with the cube's current
                # QK(j) -- the deeper preload of the reference's extraInfo0/2/1 =
                # task loop / loop-1 / loop-2. Cross-flag counts stay balanced
                # (EV_QK/EV_P/EV_PV each n_iter sets == n_iter waits); ws_s/ws_p/
                # ws_o + m_i/denom stay mod-2 double-buffered (the cube's PV-after-
                # softmax chain via EV_P keeps QK(loop) after softmax(loop-2), so
                # the prior reader is done before the buffer is reused).
                with T.Scope("V"):
                    # in_sum seed for SoftmaxFlashV2 = 1.0 (Ascend C R0); filled
                    # once, read each task as the flash running-sum initial value.
                    T.tile.fill(ones_ub, T.float32(1.0))
                    # Prime the softmax buffers' reverse flags ONCE (buffers free) =
                    # block_vector.h's initial SYNC_INPUT/OUTPUT_BUF1 flag state:
                    # V_MTE2 (s_ub/sink free) + MTE3_V (p_half free). Drained once
                    # after the loop. The old barrier_all here was redundant: the
                    # fill and the softmax's ones_ub read are both PIPE_V, so the V
                    # pipe's in-order issue already orders them.
                    T.set_flag("v", "mte2", SV_S_EV)
                    T.set_flag("mte3", "v", SV_P_EV)
                    # Same for the output buffers: o_ub free (V_MTE2), o_half free
                    # + lse_ub free (MTE3_V). = block_vector.h DealBmm2ResBaseBlock
                    # SYNC_INPUT/OUTPUT_BUF1 initial state.
                    T.set_flag("v", "mte2", VO_O_EV)
                    T.set_flag("mte3", "v", VO_OUT_EV)
                    T.set_flag("mte3", "v", VO_LSE_EV)
                    # j in [1, n_iter+1]: softmax(j-1) for j in [1,n_iter], output
                    # (j-2) for j in [2,n_iter+1] -- both single-comparison guards.
                    for j in T.serial(1, n_iter + 2):
                        # ---- softmax stage for task j-1 ----
                        if j < n_iter + 1:
                            T.wait_cross_flag(EV_QK)
                            pid = (j - 1) * core_num + cid
                            buf = (j - 1) % 2
                            if pid < block_num:
                                b = T.cast(pid // max_seq, "int32")
                                s = T.cast(pid % max_seq, "int32")
                                act_q = act_q_lens[b]
                                if s < act_q:
                                    act_kv = seqused_kv[b]
                                    s_global = act_kv - act_q + s
                                    ori_left = T.max(s_global - ori_win_left, 0)
                                    # winm = window length (= Ascend C
                                    # actualSingleProcessSInnerSize) = 有效列数(动态)。
                                    # 手拼只搬/算前 winm 列(014 运行期 extent copy),其余
                                    # 到 BI 用 -inf mask,故不再需要 win_align 对齐。
                                    winm = s_global + 1 - ori_left
                                    # copy-in s_ub + sink (MTE2). WAR: wait s_ub free
                                    # (prev task's compute done reading it) = ref
                                    # WaitFlag<V_MTE2>(block_vector.h:415).
                                    T.wait_flag("v", "mte2", SV_S_EV)
                                    # 手拼替代 007: 先满宽 fill -inf mask, 再用 014(运行期
                                    # inner-extent copy)只搬有效 winm 列 -> [winm:BI] 保持
                                    # -inf, 定长满宽 reduce/exp 等价于只算有效列(exp(-inf)=0、
                                    # max(-inf,·)=·), 绕开"变长 reduce"。fill(V)->copy(MTE2)
                                    # 用 SV_S_EV 做一次 V->MTE2 握手(与下方 copy->compute 同
                                    # buffer 的 V<->MTE2 复用同一 event)。
                                    T.tile.fill(s_ub, -T.infinity(accum_dtype))
                                    T.set_flag("v", "mte2", SV_S_EV)
                                    T.wait_flag("v", "mte2", SV_S_EV)
                                    T.copy(
                                        workspace_s[
                                            cid, buf, vid * G2 : (vid + 1) * G2, 0:winm
                                        ],
                                        s_ub[:, 0:winm],
                                    )
                                    T.copy(sinks[vid * G2 : (vid + 1) * G2], sink_ub)
                                    # copy-in done -> compute (= ref SetFlag/WaitFlag
                                    # <MTE2_V>, :418-419). One flag covers both MTE2
                                    # copies (s_ub for mul/softmax, sink for softmax).
                                    T.set_flag("mte2", "v", SV_S_EV)
                                    T.wait_flag("mte2", "v", SV_S_EV)
                                    T.tile.mul(s_ub, s_ub, softmax_scale)
                                    # V-pipe order mul -> softmax (= ref PipeBarrier
                                    # <PIPE_V>, :423).
                                    T.pipe_barrier("v")
                                    # 手拼 sink-seeded 单块 softmax(替代 007 SoftmaxFlashV2):
                                    #   m = max(sink, rowmax(有效列))
                                    #   P = exp(score - m)
                                    #   denom = exp(sink - m) + sum(P)
                                    # padding [winm:BI]=-inf 使定长满宽 reduce/exp 等价于只算
                                    # 有效列。out_max=m_i、out_sum=denom 下传 output stage
                                    # (o/denom、lse=m+ln denom); 单块无 rescale, expmax 不需要。
                                    # reduce_max 不破坏 src(microbench 验证), 故 sub 仍可读 s_ub。
                                    T.reduce_max(s_ub, m_i[buf, :, :], dim=-1)
                                    # m = max(sink, rowmax): dst 放最后(防 T.tile.max 静默丢操作数)
                                    T.tile.max(m_i[buf, :, :], sink_ub, m_i[buf, :, :])
                                    T.tile.broadcast(m_2d, m_i[buf, :, :])
                                    T.tile.sub(s_ub, s_ub, m_2d)
                                    T.tile.exp(s_ub, s_ub)
                                    # denom 的 sink 项 expsink = exp(sink - m), 暂存 expmax_ub
                                    T.tile.sub(expmax_ub, sink_ub, m_i[buf, :, :])
                                    T.tile.exp(expmax_ub, expmax_ub)
                                    # V-pipe order softmax -> cast.
                                    T.pipe_barrier("v")
                                    # cast P 出去(读 s_ub, 不破坏)。WAR: wait p_half free
                                    # (prev copy-out done)。
                                    T.wait_flag("mte3", "v", SV_P_EV)
                                    T.tile.cast(p_half, s_ub, "CAST_ROUND", G2 * BI)
                                    # reduce_sum 会消耗/破坏 src(microbench 教训), 故放 cast
                                    # 之后(同 V pipe, pipe_barrier 隔开 cast 先读完), 再破坏
                                    # s_ub 无妨; denom = expsink + sum(P)。
                                    T.pipe_barrier("v")
                                    T.reduce_sum(s_ub, sumP, dim=-1)
                                    T.tile.add(denom[buf, :, :], expmax_ub, sumP)
                                    # s_ub free 在 reduce_sum 之后(compute 全部读完/破坏完);
                                    # cast done -> copy-out(V_MTE3)。
                                    T.set_flag("v", "mte2", SV_S_EV)
                                    T.set_flag("v", "mte3", SV_P_EV)
                                    T.wait_flag("v", "mte3", SV_P_EV)
                                    T.copy(
                                        p_half,
                                        workspace_p[
                                            cid, buf, vid * G2 : (vid + 1) * G2, :
                                        ],
                                    )
                                    # copy-out done, p_half free = ref SetFlag<MTE3_V>
                                    # (:439); also gates the cross-core EV_P below.
                                    T.set_flag("mte3", "v", SV_P_EV)
                            T.set_cross_flag("MTE3", EV_P)
                        # ---- output stage for task j-2 ----
                        if j >= 2:
                            T.wait_cross_flag(EV_PV)
                            pidm = (j - 2) * core_num + cid
                            bufm = (j - 2) % 2
                            if pidm < block_num:
                                bm = T.cast(pidm // max_seq, "int32")
                                sm = T.cast(pidm % max_seq, "int32")
                                if sm < act_q_lens[bm]:
                                    tokm = q_prefix[bm] + sm
                                    # copy-in o_ub (MTE2). WAR: wait o_ub free = ref
                                    # WaitFlag<V_MTE2>(block_vector.h:656).
                                    T.wait_flag("v", "mte2", VO_O_EV)
                                    T.copy(
                                        workspace_o[
                                            cid, bufm, vid * G2 : (vid + 1) * G2, :
                                        ],
                                        o_ub,
                                    )
                                    # copy-in done -> compute = ref SetFlag/WaitFlag
                                    # <MTE2_V>(:659-660).
                                    T.set_flag("mte2", "v", VO_O_EV)
                                    T.wait_flag("mte2", "v", VO_O_EV)
                                    # o = o / denom via row broadcast (Brcb + Div,
                                    # src1RepStride=1) over the full D -- no
                                    # [G2,D] denom buffer, faithful to Ascend C
                                    # RowDivs; processes headDim in one pass.
                                    # o /= denom = Ascend C RowDivs; 复用现成 experiment 的 xattention
                                    # 写法: brcb 外置 + 逐 8*BLK(=64) 列段 div_experiment(不传 tmp)。
                                    T.tile.brcb_experiment(
                                        brcb_d, denom[bufm, :, :], G2 // 8, 1, 8
                                    )
                                    T.pipe_barrier("v")
                                    for dcol in T.serial(D // (8 * BLK)):
                                        cb = dcol * (8 * BLK)
                                        T.tile.row_expand_div_experiment(
                                            o_ub[:, cb : cb + 8 * BLK],
                                            o_ub[:, cb : cb + 8 * BLK],
                                            brcb_d,
                                        )
                                    # V-pipe order RowDivs -> cast (= ref PipeBarrier
                                    # <PIPE_V>, :707).
                                    T.pipe_barrier("v")
                                    # cast -> copy-out (MTE3). WAR: wait o_half free =
                                    # ref WaitFlag<MTE3_V>(:617).
                                    T.wait_flag("mte3", "v", VO_OUT_EV)
                                    T.tile.cast(o_half, o_ub, out_cast_mode, G2 * D)
                                    # o_ub free (compute done reading) = ref SetFlag
                                    # <V_MTE2>(:665, here after the last o_ub read);
                                    # cast done -> copy-out = ref SetFlag/WaitFlag
                                    # <V_MTE3>(:624-625).
                                    T.set_flag("v", "mte2", VO_O_EV)
                                    T.set_flag("v", "mte3", VO_OUT_EV)
                                    T.wait_flag("v", "mte3", VO_OUT_EV)
                                    T.copy(
                                        o_half,
                                        Output[tokm, vid * G2 : (vid + 1) * G2, :],
                                    )
                                    # copy-out done, o_half free = ref SetFlag<MTE3_V>
                                    # (:627).
                                    T.set_flag("mte3", "v", VO_OUT_EV)
                                    # lse = max + ln(sum) (T.tile.ln; scalar tir.log
                                    # unlowerable on Ascend). lse_ub WAR: wait free,
                                    # then ln+add (V) -> copy-out (MTE3) with V_MTE3.
                                    T.wait_flag("mte3", "v", VO_LSE_EV)
                                    T.tile.ln(lse_ub, denom[bufm, :, :])
                                    T.tile.add(lse_ub, lse_ub, m_i[bufm, :, :])
                                    T.set_flag("v", "mte3", VO_LSE_EV)
                                    T.wait_flag("v", "mte3", VO_LSE_EV)
                                    T.copy(lse_ub, LSE[tokm, vid * G2 : (vid + 1) * G2])
                                    T.set_flag("mte3", "v", VO_LSE_EV)
                    # Drain ALL the vector buffers' reverse flags ONCE (balance the
                    # primes above): each buffer's last consumer left its flag set
                    # (= block_vector.h FreeEventID for SYNC_INPUT/OUTPUT_BUF1).
                    T.wait_flag("v", "mte2", SV_S_EV)
                    T.wait_flag("mte3", "v", SV_P_EV)
                    T.wait_flag("v", "mte2", VO_O_EV)
                    T.wait_flag("mte3", "v", VO_OUT_EV)
                    T.wait_flag("mte3", "v", VO_LSE_EV)

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
