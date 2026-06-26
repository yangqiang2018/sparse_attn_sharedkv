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
        raise NotImplementedError(f"only SWA (scenario=1) is implemented; got scenario={scenario} (SCFA/CFA pending)")
    if n_kv_heads != 1 or n_heads != 64 or head_dim != 512:
        raise ValueError(f"SWA kernel assumes N1=64, N2=1, D=512 (got N1={n_heads}, N2={n_kv_heads}, D={head_dim})")
    if ori_win_left + 1 > DEFAULT_BLOCK_I:
        raise ValueError(
            f"ori_win_left={ori_win_left} exceeds single-tile window (BI={DEFAULT_BLOCK_I}); multi-tile SWA not implemented yet"
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
    # uint8 shared scratch for SoftmaxFlashV2 (mirrors Ascend C softmaxTmpUb /
    # tmpBuff1 = 32KB; ample for the [G2, BI] softmax block).
    SOFTMAX_TMP_BYTES = 32768

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

    # Intra-cube MTE2->MTE1 pipe-flag event ids for the QK/PV KV buffers
    # (kq_l1=QK's K D-halves, kv_l1=PV's V). Separate MTE2_MTE1 id namespace; must
    # avoid the gemm template's internal {0,1} (L0AB_EVENT) and the L0AB M_MTE1
    # {4,5}. So {2,3}. These let PV's copy_pa (MTE2 into kv_l1) overlap QK's gemm
    # (M reading kq_l1) instead of a full barrier_all -- faithful to the
    # reference's per-slot KV flags.
    KV_QK_EV = 2  # kq_l1 (QK K halves) MTE2 done
    KV_PV_EV = 3  # kv_l1 (PV V) MTE2 done

    # L0AB M_MTE1 ping-pong flags. These match the gemm_v0_fixp template's
    # DEDICATED shared-mode base L0AB_MM_EVENT (=4) and +1 (=5) -- NOT the default
    # L0AB_EVENT (=0). The shared (prime_drain=False) gemm holds these two M_MTE1
    # flags SET across the whole cube loop, so they must be disjoint from the
    # template's per-call MTE2_MTE1/MTE1_MTE2 self-pair fences (which stay on
    # {0,1}) and from the KV pipe-flags ({2,3}); {4,5} is the free pair (faithful
    # to the reference, which puts M_MTE1 on its own EVENT_ID3/4 disjoint from the
    # L1 flags). Primed ONCE before the cube loop (= AllocEventID,
    # block_cube.h:225-226) and drained ONCE after it (= FreeEventID, :239-240);
    # the gemm calls consume/re-arm them per tile rather than re-priming at every
    # QK/PV boundary.
    L0AB_EV0 = 4
    L0AB_EV1 = 5

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
                kq_l1_0 = T.alloc_L1([BI, D2], dtype)  # K D-half 0
                kq_l1_1 = T.alloc_L1([BI, D2], dtype)  # K D-half 1
                kv_l1 = T.alloc_L1([BI, D], dtype)  # PV's V (whole)
                p_l1 = T.alloc_L1([G, BI], dtype)
                # INCREMENT 1 (shared cL0 buffer): one cL0TensorPingPong shared by
                # QK and PV, faithful to the reference's single cL0TensorPingPong
                # (block_cube.h:127/212 -- one tmpBufL0C.Get, both ComputeMm1:559
                # and ComputeMm2:833 index it by cL0BufIter%2). [2,G,BI] = 2 slots
                # of [64,128]fp32 = 64KB < 128KB L0C. QK uses slot 0 (cl0_base=0);
                # PV rotates slots 1,0,1,0 (cl0_base=1, its 4 D-tiles ping-pong).
                # The DEBUG_SERIAL barrier still drains each iteration, so the
                # QK->PV->next-iter cL0 reuse is masked here -- this increment only
                # merges the buffer (de-risking it in isolation). Continuous
                # cross-call cL0BufIter + prime-once L0AB flags (removing the
                # per-call M_MTE1 drain that still serialises QK/PV) come next, as
                # compiler 010; deleting the barrier needs the L1 rings after that.
                cL0 = T.alloc_L0C([2, G, BI], accum_dtype)
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
                    # AllocEventID (block_cube.h:225-226): prime the two L0AB
                    # M_MTE1 ping-pong flags ONCE for the whole cube loop. The
                    # shared QK/PV gemm_v0_fixp calls (prime_drain=False) consume
                    # and re-arm these per tile instead of self-priming+draining
                    # the L0AB ring at every call boundary -- faithful to the
                    # reference, where ComputeMm1/Mm2 never touch the L0AB flag
                    # lifecycle (only the per-slot Wait/Set inside the tile loop).
                    T.set_flag("m", "mte1", L0AB_EV0)
                    T.set_flag("m", "mte1", L0AB_EV1)
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
                                    T.copy(Q[tok, :, 0:D2], q_l1_0)
                                    T.copy(Q[tok, :, D2:D], q_l1_1)
                                    T.copy_pa(
                                        kq_l1_0,
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
                                        0,
                                    )
                                    T.copy_pa(
                                        kq_l1_1,
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
                                        D2,
                                    )
                                    # q/kq D-halves loaded -> tag MTE2 done so
                                    # the gemm's MTE1 L1->L0 load waits on this point-to
                                    # -point flag instead of a full barrier_all,
                                    # letting PV's later copy_pa(kv_l1) MTE2 overlap
                                    # this QK gemm's M pipe.
                                    T.set_flag("mte2", "mte1", KV_QK_EV)
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
                                    T.wait_flag("mte2", "mte1", KV_QK_EV)
                                    # Faithful QK = ComputeMm1, per-D-chunk: each
                                    # 256-wide D-half accumulates into the SAME shared
                                    # cL0 slot (cl0_base=0); chunk 0 inits + holds
                                    # (flush_last/do_fixpipe=False), chunk 1 flushes +
                                    # fixpipes the fully-accumulated Q@Kᵀ to
                                    # workspace_s (= the reference's single Fixpipe
                                    # after both kL1 halves, cube.h:591). k_actual=D2
                                    # is each chunk's own contraction width;
                                    # n_actual=win_align = the score's real columns.
                                    T.gemm_v0_fixp(
                                        q_l1_0,
                                        kq_l1_0,
                                        cL0,
                                        workspace_s[cid, buf, :, :],
                                        k_actual=D2,
                                        transpose_B=True,
                                        init=True,
                                        n_actual=win_align,
                                        cl0_base=0,
                                        prime_drain=False,
                                        flush_last=False,
                                        do_fixpipe=False,
                                    )
                                    T.gemm_v0_fixp(
                                        q_l1_1,
                                        kq_l1_1,
                                        cL0,
                                        workspace_s[cid, buf, :, :],
                                        k_actual=D2,
                                        transpose_B=True,
                                        init=False,
                                        n_actual=win_align,
                                        cl0_base=0,
                                        prime_drain=False,
                                        flush_last=True,
                                        do_fixpipe=True,
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
                                    # the reference reloading V in Mm2). PV's V is
                                    # still loaded whole [BI,D] here; its faithful
                                    # per-output-D 4-slice load is Layer 2.
                                    T.copy_pa(
                                        kv_l1[:, :],
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
                                    # kv_l1 loaded (and p_l1, earlier on the
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
                                    # PV = ComputeMm2, DECOMPOSED to kernel-driven
                                    # per-output-D-tile mma + fused fixpipe (= the
                                    # gemm_v0_fixp internal nL0split=4 loop lifted
                                    # into the kernel, so each V D-slice can enter the
                                    # 3-slot KV ring next). Replaces the single
                                    # gemm_v0_fixp(p_l1, kv_l1, cL0, ws_o, k_actual=
                                    # winm, init, cl0_base=1, prime_drain=False).
                                    #
                                    # Each of the 4 tiles (Python-unrolled so nl/pp/cs
                                    # are literals -- avoids symbolic flag ids):
                                    #   wait L0AB slot -> load P->L0A, V D-tile->L0B
                                    #   -> L0-loaded fence -> mma into cL0 ping-pong
                                    #   slot -> free L0AB -> fixpipe to ws_o column
                                    #   band, FUSED via unit_flag=0b11 (NO software
                                    #   M_FIX flag between mma and fixpipe = the
                                    #   hardware mma->fixpipe pipeline, gemm:951-967;
                                    #   the AscendSyncInsert pass is off by default so
                                    #   nothing is auto-inserted between them).
                                    # pp = nl&1 (0,1,0,1) = gemm tileIdx&1 -> L0AB
                                    # events {4,5}, L0A/L0B slots alternate. cs =
                                    # (1+nl)&1 = 1,0,1,0 (cl0_base=1, no collision with
                                    # QK's slot 0). The L0A/L0B fractals are loaded
                                    # with K=winm (real_k=winm) and the mma contracts
                                    # K=winm (k_actual=winm) -- both MUST match: a full
                                    # K=128 L0 load + a k=winm mma read mismatched
                                    # fractals (M-block/head 16-63 wrong for winm<128).
                                    # Full operands + the real_k/k_actual runtime args
                                    # (NOT symbolic [..,0:winm] slices -- those make
                                    # access_ptr int(winm) a Var and fail). = gemm's
                                    # kSize=winm load + k_actual=winm contract.
                                    # winm<=128 single K tile -> unit_flag always 0b11.
                                    # (G/16)*(128/16)=32>=10 -> no PipeBarrier<PIPE_M>
                                    # (= gemm:944 gate). The per-tile M_MTE1(4/5) chain
                                    # also orders QK's whole-L0 use before these loads,
                                    # so p_l0a/v_l0b sharing QK's A2/B2 space is safe.
                                    for nl in range(PV_NT):
                                        pp = nl % 2
                                        cs = (1 + nl) % 2
                                        T.wait_flag("m", "mte1", L0AB_EV0 + pp)
                                        # real_k=winm: load the L0A/L0B fractals with
                                        # K=winm (NOT the full 128) so they match the
                                        # mma's k_actual=winm -- a full-width L0 load +
                                        # a k=winm mma read mismatched fractals (M-block
                                        # / head 16-63 addressing wrong for winm<128).
                                        T.copy(p_l1[:, :], p_l0a[pp, :, :], real_k=winm)
                                        T.copy(
                                            kv_l1[:, nl * PV_NW : (nl + 1) * PV_NW],
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
                                    # Iteration-boundary full drain (kept while
                                    # DEBUG_SERIAL): PipeBarrier<PIPE_ALL> drains
                                    # every pipe, so it still covers all cross-
                                    # iteration hazards -- q_l1/p_l1/kv_l1 WAR and
                                    # the shared cL0 ping-pong cross-iteration reuse
                                    # (which is why the L0AB flags can be primed once
                                    # with only local per-call iterators here). The
                                    # 3-slot KV ring / 4-slot QP ring / persistent
                                    # cL0BufIter that let this be removed come next
                                    # (increment 3).
                                    if DEBUG_SERIAL:
                                        T.barrier_all()
                            T.set_cross_flag("FIX", EV_PV)
                    # FreeEventID (block_cube.h:239-240): drain the two L0AB
                    # M_MTE1 flags ONCE after the whole cube loop, balancing the
                    # AllocEventID prime above (the shared gemm calls left them
                    # armed instead of self-draining at each call boundary).
                    T.wait_flag("m", "mte1", L0AB_EV0)
                    T.wait_flag("m", "mte1", L0AB_EV1)

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
                                        workspace_s[cid, buf, vid * G2 : (vid + 1) * G2, :],
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
                                        workspace_p[cid, buf, vid * G2 : (vid + 1) * G2, :],
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
                                        workspace_o[cid, bufm, vid * G2 : (vid + 1) * G2, :],
                                        o_ub,
                                    )
                                    T.barrier_all()
                                    # o = o / denom via row broadcast (Brcb + Div,
                                    # src1RepStride=1) over the full D -- no
                                    # [G2,D] denom buffer, faithful to Ascend C
                                    # RowDivs; processes headDim in one pass.
                                    T.tile.row_expand_div(o_ub, o_ub, denom[bufm, :, :], brcb_d)
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
