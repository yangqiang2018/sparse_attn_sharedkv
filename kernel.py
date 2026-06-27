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
            f"only SWA (scenario=1) is implemented; got scenario={scenario} (SCFA/CFA pending)"
        )
    if n_kv_heads != 1 or n_heads != 64 or head_dim != 512:
        raise ValueError(
            f"SWA kernel assumes N1=64, N2=1, D=512 (got N1={n_heads}, N2={n_kv_heads}, D={head_dim})"
        )
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
                                        cl0_base=cl0_iter[0] % 2,
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
                                        cl0_base=cl0_iter[0] % 2,
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
                                    cl0_iter[0] = cl0_iter[0] + 1
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
                                        cs = cl0_iter[0] % 2
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
                                        cl0_iter[0] = cl0_iter[0] + 1
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
