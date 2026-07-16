"""microbench: flash online-softmax —— 三种实现的 msprof aiv_time 对比,分解
"障碍 A(算宽 vs 算窄)" vs "障碍 B(手拼多 pass vs 库融合)",回答 007
(softmax_flash_v2)该保留还是能撤去走手拼、以及 per-row 变长 reduce 值不值得做。

真实场景(kernel.py CFA vector softmax):UB 里 score 是 [M, N] 定长 buffer
(N = S2_BASE = 512),有效窗口只有前 COL 列(ori≈128)。三个 kernel:

  1. flash_v2 (= 007): 输入 [M, N],softmax_flash_v2 compact 前 col_count=COL、变长、
                       库融合,只算 COL 列。基准。
  2. narrow (窄手拼) : 输入 [M, COL] 连续,手拼整块算 COL 列。代理"若有 per-row 变长
                       reduce、只算有效窗口后"的手拼下界。
  3. wide   (满宽手拼): 输入 [M, N] 连续,手拼整块算 N 列。代理"没有变长 reduce、被迫
                       满宽算"的开销(算了 N-COL 列白工)。

⚠️ 为避开 [M,N] buffer 前 COL 的 **strided 子区域**操作(fill/copy/reduce 切片在
   tilelang-ascend 上会踩坑,这正是 007 内部要 compact 的原因),手拼 kernel 一律用
   **[M, width] 连续 buffer 整块算**,输入在 host 端切好、contiguous 后喂进来。
   → narrow 因此是"理想化下界"(假设窗口已连续,省了 007 的 compact 开销);解读时记得。

分解:
  wide  vs narrow    = 障碍 A(算 N 列 vs COL 列的手拼开销差)—— per-row 变长 reduce 能省
  narrow vs flash_v2 = 障碍 B(COL 列手拼多 pass vs 007 库融合)—— 变长 reduce 省不掉
判读:
  narrow ≈ flash_v2 → 融合优势小 → 变长 reduce 能追平 007 → 撤 007 划算
  narrow ≫ flash_v2 → 007 融合真价值 → 保留 007 + 数据硬刚
  wide  ≫ narrow    → 白算是主要损失 → 变长 reduce 价值高

在含 007(softmax_flash_v2)的 build 下跑:
   python microbench_softmax_flash_v2.py --check          # 先验三 kernel 数值各自对
   python microbench_softmax_flash_v2.py --bench --iters 20   # msprof aiv 对比
"""

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys

import tilelang
import tilelang.language as T
import torch

M = 16  # M_CHUNK
N = 512  # S2_BASE (定长 buffer 宽)
COL = 128  # 有效窗口 col_count(ori win_align);actual_col 取同值简化
DTYPE = "float"  # fp32 accumulator
NEG_INF = -3.0e38

TARGET = "ascendc"
PASS_CFG = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def _compile(func):
    return tilelang.compile(func, out_idx=[-1], pass_configs=PASS_CFG, target=TARGET)


# ---- kernel 1: 007 softmax_flash_v2 (输入 [M,N],compact 前 COL) --------------
def build_flash_v2():
    @T.prim_func
    def flash_v2(
        Score: T.Tensor((M, N), DTYPE),  # type: ignore
        Sink: T.Tensor((M, 1), DTYPE),  # type: ignore
        Pout: T.Tensor((M, N), DTYPE),  # type: ignore
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            s_ub = T.alloc_ub((M, N), DTYPE)
            sink_ub = T.alloc_ub((M, 1), DTYPE)
            ones_ub = T.alloc_ub((M, 1), DTYPE)
            osum = T.alloc_ub((M, 1), DTYPE)
            omax = T.alloc_ub((M, 1), DTYPE)
            emax = T.alloc_ub((M, 1), DTYPE)
            tmp = T.alloc_ub((M, N), "uint8")  # NOTE: 007 tmp uint8 scratch;不足就调大
            compact = T.alloc_ub((M, N), DTYPE)
            T.copy(Score, s_ub)
            T.copy(Sink, sink_ub)
            T.tile.fill(ones_ub, 1.0)  # in_sum = 1 (seed)
            T.pipe_barrier("v")
            T.tile.softmax_flash_v2(
                s_ub, osum, omax, emax, s_ub, ones_ub, sink_ub, tmp, compact, COL, COL
            )
            T.pipe_barrier("v")
            T.copy(s_ub, Pout)

    return flash_v2


# ---- kernel 2/3: 手拼 flash softmax,[M, width] 连续 buffer 整块算 -----------
def build_manual(width, name):
    @T.prim_func
    def manual(
        Score: T.Tensor((M, width), DTYPE),  # type: ignore  连续窗口(host 端切好)
        Sink: T.Tensor((M, 1), DTYPE),  # type: ignore
        Pout: T.Tensor((M, width), DTYPE),  # type: ignore
    ):
        T.func_attr(
            {"enable_auto_sync": True}
        )  # 照官方 example:多-pass 依赖链靠 auto_sync 插同步
        with T.Kernel(1, is_npu=True) as (cid, vid):
            s_ub = T.alloc_ub((M, width), DTYPE)
            nmax = T.alloc_ub((M, 1), DTYPE)
            rsum = T.alloc_ub((M, 1), DTYPE)
            nmax_2d = T.alloc_ub((M, width), DTYPE)
            # 前端手拼单块 softmax,严格照官方 examples/softmax/example_online_softmax.py 的
            # 标准写法:reduce_max -> broadcast 到 2D -> 整块 sub -> 整块 exp -> reduce_sum,
            # 全连续整块 + 标准原语(不碰 strided 子区域、不用 experiment;同步交给
            # TL_ASCEND_AUTO_SYNC)。官方 example 对 [64,128] 整块 exp 验证通过,故此处整块
            # exp(width=128/512)走同一条已验证的连续路径。Sink 参数 runner 统一传但不用:
            # seed in_max=-inf 不影响 rowmax,直接 nmax=rowmax。
            T.copy(Score, s_ub)
            T.reduce_max(s_ub, nmax, dim=-1)  # nmax = rowmax
            T.tile.broadcast(nmax_2d, nmax)  # [M,1] -> [M,width]
            T.tile.sub(s_ub, s_ub, nmax_2d)  # score - rowmax
            T.tile.exp(s_ub, s_ub)  # P = exp(score - rowmax)
            T.copy(
                s_ub, Pout
            )  # 先存输出:reduce_sum 会消耗/破坏 src(官方 example 亦 reduce 后即弃 src)
            T.reduce_sum(
                s_ub, rsum, dim=-1
            )  # 再 rowsum(P)(占 softmax pass 开销;其后不再读 s_ub)

    manual.__name__ = name
    return manual


# ---- kernel 4: path B —— 宽 [M,N] 输入,strided-load 前 COL 列进连续 [M,COL] temp,
#      再在连续 temp 上做 narrow softmax(全现成连续原语,零新编译器原语)。--------
def build_block():
    """真实 CFA 场景的 path B 代理:score 在 [M,N=512] buffer 里(和 007 一样),但把有效
    前 COL 列 **strided-load 进连续 [M,COL] temp**,之后 softmax 全在连续 temp 上用现有
    原语算(reduce_max/sub/exp/reduce_sum,任意宽度对连续 buffer 都对)。相比 narrow(连续
    输入)只多一步 strided load。measure:strided load 是否便宜到让 path B ≈ narrow < 007。"""

    @T.prim_func
    def block(
        Score: T.Tensor((M, N), DTYPE),  # type: ignore  宽 buffer(同 007)
        Sink: T.Tensor((M, 1), DTYPE),  # type: ignore
        Pout: T.Tensor((M, N), DTYPE),  # type: ignore
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(1, is_npu=True) as (cid, vid):
            s_ub = T.alloc_ub((M, COL), DTYPE)  # 连续 COL temp
            nmax = T.alloc_ub((M, 1), DTYPE)
            rsum = T.alloc_ub((M, 1), DTYPE)
            nmax_2d = T.alloc_ub((M, COL), DTYPE)
            T.copy(Score[:, 0:COL], s_ub)  # strided load [M,COL]@N -> 连续 temp
            T.reduce_max(s_ub, nmax, dim=-1)
            T.tile.broadcast(nmax_2d, nmax)
            T.tile.sub(s_ub, s_ub, nmax_2d)
            T.tile.exp(s_ub, s_ub)
            T.copy(s_ub, Pout[:, 0:COL])  # strided store 回宽 buffer
            T.reduce_sum(s_ub, rsum, dim=-1)

    block.__name__ = "block"
    return block


# ---- kernel 6: strided-in-place —— 宽 [M,N] 输入,softmax 只碰前 COL 列、**原地**算,
#      不 compact、不加分数 buffer(= 编译器 strided-narrow 原语要让 kernel.py 走的形态)。
#      零编译器积木拼:reduce/exp per-row(唯一能对 512-strided 子区正确的现成写法),
#      sub 走 xattention 的 experiment(brcb + 64 列 row_expand_sub_experiment 循环,全 M 行/次)。
def build_inplace():
    """真实 CFA 场景的 strided-in-place 代理:score 在 [M,N=512] buffer(同 007),softmax
    **原地**只算前 COL 列——不像 007/block 那样 compact 到连续 temp(那块 temp 正是 CFA UB
    装不下的 path B 死因)。零新编译器原语:
      - reduce_max/reduce_sum:per-row(reduce 吃 BufferRegion,单行 [1,COL] 物理连续=正确;
        strided 满宽 reduce 现有原语发不出,per-row 是唯一现成正确写法 → M 次调用)。
      - sub:xattention 的 experiment 写法(brcb rowmax→[M,8],再按 64 列循环
        row_expand_sub_experiment,rep_stride 从物理列 512 推=跨行 strided,全 M 行/次)。
        ★ 同时验证 experiment-sub 在 512-strided buffer 上正确(之前 NaN 是错在 tw>64 一次性调)。
      - exp:per-row(满宽 strided exp 现有原语发不出;单行 [0:COL] 物理连续=正确 → M 次)。
    measure:零编译器 strided-in-place vs 007。唯一低效点是 reduce/exp 的 per-row(M 次);
    若仍追平/超 007 → ① 稳赢;若输且差在 per-row → 正是 strided reduce/exp 原语要消的开销。"""
    brcb_rep = M // 8  # brcb_experiment repeat:M 行 / 8-行块

    @T.prim_func
    def inplace(
        Score: T.Tensor((M, N), DTYPE),  # type: ignore  宽 buffer(同 007)
        Sink: T.Tensor((M, 1), DTYPE),  # type: ignore
        Pout: T.Tensor((M, N), DTYPE),  # type: ignore
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(1, is_npu=True) as (cid, vid):
            s_ub = T.alloc_ub((M, N), DTYPE)  # 512-wide,原地(无 compact temp)
            nmax = T.alloc_ub((M, 1), DTYPE)
            rsum = T.alloc_ub((M, 1), DTYPE)
            brcb_buf = T.alloc_ub((M, 8), DTYPE)  # rowmax 广播目标 [M,8]
            T.copy(Score, s_ub)
            # reduce_max over [0:COL],per-row(单行 [1,COL] 连续)
            for i in range(M):
                T.reduce_max(s_ub[i : i + 1, 0:COL], nmax[i : i + 1, 0:1], dim=-1)
            # sub:广播 rowmax 后 experiment 64 列循环(全 M 行/次,跨行按物理 512 stride)
            T.tile.brcb_experiment(brcb_buf, nmax, brcb_rep, 1, 8)
            for k in range(COL // 64):
                sc = k * 64
                T.tile.row_expand_sub_experiment(
                    s_ub[:, sc : sc + 64], s_ub[:, sc : sc + 64], brcb_buf
                )
            # exp over [0:COL],per-row(单行连续)
            for i in range(M):
                T.tile.exp(s_ub[i, 0:COL], s_ub[i, 0:COL])
            T.copy(
                s_ub, Pout
            )  # 先存 P(整块拷=同 007,apples-to-apples;reduce_sum 破坏 src)
            # reduce_sum over [0:COL],per-row
            for i in range(M):
                T.reduce_sum(s_ub[i : i + 1, 0:COL], rsum[i : i + 1, 0:1], dim=-1)

    inplace.__name__ = "inplace"
    return inplace


# ---- kernel 5: path B 宽-tw —— WIN 列分 BLK 块,2-pass 跨块 softmax(全连续 temp)---
def build_block_online(win, blk, name):
    """宽窗口的 path B:WIN 列分成 nblk 个 BLK 宽块,2-pass 跨块 softmax——
    pass1 逐块 reduce_max→跨块 row max;pass2 逐块 exp(block-rowmax)+reduce_sum+写 P。
    所有算子都在连续 [M,BLK] temp 上(现成原语)。测:多块 + 跨块合并的开销会不会让
    宽-tw path B 反超当前 wide(满宽一发)。strided-load 已证便宜,故 2-pass 重载可接受。"""
    nblk = win // blk

    @T.prim_func
    def bo(
        Score: T.Tensor((M, N), DTYPE),  # type: ignore
        Sink: T.Tensor((M, 1), DTYPE),  # type: ignore
        Pout: T.Tensor((M, N), DTYPE),  # type: ignore
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(1, is_npu=True) as (cid, vid):
            temp = T.alloc_ub((M, blk), DTYPE)  # 连续 BLK temp
            rowmax = T.alloc_ub((M, 1), DTYPE)
            blkred = T.alloc_ub((M, 1), DTYPE)
            rowsum = T.alloc_ub((M, 1), DTYPE)
            max2d = T.alloc_ub((M, blk), DTYPE)
            # pass 1: 跨块 row max
            T.tile.fill(rowmax, NEG_INF)
            for b in range(nblk):
                T.copy(Score[:, b * blk : (b + 1) * blk], temp)
                T.reduce_max(temp, blkred, dim=-1)
                T.tile.max(rowmax, rowmax, blkred)  # dst 第一位(Max(dst,a,b))
            T.tile.broadcast(max2d, rowmax)
            # pass 2: exp(block-rowmax) + 累加 sum + 写 P
            T.tile.fill(rowsum, 0.0)
            for b in range(nblk):
                T.copy(Score[:, b * blk : (b + 1) * blk], temp)
                T.tile.sub(temp, temp, max2d)
                T.tile.exp(temp, temp)
                T.copy(
                    temp, Pout[:, b * blk : (b + 1) * blk]
                )  # 先写(reduce_sum 破坏 src)
                T.reduce_sum(temp, blkred, dim=-1)
                T.tile.add(rowsum, rowsum, blkred)

    bo.__name__ = name
    return bo


def build_dbg_reduce(width, name):
    """只测 reduce_max+sub: copy -> reduce_max -> broadcast -> sub,输出 s_ub = score - nmax。
    reduce_max 归约整行则每元素 <= 0;若 nmax 偏小(只归约了部分列)则会出现 > 0。"""

    @T.prim_func
    def dbg(
        Score: T.Tensor((M, width), DTYPE),  # type: ignore
        Sink: T.Tensor((M, 1), DTYPE),  # type: ignore
        Pout: T.Tensor((M, width), DTYPE),  # type: ignore
    ):
        T.func_attr(
            {"enable_auto_sync": True}
        )  # 照官方 example:多-pass 依赖链靠 auto_sync 插同步
        with T.Kernel(1, is_npu=True) as (cid, vid):
            s_ub = T.alloc_ub((M, width), DTYPE)
            nmax = T.alloc_ub((M, 1), DTYPE)
            nmax_2d = T.alloc_ub((M, width), DTYPE)
            T.copy(Score, s_ub)
            T.reduce_max(s_ub, nmax, dim=-1)
            T.tile.broadcast(nmax_2d, nmax)
            T.tile.sub(s_ub, s_ub, nmax_2d)
            T.copy(s_ub, Pout)

    dbg.__name__ = name
    return dbg


def build_dbg_exp(width, name):
    """隔离 exp 段:copy→reduce_max→broadcast→sub→exp→copy(Pout),不含 reduce_sum。
    对则 exp 无辜、罪在 reduce_sum(内部 scratch 复用/污染了随后还要 copy 出的 s_ub);
    错/波动则罪在 exp 本身。"""

    @T.prim_func
    def dbg(
        Score: T.Tensor((M, width), DTYPE),  # type: ignore
        Sink: T.Tensor((M, 1), DTYPE),  # type: ignore
        Pout: T.Tensor((M, width), DTYPE),  # type: ignore
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(1, is_npu=True) as (cid, vid):
            s_ub = T.alloc_ub((M, width), DTYPE)
            nmax = T.alloc_ub((M, 1), DTYPE)
            nmax_2d = T.alloc_ub((M, width), DTYPE)
            T.copy(Score, s_ub)
            T.reduce_max(s_ub, nmax, dim=-1)
            T.tile.broadcast(nmax_2d, nmax)
            T.tile.sub(s_ub, s_ub, nmax_2d)
            T.tile.exp(s_ub, s_ub)
            T.copy(s_ub, Pout)

    dbg.__name__ = name
    return dbg


# KERNELS[name] = (builder, in_width, cmp_width)
KERNELS = {
    "flash_v2": (build_flash_v2, N, COL),
    "narrow": (lambda: build_manual(COL, "narrow"), COL, COL),
    "wide": (lambda: build_manual(N, "wide"), N, N),
    "block": (
        build_block,
        N,
        COL,
    ),  # path B: 宽输入 + strided-load 连续 COL + narrow softmax
    # strided-in-place:宽输入,原地只算 COL 列(无 compact),reduce/exp per-row + experiment sub
    "inplace": (build_inplace, N, COL),
    # path B 宽-tw:多块 2-pass。bo256=部分宽(2 块,应超 wide);bo512=满宽(4 块,测 rescale 开销)
    "bo256": (lambda: build_block_online(256, 128, "bo256"), N, 256),
    "bo512": (lambda: build_block_online(512, 128, "bo512"), N, 512),
}


def _score_full():
    torch.manual_seed(0)
    return torch.randn(M, N, dtype=torch.float32)  # 所有 kernel 的前若干列取自同一份


def _golden(cmp_w):
    # 与最小版 kernel 对齐:P = exp(score - rowmax)。flash_v2 的 sink=-inf 也不影响 rowmax。
    s = _score_full()[:, :cmp_w].float()
    nmax = s.max(dim=-1, keepdim=True).values
    return torch.exp(s - nmax)  # [M, cmp_w]


def check():
    sf = _score_full()
    sink = torch.full((M, 1), NEG_INF, dtype=torch.float32)
    for name, (builder, in_w, cmp_w) in KERNELS.items():
        func = _compile(builder())
        a = sf[:, :in_w].contiguous().npu()
        sk = sink.npu()
        torch.npu.synchronize()
        p = func(a, sk)
        torch.npu.synchronize()
        got = p.cpu()[:, :cmp_w].float()
        ref = _golden(cmp_w)
        ok = torch.allclose(got, ref, rtol=2e-2, atol=2e-2)
        md = (got - ref).abs().max().item()
        print(f"  [{name:8s}] cmp_w={cmp_w} match={ok}  max|diff|={md:.4g}")


def dbg_reduce():
    """决定性单测: reduce_max 是否归约整行。输出 score-nmax,对则 max<=0;偏则 >0。"""
    sf = _score_full()
    sink = torch.full((M, 1), NEG_INF, dtype=torch.float32).npu()
    for w in (64, COL, N):
        func = _compile(build_dbg_reduce(w, f"dbg{w}"))
        a = sf[:, :w].contiguous().npu()
        torch.npu.synchronize()
        p = func(a, sink)
        torch.npu.synchronize()
        got = p.cpu().float()  # score - nmax
        s = sf[:, :w].float()
        golden = s - s.max(dim=-1, keepdim=True).values  # score - rowmax (<=0)
        gotmax = got.max().item()
        diff = (got - golden).abs().max().item()
        flag = (
            "OK(reduce 归约整行)" if gotmax <= 0.02 else "!! nmax 偏小(只归约了部分列)"
        )
        print(
            f"  [reduce w={w:3d}] (score-nmax).max()={gotmax:+.4g}  golden.diff={diff:.4g}  {flag}"
        )


def dbg_exp():
    """隔离 exp 段(比 dbgreduce 多 exp、比 check 少 reduce_sum)。每 width 连跑两次:
    golden.diff 大 = exp 段错;run2run>0 = race(两次不一致)。据此分清 exp vs reduce_sum。"""
    sf = _score_full()
    sink = torch.full((M, 1), NEG_INF, dtype=torch.float32).npu()
    for w in (64, COL, N):
        func = _compile(build_dbg_exp(w, f"dbgexp{w}"))
        a = sf[:, :w].contiguous().npu()
        s = sf[:, :w].float()
        golden = torch.exp(s - s.max(dim=-1, keepdim=True).values)  # exp(score-rowmax)
        outs = []
        for _ in range(2):
            torch.npu.synchronize()
            p = func(a, sink)
            torch.npu.synchronize()
            outs.append(p.cpu().float())
        diff = (outs[0] - golden).abs().max().item()
        run2run = (outs[0] - outs[1]).abs().max().item()
        tag = "OK" if diff <= 2e-2 else "!! exp 段错"
        race = " RACE(两次不一致)" if run2run > 1e-6 else ""
        print(
            f"  [exp w={w:3d}] golden.diff={diff:.4g}  run2run={run2run:.4g}  {tag}{race}"
        )


def run_one(name, iters):
    builder, in_w, _ = KERNELS[name]
    func = _compile(builder())
    a = _score_full()[:, :in_w].contiguous().npu()
    sk = torch.full((M, 1), NEG_INF, dtype=torch.float32).npu()
    torch.npu.synchronize()
    for _ in range(iters):
        func(a, sk)
    torch.npu.synchronize()


def _collect(name, iters, outdir):
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)
    app = f"python {os.path.abspath(__file__)} --kernel {name} --iters {iters}"
    cmd = [
        "msprof",
        f"--output={outdir}",
        f"--application={app}",
        "--ai-core=on",
        "--aic-metrics=PipeUtilization",
    ]
    print(f"  [{name:8s}] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    pat = os.path.join(outdir, "**", "*op_summary*.csv")
    if not glob.glob(pat, recursive=True):
        for pd in glob.glob(os.path.join(outdir, "PROF_*")):
            subprocess.run(["msprof", "--export=on", f"--output={pd}"], check=False)
    cands = sorted(glob.glob(pat, recursive=True), key=os.path.getmtime)
    return cands[-1] if cands else None


def _durs(csv_path, match, col_substr):
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    hdr = [h.strip().lower() for h in rows[0]]

    def find(*subs):
        for s in subs:
            for i, h in enumerate(hdr):
                if s in h.replace(" ", "").replace("(us)", ""):
                    return i
        return -1

    nc = find("opname", "name")
    dc = find(col_substr, "aivtime", "aiv_time", "taskduration", "duration")
    out = []
    for r in rows[1:]:
        if 0 <= nc < len(r) and 0 <= dc < len(r) and match in r[nc].strip().lower():
            try:
                out.append(float(r[dc].strip()))
            except ValueError:
                pass
    return out


def bench(iters, drop, col_substr):
    print(
        f"\nshape M={M} N={N} COL={COL} dtype={DTYPE}; iters={iters} drop-cold={drop}"
    )
    res = {}
    for name in ("flash_v2", "narrow", "wide", "block", "inplace", "bo256", "bo512"):
        csv_path = _collect(name, iters, f"/tmp/msbench_{name}")
        if not csv_path:
            print(f"  [{name:8s}] no CSV")
            continue
        key = "flash" if name == "flash_v2" else name
        d = _durs(csv_path, key, col_substr) or _durs(csv_path, "", col_substr)
        warm = d[drop:] if len(d) > drop else d
        avg = sum(warm) / len(warm) if warm else None
        res[name] = avg
        print(
            f"  {name:8s}  avg_aiv={avg if avg is None else round(avg, 3)} us  ({len(warm)} warm)"
        )
    if all(res.get(k) for k in ("flash_v2", "narrow", "wide")):
        A = res["wide"] - res["narrow"]
        B = res["narrow"] - res["flash_v2"]
        print(f"\n障碍 A(算宽,wide-narrow)   = {A:.3f} us")
        print(f"障碍 B(融合,narrow-flash_v2) = {B:.3f} us")
        print(f"总差(wide-flash_v2)        = {res['wide'] - res['flash_v2']:.3f} us")
        print(
            "判读: B 小→变长手拼能追平 007(撤划算); B 大→007 融合真价值(保留); A 大→变长 reduce 价值高"
        )
    if res.get("block") and res.get("flash_v2") and res.get("narrow"):
        C = res["block"] - res["flash_v2"]
        D = res["block"] - res["narrow"]
        print(f"\n[path B] block(strided-load+连续 narrow) = {res['block']:.3f} us")
        print(f"  block - flash_v2 = {C:.3f} us  (<=0 → path B 追平/超 007,值得建)")
        print(f"  block - narrow   = {D:.3f} us  (strided-load 相比连续输入的额外开销)")
    if res.get("inplace") and res.get("flash_v2"):
        E = res["inplace"] - res["flash_v2"]
        print(
            "\n[strided-in-place] inplace(原地,无 compact;reduce/exp per-row+experiment sub)"
        )
        print(
            f"  inplace = {res['inplace']:.3f} us  vs flash_v2(007) {res['flash_v2']:.3f}"
        )
        print(
            f"  inplace - flash_v2 = {E:.3f} us  "
            "(<=0 → 零编译器 strided-in-place 已追平/超 007,① 稳赢;"
            ">0 → per-row reduce/exp 是差距 → strided reduce/exp 原语要消的就是它)"
        )
        if res.get("block"):
            print(
                f"  inplace - block   = {res['inplace'] - res['block']:.3f} us  "
                "(strided-in-place per-row 相比 compact+连续 的代价)"
            )
    if res.get("bo256") and res.get("bo512") and res.get("wide"):
        print("\n[path B 宽-tw] 多块 2-pass block-online:")
        print(
            f"  bo256(2 块, tw=256) = {res['bo256']:.3f} us  vs wide(满512) {res['wide']:.3f}"
        )
        print(
            f"    bo256 - wide = {res['bo256'] - res['wide']:.3f} us  (<0 → 部分宽 tile 也省)"
        )
        print(
            f"  bo512(4 块, tw=512) = {res['bo512']:.3f} us  vs wide(满512) {res['wide']:.3f}"
        )
        print(
            f"    bo512 - wide = {res['bo512'] - res['wide']:.3f} us  "
            "(<=0 → 满宽 tile 也不亏,path B 全程可用; >0 → 满宽 tile 走 hybrid)"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--kernel", choices=list(KERNELS))
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--drop", type=int, default=2)
    ap.add_argument("--col", default="aiv", help="msprof 时间列关键词(aiv/duration)")
    ap.add_argument("--dump", choices=list(KERNELS), help="打印某 kernel 的 codegen")
    ap.add_argument("--dbgreduce", action="store_true", help="决定性单测 reduce_max")
    ap.add_argument(
        "--dbgexp", action="store_true", help="隔离 exp 段(分清 exp vs reduce_sum)"
    )
    args = ap.parse_args()
    tilelang.disable_cache()
    tilelang.cache.clear_cache()  # 清磁盘缓存:改了 kernel 实现却命中旧编译产物会静默复用旧 kernel
    if args.dbgreduce:
        dbg_reduce()
    elif args.dbgexp:
        dbg_exp()
    elif args.dump:
        print(_compile(KERNELS[args.dump][0]()).get_kernel_source())
    elif args.kernel:
        run_one(args.kernel, args.iters)
    elif args.check:
        check()
    elif args.bench:
        bench(args.iters, args.drop, args.col)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
