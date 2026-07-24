"""sparse_flash_mla(上游 example)的 TileLang vs Ascend C 性能计分板(msprof 口径,单文件)。

跟根目录的 ``msprof_compare.py`` 同款两层口径,但**合并成一个文件**:

* orchestrator(默认模式):对每个场景各跑两次 msprof(一次 Ascend C、一次 TileLang),
  从 op_summary CSV 读该 op 的 **device Task Duration**,丢掉第 1 条(冷启动)、其余取平均,
  打印 ``ascendc / tilelang`` 倍率(>1.0 = TileLang 更快)。测的是 msprof 抓的 device kernel
  时长,不含 host 下发/同步开销 —— 和历史台账严格同口径。
* ``--run-once``(内部模式):被 msprof profile 的 launch 靶子 —— build 一次输入、反复 launch
  kernel,不计时不打印(计时全交给外层 msprof)。orchestrator 通过
  ``msprof --application="python <本文件> --run-once ..."`` 调它,一般不手动跑。

两侧 op 名不同(Ascend C 是 ``npu_sparse_attn_sharedkv``、TileLang 上游版是
``sparse_flash_mla``),所以过滤串分开给 —— ``--ac-match`` / ``--tl-match``。

在 NPU 容器里、cd 到本目录后跑(要 ``msprof`` + 同目录的 api/kernel/golden;Ascend C 侧还要
``custom_ops`` 已装)。容器专用文件,不合入官方 example。

用法
----
  # 全量:swa/hca/csa × prefill/decode = 6 个场景(bf16):
      python sparse_flash_mla_msprof_compare.py

  # 只 prefill 三场景 / 单场景:
      python sparse_flash_mla_msprof_compare.py --phases prefill
      python sparse_flash_mla_msprof_compare.py --ops csa --phases decode

若 TileLang 侧报 "no row matched",看它打印的 distinct ops,用 ``--tl-match`` 覆盖成 CSV 里
真实的 TileLang op 名。
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

# 让 --run-once 子进程能 import 同目录的 api / golden(不管 msprof 从哪个 cwd 启动)。
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

OPS = ["swa", "hca", "csa"]
PHASES = ["prefill", "decode"]

# ---- 场景表:prefill 驱动 S1=8192(8K 长序列);decode S1=1。与 Ascend C / TileLang 参数集 1:1。 ----
SCENARIOS = {
    "csa_prefill": dict(
        scenario=3,
        layout_q="TND",
        B=1,
        S1=8192,
        T1=8192,
        N1=64,
        N2=1,
        D=512,
        K=512,
        block_num1=65,
        block_num2=17,
        block_size1=128,
        block_size2=128,
        cu_seqlens_q=[0, 8192],
        seqused_kv=[8192],
        softmax_scale=0.04419417,
        cmp_ratio=4,
        ori_win_left=127,
        ori_win_right=0,
        ori_mask_mode=4,
        cmp_mask_mode=3,
    ),
    "swa_prefill": dict(
        scenario=1,
        layout_q="TND",
        B=1,
        S1=8192,
        T1=8192,
        N1=64,
        N2=1,
        D=512,
        K=0,
        block_num1=65,
        block_num2=1,
        block_size1=128,
        block_size2=1,
        cu_seqlens_q=[0, 8192],
        seqused_kv=[8192],
        softmax_scale=0.04419417,
        cmp_ratio=4,
        ori_win_left=127,
        ori_win_right=0,
        ori_mask_mode=4,
        cmp_mask_mode=3,
    ),
    "hca_prefill": dict(
        scenario=2,
        layout_q="TND",
        B=1,
        S1=8192,
        T1=8192,
        N1=64,
        N2=1,
        D=512,
        K=512,
        block_num1=65,
        block_num2=17,
        block_size1=128,
        block_size2=128,
        cu_seqlens_q=[0, 8192],
        seqused_kv=[8192],
        softmax_scale=0.04419417,
        cmp_ratio=128,
        ori_win_left=127,
        ori_win_right=0,
        ori_mask_mode=4,
        cmp_mask_mode=3,
    ),
    "csa_decode": dict(
        scenario=3,
        layout_q="TND",
        B=1,
        S1=1,
        T1=1,
        N1=64,
        N2=1,
        D=512,
        K=512,
        block_num1=65,
        block_num2=17,
        block_size1=128,
        block_size2=128,
        cu_seqlens_q=[0, 1],
        seqused_kv=[8193],
        softmax_scale=0.04419417,
        cmp_ratio=4,
        ori_win_left=127,
        ori_win_right=0,
        ori_mask_mode=4,
        cmp_mask_mode=3,
    ),
    "swa_decode": dict(
        scenario=1,
        layout_q="TND",
        B=1,
        S1=1,
        T1=1,
        N1=64,
        N2=1,
        D=512,
        K=0,
        block_num1=65,
        block_num2=1,
        block_size1=128,
        block_size2=1,
        cu_seqlens_q=[0, 1],
        seqused_kv=[8193],
        softmax_scale=0.04419417,
        cmp_ratio=4,
        ori_win_left=127,
        ori_win_right=0,
        ori_mask_mode=4,
        cmp_mask_mode=3,
    ),
    "hca_decode": dict(
        scenario=2,
        layout_q="TND",
        B=1,
        S1=1,
        T1=1,
        N1=64,
        N2=1,
        D=512,
        K=512,
        block_num1=65,
        block_num2=17,
        block_size1=128,
        block_size2=128,
        cu_seqlens_q=[0, 1],
        seqused_kv=[8193],
        softmax_scale=0.04419417,
        cmp_ratio=128,
        ori_win_left=127,
        ori_win_right=0,
        ori_mask_mode=4,
        cmp_mask_mode=3,
    ),
}


# =============================================================================
# --run-once 的 launch 靶子:build 输入 + 反复 launch kernel,不计时(计时交给外层 msprof)。
# =============================================================================
def create_tensor_with_stride_padding(src_tensor, pad_len):
    """Ascend C PA_ND 的 stride 补齐(块行之间多 ``pad_len`` 个元素的跨距),逐字复刻参考流程,
    让 Ascend C 侧看到的正是它自家测试喂进去的张量。"""
    import torch_npu  # noqa: F401  (registers the npu backend)

    if src_tensor is None or src_tensor.dim() != 4:
        return src_tensor

    device = src_tensor.device
    dtype = src_tensor.dtype
    shape = list(src_tensor.shape)  # e.g. [65, 128, 1, 512]

    row_logical_size = shape[1] * shape[2] * shape[3]
    stride_0 = row_logical_size + pad_len
    new_strides = [stride_0, shape[2] * shape[3], shape[3], 1]

    physical_numel = stride_0 * shape[0]
    storage_tensor = torch.full(
        (physical_numel,), float("nan"), dtype=dtype, device=device
    )
    raw_storage = storage_tensor.untyped_storage()

    target_tensor = torch.empty((0,), dtype=dtype, device=device)
    target_tensor.set_(raw_storage, 0, shape, new_strides)

    for i in range(shape[0]):
        target_tensor[i].copy_(src_tensor[i])

    torch_npu.npu.synchronize()
    return target_tensor


def build_inputs(cfg, dtype, seed=42):
    """在 CPU 上生成分页 KV / Q / indices(复用 golden 的数据生成器)。这是性能压测,不算 CPU
    golden —— 只有形状/dtype 要紧,两侧吃同一批张量,workload 一致。"""
    import sparse_flash_mla_golden as G  # local import: same dir on sys.path

    torch.manual_seed(seed)
    np.random.seed(seed)

    layout_q = cfg["layout_q"]
    B, S1 = cfg["B"], cfg["S1"]
    T1 = cfg.get("T1", S1 * B)
    N1, N2, D = cfg["N1"], cfg["N2"], cfg["D"]
    K = cfg.get("K", 0)
    cmp_ratio = cfg["cmp_ratio"]
    bs1, bn1 = cfg["block_size1"], cfg["block_num1"]
    bs2, bn2 = cfg.get("block_size2"), cfg.get("block_num2")
    seqused_kv = cfg["seqused_kv"]
    cu = torch.tensor(cfg["cu_seqlens_q"], dtype=torch.int32)
    scenario = cfg["scenario"]

    if layout_q == "TND":
        q = (torch.rand((T1, N1, D)) * 20 - 10).to(dtype)
    else:
        q = (torch.rand((B, S1, N1, D)) * 20 - 10).to(dtype)

    ori_pa, ori_bt, _ = G.gen_ori_kv_paged(
        B=B,
        N2=N2,
        D=D,
        block_num=bn1,
        block_size=bs1,
        seqused_kv=seqused_kv,
        dtype=dtype,
        data_range=(-10, 10),
        rng=np.random.default_rng(seed),
    )

    if scenario >= 2:
        cmp_pa, cmp_bt, _, _ = G.gen_cmp_kv_paged(
            B=B,
            N2=N2,
            D=D,
            block_num=bn2,
            block_size=bs2,
            seqused_kv=seqused_kv,
            cmp_ratio=cmp_ratio,
            dtype=dtype,
            data_range=(-5, 10),
            rng=np.random.default_rng(seed + 1),
        )
    else:
        cmp_pa, cmp_bt = None, None

    if scenario == 3:
        g = torch.Generator()
        g.manual_seed(seed + 7)
        cmp_idx = G.gen_cmp_sparse_indices_tnd(
            B=B,
            T1=T1,
            N2=N2,
            K=K,
            cu_seqlens_q=cu,
            seqused_kv=seqused_kv,
            cmp_ratio=cmp_ratio,
            cmp_mask_mode=3,
            rng=g,
        )
    else:
        cmp_idx = None

    sinks = (torch.rand(N1) * 2 - 1).to(torch.float32)
    return dict(
        q=q,
        ori_pa=ori_pa,
        ori_bt=ori_bt,
        cmp_pa=cmp_pa,
        cmp_bt=cmp_bt,
        cmp_idx=cmp_idx,
        sinks=sinks,
        cu_seqlens_q=cu,
        seqused_kv=torch.tensor(seqused_kv, dtype=torch.int32),
    )


def stage_on_npu(c, stride_pad=True):
    """把输入一次性搬上 NPU。共享只读张量只上传一次;Ascend C 侧额外收一份 stride 补齐的分页 KV
    (它的参考约定),TileLang 的 api 会 ``.contiguous()`` 用普通分页布局。"""
    import torch_npu  # noqa: F401

    def dev(t):
        return None if t is None else t.npu().contiguous()

    inp = dict(
        q_npu=dev(c["q"]),
        sinks_npu=dev(c["sinks"]),
        cu_seqlens_q_npu=dev(c["cu_seqlens_q"]),
        cu_seqlens_q_cpu=c["cu_seqlens_q"],
        seqused_kv_npu=dev(c["seqused_kv"]),
        seqused_kv_cpu=c["seqused_kv"],
        ori_bt_npu=dev(c["ori_bt"]),
        cmp_bt_npu=dev(c["cmp_bt"]),
        cmp_idx_npu=dev(c["cmp_idx"]),
        ori_pa_npu=dev(c["ori_pa"]),  # TileLang: contiguous paged KV
        cmp_pa_npu=dev(c["cmp_pa"]),
        empty_npu=torch.tensor([]).npu(),
    )
    if stride_pad:
        inp["ori_pa_pad_npu"] = create_tensor_with_stride_padding(dev(c["ori_pa"]), 64)
        inp["cmp_pa_pad_npu"] = (
            create_tensor_with_stride_padding(dev(c["cmp_pa"]), 64)
            if c["cmp_pa"] is not None
            else None
        )
    else:
        inp["ori_pa_pad_npu"] = inp["ori_pa_npu"]
        inp["cmp_pa_pad_npu"] = inp["cmp_pa_npu"]
    return inp


def tilelang_sharedkv(inp, cfg):
    """Launch 一次 TileLang sharedkv op。TileLang 端不消费调度 metadata(kernel 用静态均分),
    所以没有单独的 metadata op。"""
    from sparse_flash_mla_api import sparse_flash_mla as tl_sharedkv

    scenario = cfg["scenario"]
    K = cfg.get("K", 0)
    layout_q = cfg["layout_q"]
    with torch.device("npu"):
        tl_sharedkv(
            inp["q_npu"],
            ori_kv=inp["ori_pa_npu"],
            cmp_kv=inp["cmp_pa_npu"],
            cmp_sparse_indices=inp["cmp_idx_npu"],
            ori_block_table=inp["ori_bt_npu"],
            cmp_block_table=inp["cmp_bt_npu"],
            cu_seqlens_q=inp["cu_seqlens_q_npu"] if layout_q == "TND" else None,
            seqused_kv=inp["seqused_kv_npu"],
            sinks=inp["sinks_npu"],
            softmax_scale=cfg["softmax_scale"],
            cmp_ratio=cfg["cmp_ratio"] if scenario >= 2 else None,
            ori_mask_mode=cfg["ori_mask_mode"],
            cmp_mask_mode=cfg["cmp_mask_mode"],
            ori_win_left=cfg["ori_win_left"],
            ori_win_right=cfg["ori_win_right"],
            layout_q=layout_q,
            layout_kv="PA_ND",
            return_softmax_lse=True,
            topk_cmp=K,
        )


def ascendc_metadata(inp, cfg):
    """Build Ascend C 的 metadata 张量 —— 独立的伴生 op(在 AI CPU 上跑)。sharedkv 依赖它。"""
    import torch_npu

    scenario = cfg["scenario"]
    N1, N2, D, B = cfg["N1"], cfg["N2"], cfg["D"], cfg["B"]
    K = cfg.get("K", 0)
    layout_q = cfg["layout_q"]
    max_seqlen_q = cfg.get("T1") if layout_q == "TND" else cfg["S1"]
    ori_max_s2 = int(max(cfg["seqused_kv"]))
    empty = inp["empty_npu"]

    md_kwargs = dict(
        num_heads_q=N1,
        num_heads_kv=N2,
        head_dim=D,
        cu_seqlens_q=inp["cu_seqlens_q_npu"],
        cu_seqlens_ori_kv=empty,
        cu_seqlens_cmp_kv=empty,
        seqused_q=empty,
        seqused_kv=inp["seqused_kv_npu"],
        batch_size=B,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=ori_max_s2,
        ori_mask_mode=cfg["ori_mask_mode"],
        ori_win_left=cfg["ori_win_left"],
        ori_win_right=cfg["ori_win_right"],
        layout_q=layout_q,
        layout_kv="PA_ND",
        has_ori_kv=True,
        has_cmp_kv=scenario >= 2,
        device="npu:0",
    )
    if scenario == 2:
        md_kwargs["cmp_ratio"] = cfg["cmp_ratio"]
        md_kwargs["cmp_mask_mode"] = cfg["cmp_mask_mode"]
    elif scenario == 3:
        md_kwargs["cmp_topk"] = K
        md_kwargs["cmp_ratio"] = cfg["cmp_ratio"]
        md_kwargs["cmp_mask_mode"] = cfg["cmp_mask_mode"]
    return torch_npu.npu_sparse_attn_sharedkv_metadata(**md_kwargs)


def ascendc_sharedkv(inp, cfg, md):
    """Launch 一次 Ascend C sharedkv op(用预算好的 metadata ``md``)。"""
    import torch_npu  # noqa: F401  (registers torch.ops.custom)

    scenario = cfg["scenario"]
    layout_q = cfg["layout_q"]
    cu_q = inp["cu_seqlens_q_npu"] if layout_q == "TND" else None
    if scenario == 1:  # SWA
        torch.ops.custom.npu_sparse_attn_sharedkv(
            inp["q_npu"],
            ori_kv=inp["ori_pa_pad_npu"],
            ori_block_table=inp["ori_bt_npu"],
            cu_seqlens_q=cu_q,
            seqused_q=None,
            seqused_kv=inp["seqused_kv_npu"],
            sinks=inp["sinks_npu"],
            metadata=md,
            softmax_scale=cfg["softmax_scale"],
            ori_mask_mode=cfg["ori_mask_mode"],
            ori_win_left=cfg["ori_win_left"],
            ori_win_right=cfg["ori_win_right"],
            layout_q=layout_q,
            layout_kv="PA_ND",
        )
    elif scenario == 2:  # HCA
        torch.ops.custom.npu_sparse_attn_sharedkv(
            inp["q_npu"],
            ori_kv=inp["ori_pa_pad_npu"],
            cmp_kv=inp["cmp_pa_pad_npu"],
            ori_block_table=inp["ori_bt_npu"],
            cmp_block_table=inp["cmp_bt_npu"],
            cu_seqlens_q=cu_q,
            seqused_q=None,
            seqused_kv=inp["seqused_kv_npu"],
            sinks=inp["sinks_npu"],
            metadata=md,
            softmax_scale=cfg["softmax_scale"],
            cmp_ratio=cfg["cmp_ratio"],
            ori_mask_mode=cfg["ori_mask_mode"],
            cmp_mask_mode=cfg["cmp_mask_mode"],
            ori_win_left=cfg["ori_win_left"],
            ori_win_right=cfg["ori_win_right"],
            layout_q=layout_q,
            layout_kv="PA_ND",
        )
    else:  # CSA
        torch.ops.custom.npu_sparse_attn_sharedkv(
            inp["q_npu"],
            ori_kv=inp["ori_pa_pad_npu"],
            cmp_kv=inp["cmp_pa_pad_npu"],
            cmp_sparse_indices=inp["cmp_idx_npu"],
            ori_block_table=inp["ori_bt_npu"],
            cmp_block_table=inp["cmp_bt_npu"],
            cu_seqlens_q=cu_q,
            seqused_q=None,
            seqused_kv=inp["seqused_kv_npu"],
            sinks=inp["sinks_npu"],
            metadata=md,
            softmax_scale=cfg["softmax_scale"],
            cmp_ratio=cfg["cmp_ratio"],
            ori_mask_mode=cfg["ori_mask_mode"],
            cmp_mask_mode=cfg["cmp_mask_mode"],
            ori_win_left=cfg["ori_win_left"],
            ori_win_right=cfg["ori_win_right"],
            layout_q=layout_q,
            layout_kv="PA_ND",
        )


def run_once(scenario, only, dtype_str, warmup, iters, stride_pad):
    """--run-once 模式:build 一次输入,把对应实现 launch ``warmup + iters`` 次供 msprof 抓,
    不计时不打印。``only`` 只接受 'ascendc' / 'tilelang'(orchestrator 逐侧单跑)。"""
    import torch_npu  # noqa: F401

    if only not in ("ascendc", "tilelang"):
        print(
            f"[fatal] --run-once needs --only ascendc|tilelang (got {only!r}).",
            file=sys.stderr,
        )
        return 2
    if not torch.npu.is_available():
        print(
            "[fatal] torch.npu.is_available() == False; need an NPU.", file=sys.stderr
        )
        return 2
    if only == "ascendc":
        try:
            import custom_ops  # noqa: F401  (registers torch.ops.custom.*)
        except Exception as exc:  # pragma: no cover - host dependent
            print(
                f"[fatal] could not import custom_ops (Ascend C op): {exc!r}",
                file=sys.stderr,
            )
            return 2

    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    cfg = SCENARIOS[scenario]
    inp = stage_on_npu(build_inputs(cfg, dtype), stride_pad=stride_pad)

    torch.npu.synchronize()
    total = warmup + iters
    if only == "tilelang":
        for _ in range(total):
            tilelang_sharedkv(inp, cfg)
    else:  # ascendc: metadata + sharedkv each launch; msprof reads sharedkv op, excludes metadata
        for _ in range(total):
            md = ascendc_metadata(inp, cfg)
            ascendc_sharedkv(inp, cfg, md)
    torch.npu.synchronize()
    return 0


# =============================================================================
# msprof orchestrator(默认模式):profile 本文件的 --run-once,读 device Task Duration。
# =============================================================================
def _norm(s: str) -> str:
    """Lowercase + drop non-alphanumerics (so 'Task Duration(us)' ~ 'taskduration')."""
    return "".join(c for c in s.lower() if c.isalnum())


def _find_col(header: list[str], *substr_priorities: str) -> int:
    """Index of the first header whose normalized form contains one of the given
    normalized substrings, tried in priority order; -1 if none."""
    norm = [_norm(h) for h in header]
    for want in substr_priorities:
        w = _norm(want)
        for i, h in enumerate(norm):
            if w in h:
                return i
    return -1


def _name_col(header: list[str]) -> int:
    i = _find_col(header, "op name", "op_name", "opname", "op type", "name")
    return i if i >= 0 else 0


def collect(impl, scenario, dtype, warmup, iters, outdir, no_stride_pad):
    """Run msprof over this file's --run-once for one (impl, scenario); return the
    newest op_summary CSV path, or None on failure."""
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)
    me = os.path.basename(__file__)
    pad_flag = " --no-stride-pad" if no_stride_pad else ""
    app = (
        f"python {me} --run-once --scenario {scenario} --only {impl} "
        f"--dtype {dtype} --warmup {warmup} --iters {iters}{pad_flag}"
    )
    cmd = [
        "msprof",
        f"--output={outdir}",
        f"--application={app}",
        "--ai-core=on",
        "--aic-metrics=PipeUtilization",
    ]
    print(f"  [{impl:8s}] running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  msprof failed ({impl}/{scenario}): {e}")
        return None
    # Newer msprof collects into PROF_* but doesn't auto-export the CSV.
    pat = os.path.join(outdir, "**", "*op_summary*.csv")
    if not glob.glob(pat, recursive=True):
        for prof_dir in glob.glob(os.path.join(outdir, "PROF_*")):
            try:
                subprocess.run(
                    ["msprof", "--export=on", f"--output={prof_dir}"], check=True
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"  export failed for {prof_dir}: {e}")
    cands = glob.glob(pat, recursive=True)
    cands.sort(key=os.path.getmtime)
    if not cands:
        print(f"  no op_summary CSV under {outdir} (run msprof --export=on manually).")
        return None
    return cands[-1]


def op_durations(csv_path, match, exclude, duration_col):
    """Return (durations, info): the per-launch Task Duration floats for the matched
    op (chronological if a start-time column exists), plus a one-line info string."""
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], "empty CSV"
    header = rows[0]
    nc = _name_col(header)
    dc = _find_col(header, duration_col, "task duration", "duration")
    if dc < 0:
        return [], f"no duration column (headers: {header})"
    sc = _find_col(header, "task start time", "start time", "op start", "timestamp")

    nmatch, nexcl = _norm(match), (_norm(exclude) if exclude else "")
    hits, distinct = [], set()
    for r in rows[1:]:
        if nc >= len(r) or dc >= len(r):
            continue
        nm = r[nc].strip()
        distinct.add(nm)
        nn = _norm(nm)
        if nmatch in nn and (not nexcl or nexcl not in nn):
            try:
                dur = float(r[dc].strip())
            except ValueError:
                continue
            key = None
            if 0 <= sc < len(r):
                try:
                    key = float(r[sc].strip())
                except ValueError:
                    key = None
            hits.append((key, dur))
    if not hits:
        return [], f"no row matched {match!r}; distinct ops: {sorted(distinct)[:20]}"
    # chronological order if we have start times, else CSV order.
    if all(k is not None for k, _ in hits):
        hits.sort(key=lambda kd: kd[0])
    durs = [d for _, d in hits]
    return durs, f"col {header[dc].strip()!r}, {len(durs)} launch row(s)"


def avg_warm(durs, drop):
    """Drop the first `drop` DATA samples (cold-start) and average the rest. If there
    are not more than `drop` rows, average them all (and flag it)."""
    if len(durs) > drop:
        dropped, kept = durs[:drop], durs[drop:]
        return sum(kept) / len(kept), kept, dropped, False
    return (sum(durs) / len(durs) if durs else None), durs, [], True


def _scoreboard(args) -> int:
    scenarios = [f"{op}_{ph}" for ph in args.phases for op in args.ops]
    print(
        f"config: scenarios={scenarios} dtype={args.dtype} "
        f"warmup={args.warmup} iters={args.iters} drop={args.drop}\n"
    )

    match_for = {"ascendc": args.ac_match, "tilelang": args.tl_match}
    exclude_for = {"ascendc": args.ac_exclude, "tilelang": args.tl_exclude}

    results = []  # (scenario, ac_avg, tl_avg, ratio)
    for scn in scenarios:
        print(f"==== {scn} ====")
        avgs = {}
        for impl in ("ascendc", "tilelang"):
            outdir = f"{args.outdir_base}_{impl}_{scn}"
            csv_path = collect(
                impl,
                scn,
                args.dtype,
                args.warmup,
                args.iters,
                outdir,
                args.no_stride_pad,
            )
            if not csv_path:
                avgs[impl] = None
                continue
            durs, info = op_durations(
                csv_path, match_for[impl], exclude_for[impl], args.duration_col
            )
            if not durs:
                print(f"  [{impl:8s}] {info}")
                avgs[impl] = None
                continue
            avg, kept, dropped, fell_back = avg_warm(durs, args.drop)
            rnd = [round(d, 3) for d in durs]
            drp = [round(d, 3) for d in dropped]
            kpt = [round(d, 3) for d in kept]
            tag = " (<=drop rows: averaged ALL, nothing dropped)" if fell_back else ""
            print(
                f"  [{impl:8s}] {info}{tag}\n"
                f"             all data rows = {rnd}\n"
                f"             dropped(cold) = {drp}   averaged = {kpt}\n"
                f"             avg of {len(kept)} = {avg:.3f} us"
            )
            avgs[impl] = avg
        ac, tl = avgs.get("ascendc"), avgs.get("tilelang")
        ratio = (ac / tl) if (ac and tl) else None
        results.append((scn, ac, tl, ratio))
        print()

    print("=" * 78)
    print(
        f"{'scenario':<16}{'ascendc(us)':>14}{'tilelang(us)':>14}{'ac/tl':>10}{'tl reaches':>14}"
    )
    print("-" * 78)
    for scn, ac, tl, ratio in results:
        ac_s = f"{ac:.3f}" if ac else "n/a"
        tl_s = f"{tl:.3f}" if tl else "n/a"
        r_s = f"{ratio:.4f}" if ratio else "n/a"
        pct = f"{ratio * 100:.1f}%" if ratio else "n/a"
        print(f"{scn:<16}{ac_s:>14}{tl_s:>14}{r_s:>10}{pct:>14}")
    print("=" * 78)
    print(
        "ac/tl = ascendc_duration / tilelang_duration "
        "(>1.0 TileLang faster, =1.0 parity, <1.0 = TileLang reaches that % of "
        "Ascend C speed). First sample dropped as cold-start."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="sparse_flash_mla: TileLang vs Ascend C task-duration scoreboard (msprof).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --run-once(内部 launch 靶子)
    ap.add_argument(
        "--run-once",
        action="store_true",
        help="internal: launch target profiled by msprof",
    )
    ap.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        help="(--run-once) single scenario to launch",
    )
    ap.add_argument(
        "--only",
        choices=["ascendc", "tilelang"],
        help="(--run-once) which implementation to launch",
    )
    ap.add_argument(
        "--no-stride-pad",
        action="store_true",
        help="skip the Ascend C PA_ND stride padding",
    )
    # orchestrator
    ap.add_argument("--ops", nargs="+", choices=OPS, default=OPS, help="default: all")
    ap.add_argument(
        "--phases", nargs="+", choices=PHASES, default=PHASES, help="default: both"
    )
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--warmup", type=int, default=0, help="untimed launches (keep 0)")
    ap.add_argument(
        "--iters", type=int, default=6, help="profiled launches (default 6)"
    )
    ap.add_argument("--drop", type=int, default=1, help="leading cold rows to drop")
    ap.add_argument(
        "--ac-match", default="sparse_attn_sharedkv", help="Ascend C op-name filter"
    )
    ap.add_argument(
        "--ac-exclude", default="metadata", help="skip Ascend C ops containing this"
    )
    ap.add_argument(
        "--tl-match", default="sparse_flash_mla", help="TileLang op-name filter"
    )
    ap.add_argument(
        "--tl-exclude", default="", help="skip TileLang ops containing this"
    )
    ap.add_argument(
        "--duration-col",
        default="task duration",
        help="header substring for the duration column (default 'task duration')",
    )
    ap.add_argument(
        "--outdir-base", default="./prof_cmp_sfm", help="msprof scratch dir base"
    )
    args = ap.parse_args()

    if args.run_once:
        if not args.scenario or not args.only:
            print("[fatal] --run-once needs --scenario and --only.", file=sys.stderr)
            return 2
        return run_once(
            args.scenario,
            args.only,
            args.dtype,
            args.warmup,
            args.iters,
            not args.no_stride_pad,
        )

    return _scoreboard(args)


if __name__ == "__main__":
    raise SystemExit(main())
