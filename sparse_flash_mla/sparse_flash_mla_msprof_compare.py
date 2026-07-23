"""sparse_flash_mla(上游 example)的 TileLang vs Ascend C 性能计分板(msprof 口径)。

与根目录的 ``msprof_compare.py`` 同款两层结构:对每个场景各跑两次 msprof(一次 Ascend C
kernel、一次 TileLang kernel),从 op_summary CSV 读该 op 的 **Task Duration**,丢掉第 1 条
(冷启动)样本、其余取平均,再打印:

    ascendc 平均 Task Duration | tilelang 平均 Task Duration | ascendc / tilelang

最后一列 ``ascendc / tilelang`` 衡量 TileLang 追平程度(>1.0 = TileLang 更快、=1.0 = 持平、
<1.0 = TileLang 达到 Ascend C 速度的该比例),同时以百分比打印。测的是 msprof 抓到的
**device 侧 kernel 时长**,不含 host 下发/同步开销 —— 这是和历史台账严格同口径的量。

与 msprof_compare.py 的唯一区别:被 profile 的 runner 是上游版 ``sparse_flash_mla_perf.py``,
而上游版把 TileLang kernel 改名成 ``sparse_flash_mla``(Ascend C 侧仍是
``npu_sparse_attn_sharedkv``)。两侧 op 名不同,所以过滤串分开给 —— Ascend C 用 ``--ac-match``
(默认 ``sparse_attn_sharedkv``、排除 ``metadata``),TileLang 用 ``--tl-match``(默认
``sparse_flash_mla``)。

在 NPU 容器里、cd 到本目录后跑(它会调用 ``msprof`` 和同目录的 ``sparse_flash_mla_perf.py``)。
容器专用文件,不合入官方 example。

用法
----
--ops 和 --phases 都可给一个或多个值;实际跑的场景 = 两者的笛卡尔积。缺省 = 全 3 op × 2 phase。

  # 全量:swa/hca/csa × prefill/decode = 6 个场景(bf16):
      python sparse_flash_mla_msprof_compare.py

  # 只 prefill 三场景:
      python sparse_flash_mla_msprof_compare.py --phases prefill

  # 单个场景:
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

OPS = ["swa", "hca", "csa"]
PHASES = ["prefill", "decode"]
RUNNER = "sparse_flash_mla_perf.py"


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


def collect(impl: str, scenario: str, dtype: str, warmup: int, iters: int, outdir: str):
    """Run msprof over the bench runner for one (impl, scenario); return the newest
    op_summary CSV path, or None on failure."""
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)
    app = (
        f"python {RUNNER} --scenarios {scenario} --only {impl} "
        f"--dtype {dtype} --warmup {warmup} --iters {iters}"
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="sparse_flash_mla: TileLang vs Ascend C task-duration scoreboard (msprof).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
            csv_path = collect(impl, scn, args.dtype, args.warmup, args.iters, outdir)
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

    # ---- scoreboard ----
    print("=" * 78)
    print(
        f"{'scenario':<16}{'ascendc(us)':>14}{'tilelang(us)':>14}"
        f"{'ac/tl':>10}{'tl reaches':>14}"
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


if __name__ == "__main__":
    main()
