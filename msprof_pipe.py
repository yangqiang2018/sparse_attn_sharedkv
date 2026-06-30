"""采集并提取 sparse_attn_sharedkv kernel 的逐-pipe msprof 数据(cube/vector 时间 +
mac/scalar/mte/fixpipe/vec 各占比),用于 TileLang vs Ascend C 的瓶颈定位。

它回答的核心问题:kernel 是算力受限、访存受限、还是 overlap/同步受限,卡在哪个 pipe ——
这样下一步优化能对症,而不是靠猜。

两种用法(都在 NPU 容器里跑):

  A) 一键采集 + 解析(自动帮你跑 msprof):
       python msprof_pipe.py --impl tilelang  --scenario swa_prefill
       python msprof_pipe.py --impl ascendc   --scenario swa_prefill

  B) 解析你自己已经产出的 CSV(若你那边 msprof 命令行不同,这种最稳):
       # 先用你惯常的方式跑出带 PipeUtilization 的 op_summary CSV,然后:
       python msprof_pipe.py --csv /path/to/op_summary_*.csv

参数:
  --impl     tilelang | ascendc    选哪个实现(缺省 tilelang)
  --scenario 场景名,可选 swa_prefill / cfa_prefill / scfa_prefill /
             swa_decode / cfa_decode / scfa_decode(缺省 swa_prefill)
  --csv      直接解析已有的 op_summary CSV(给了它就跳过采集)
  --match    op 名过滤串(缺省 "sparse_attn_sharedkv")
  --exclude  跳过名字含此串的 op(缺省 "metadata";传 "" 关闭)
  --outdir   msprof 输出目录

举例:
  # 采集 + 解析 TileLang 的 scfa_prefill:
      python msprof_pipe.py --impl tilelang --scenario scfa_prefill
  # 采集 + 解析 Ascend C 的 scfa_decode:
      python msprof_pipe.py --impl ascendc --scenario scfa_decode
  # 只解析一个已有的 CSV:
      python msprof_pipe.py --csv ./prof_tilelang_scfa_prefill/PROF_xxx/.../op_summary_x.csv

它会打印:每个名字匹配 --match 的 op 行 —— op 名 + 所有表头含 time/ratio/cycles 的列;
并列出找到的全部不同 op 名,万一过滤没命中,你能看到真实 op 名再用 --match 重跑。

关键的人工判读(对着打印的数字眼看):
  * device kernel time ≈ max(aicore_time, aiv_time)  → cube/vector overlap 好
  * device kernel time ≈ aicore_time + aiv_time      → cube/vector 串行(没 overlap)
  * aic_scalar_ratio 对比老的 0.63                    → scalar 负载有没有降下来

注:本工具只看逐-pipe 占比/瓶颈;要直接对比 ascendc vs tilelang 的 Task Duration 平均值
(丢冷启动、算追平百分比),用同目录的 msprof_compare.py。
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys

# Columns we care about: any header containing one of these (case-insensitive).
_WANT = ("time", "ratio", "cycle", "mac", "scalar", "mte", "fixpipe", "vec", "dur")
# How to find the op-name column.
_NAME_HINTS = ("op name", "op_name", "opname", "op type", "name")


def _norm(s: str) -> str:
    """Lowercase and drop non-alphanumerics, so 'sparse_attn_sharedkv' and the
    Ascend C 'SparseAttnSharedkv' compare equal."""
    return "".join(c for c in s.lower() if c.isalnum())


def _find_name_col(header: list[str]) -> int:
    low = [h.strip().lower() for h in header]
    for hint in _NAME_HINTS:
        for i, h in enumerate(low):
            if hint in h:
                return i
    return 0


def _want_cols(header: list[str]) -> list[int]:
    out = []
    for i, h in enumerate(header):
        hl = h.strip().lower()
        if any(w in hl for w in _WANT):
            out.append(i)
    return out


def parse_csv(path: str, match: str, exclude: str = "metadata") -> None:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        print(f"  (empty CSV: {path})")
        return
    header = rows[0]
    name_col = _find_name_col(header)
    cols = _want_cols(header)
    print(f"=== {os.path.basename(path)} ===")
    print(f"op-name column: [{name_col}] {header[name_col]!r}")

    nmatch = _norm(match)
    nexcl = _norm(exclude) if exclude else ""
    names = []
    hits = []
    for r in rows[1:]:
        if name_col >= len(r):
            continue
        nm = r[name_col].strip()
        names.append(nm)
        nn = _norm(nm)
        # Match on normalized name (underscore/case-insensitive); skip the
        # separate metadata op which also contains "SharedkvMetadata".
        if nmatch in nn and (not nexcl or nexcl not in nn):
            hits.append(r)

    print(f"distinct op names ({len(set(names))}): {sorted(set(names))[:40]}")
    if not hits:
        print(f"  !! no row matched --match={match!r}; pick the real name above.")
        return
    print(f"\nmatched rows for {match!r} ({len(hits)}):")
    for r in hits:
        nm = r[name_col].strip() if name_col < len(r) else "?"
        print(f"\n  op: {nm}")
        for c in cols:
            if c < len(r) and r[c].strip() != "":
                print(f"    {header[c].strip():28s} = {r[c].strip()}")


def collect(impl: str, scenario: str, outdir: str) -> str | None:
    """Run msprof with PipeUtilization, return the op_summary CSV path (or None).

    msprof drops a fresh ``PROF_*`` subdir per run and never cleans old ones, so
    we wipe ``outdir`` first and then pick the CSV by newest mtime -- never a
    stale result from a previous run.
    """
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)
    app = (
        f"python sparse_attn_sharedkv_perf_compare.py "
        f"--scenarios {scenario} --only {impl} --warmup 2 --iters 5"
    )
    cmd = [
        "msprof",
        f"--output={outdir}",
        f"--application={app}",
        "--ai-core=on",
        "--aic-metrics=PipeUtilization",
    ]
    print("running:", " ".join(cmd))
    print("(if this errors, run msprof your own way with PipeUtilization, then")
    print(" re-run:  python msprof_pipe.py --csv <op_summary csv>)")
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\nmsprof failed: {e}")
        return None
    # Newer msprof COLLECTS into a PROF_* dir but does NOT auto-export the
    # op_summary CSV (it just prints "Data is saved in .../PROF_*"). So run the
    # explicit export pass on each fresh PROF_* dir to generate the CSVs, then
    # glob. (outdir was wiped above, so every PROF_* here is from THIS run.)
    if not glob.glob(os.path.join(outdir, "**", "*op_summary*.csv"), recursive=True):
        for prof_dir in glob.glob(os.path.join(outdir, "PROF_*")):
            print(f"exporting: msprof --export=on --output={prof_dir}")
            try:
                subprocess.run(
                    ["msprof", "--export=on", f"--output={prof_dir}"], check=True
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"export failed for {prof_dir}: {e}")
    # Take the newest op_summary CSV by mtime (this run's).
    cands = glob.glob(os.path.join(outdir, "**", "*op_summary*.csv"), recursive=True)
    cands.sort(key=os.path.getmtime)
    if not cands:
        print(f"\nno op_summary CSV under {outdir} even after --export=on. Try:")
        print(f"  msprof --export=on --output={outdir}/PROF_*")
        print(f"then: python msprof_pipe.py --csv {outdir}/PROF_*/.../op_summary_*.csv")
        return None
    return cands[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["tilelang", "ascendc"], default="tilelang")
    ap.add_argument("--scenario", default="swa_prefill")
    ap.add_argument("--csv", default=None, help="parse an existing op_summary CSV")
    ap.add_argument("--match", default="sparse_attn_sharedkv", help="op-name filter")
    ap.add_argument(
        "--exclude",
        default="metadata",
        help="skip op names containing this (normalized); '' to disable",
    )
    ap.add_argument("--outdir", default=None, help="msprof output dir")
    args = ap.parse_args()

    if args.csv:
        parse_csv(args.csv, args.match, args.exclude)
        return

    outdir = args.outdir or f"./prof_{args.impl}_{args.scenario}"
    os.makedirs(outdir, exist_ok=True)
    csv_path = collect(args.impl, args.scenario, outdir)
    if csv_path:
        print()
        parse_csv(csv_path, args.match, args.exclude)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
