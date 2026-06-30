"""TileLang-vs-Ascend C perf scoreboard for sparse_attn_sharedkv (8K prefill + decode).

For each selected scenario it runs msprof TWICE (once with the Ascend C kernel,
once with the TileLang kernel), reads the op's **Task Duration** from the
op_summary CSV, **drops the first (cold-start) sample** and averages the rest,
then prints, per scenario:

    ascendc avg Task Duration | tilelang avg Task Duration | ascendc / tilelang

The last column is how far TileLang has caught up: ``ascendc / tilelang`` (a
duration ratio, so >1.0 = TileLang is FASTER, =1.0 = parity, <1.0 = TileLang
reaches that fraction of Ascend C's speed). It is also printed as a percentage.

Run this ON THE NPU CONTAINER (it shells out to ``msprof`` and the bench runner
``sparse_attn_sharedkv_perf_compare.py``). A full sweep is 6 scenarios x 2 impls
x (warmup+iters) launches under msprof -- the three *prefill* cases are 8K and
take minutes each, so narrow with --ops / --phases when iterating.

WHY "drop the first sample": the first profiled launch carries one-time cost
(cold i-cache/d-cache, kernel load, HBM/TLB cold, DVFS freq ramp) that is not
the kernel's steady-state time -- so the first op_summary row reads too long.
We discard it (configurable via --drop) and average the warm remainder.

Usage
-----
--ops and --phases each take ONE OR MORE values; the scenarios actually run are
the FULL CROSS PRODUCT  (#ops) x (#phases).  Default = all 3 ops x both phases.

  # Everything: swa/cfa/scfa x prefill/decode = 6 scenarios (bf16):
      python msprof_compare.py

  # A SINGLE scenario -- one op + one phase:
      python msprof_compare.py --ops scfa --phases decode
      #   -> scfa_decode

  # One op, BOTH phases:
      python msprof_compare.py --ops scfa
      #   -> scfa_prefill, scfa_decode

  # SEVERAL ops, ONE phase:
      python msprof_compare.py --ops cfa scfa --phases prefill
      #   -> cfa_prefill, scfa_prefill

  # SEVERAL ops x BOTH phases (cross product = 2 x 2 = 4 scenarios):
      python msprof_compare.py --ops swa scfa --phases prefill decode
      #   -> swa_prefill, scfa_prefill, swa_decode, scfa_decode

  # All ops, decode only:
      python msprof_compare.py --phases decode
      #   -> swa_decode, cfa_decode, scfa_decode

  # fp16 / more samples / override the duration column header:
      python msprof_compare.py --ops scfa --dtype float16 --iters 9
      python msprof_compare.py --ops scfa --duration-col "aicore time"

Notes
-----
* --iters N  -> N profiled launches per (scenario, impl); after --drop D leading
  cold samples, the remaining N-D are averaged. Default N=6, D=1 -> avg of 5.
* --warmup is passed to the bench runner as UNtimed launches BEFORE the profiled
  ones; msprof still records them, so we instead rely on --drop. Keep warmup=0
  (default) so the profiled rows == the N iters and --drop is exact.
* Needs both kernels available in the container (the runner's --only ascendc /
  --only tilelang paths). If one side fails to collect, that cell prints "n/a".
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import subprocess

OPS = ["swa", "cfa", "scfa"]
PHASES = ["prefill", "decode"]
RUNNER = "sparse_attn_sharedkv_perf_compare.py"


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
    op_summary CSV path, or None on failure. Mirrors msprof_pipe.collect but with
    configurable dtype/warmup/iters."""
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
    """Drop the first `drop` DATA samples (cold-start) and average the rest.

    `durs` already contains ONLY data-row durations -- the CSV header was excluded
    in op_durations (rows[1:] + float() parse), so durs[0] is the first real
    profiled launch, NOT the header. Returns (avg, kept_list, dropped_list,
    fell_back). If there are not more than `drop` rows, average them all (and flag
    it) rather than dropping everything."""
    if len(durs) > drop:
        dropped, kept = durs[:drop], durs[drop:]
        return sum(kept) / len(kept), kept, dropped, False
    return (sum(durs) / len(durs) if durs else None), durs, [], True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="TileLang vs Ascend C task-duration scoreboard (msprof).",
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
    ap.add_argument("--match", default="sparse_attn_sharedkv", help="op-name filter")
    ap.add_argument("--exclude", default="metadata", help="skip ops containing this")
    ap.add_argument(
        "--duration-col",
        default="task duration",
        help="header substring for the duration column (default 'task duration')",
    )
    ap.add_argument(
        "--outdir-base", default="./prof_cmp", help="msprof scratch dir base"
    )
    args = ap.parse_args()

    scenarios = [f"{op}_{ph}" for ph in args.phases for op in args.ops]
    print(
        f"config: scenarios={scenarios} dtype={args.dtype} "
        f"warmup={args.warmup} iters={args.iters} drop={args.drop}\n"
    )

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
                csv_path, args.match, args.exclude, args.duration_col
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
