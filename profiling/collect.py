"""Collect + summarize D6 MC profiling artifacts into tidy tables.

Run LOCALLY after pulling the personal-folder outputs back from the clusters.
Reads:
  * phases.jsonl        (single-process phase + hotspot records)
  * concurrency.jsonl   (GPU concurrency sweep, one line per process)
  * *.out slurm logs    (GNU /usr/bin/time -v wall + Max RSS; manager internal timer)
Writes CSVs next to --base and prints headline tables.

    python profiling/collect.py --base ./pulled_profiling
"""
import argparse
import glob
import json
import os
import re

import pandas as pd


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def phases_table(records):
    out = []
    for r in records:
        row = {
            "tag": r.get("tag"), "backend": r.get("resource", {}).get("backend"),
            "host": r.get("resource", {}).get("hostname"),
            "ncpu": r.get("resource", {}).get("n_cpu_available"),
            "system": r.get("system"), "L": r.get("L"), "grads": r.get("grads"),
            "warmup": r.get("warmup_steps"), "meas": r.get("meas_steps"),
            "acceptance": r.get("derived", {}).get("acceptance"),
            "ms_per_warmup_step": r.get("derived", {}).get("ms_per_warmup_step"),
            "ms_per_run_step": r.get("derived", {}).get("ms_per_run_step"),
        }
        row.update({f"phase.{k}": v for k, v in r.get("phases_sec", {}).items()})
        # top-3 hotspot functions
        for i, (fn, d) in enumerate(list(r.get("functions", {}).items())[:3]):
            row[f"hot{i+1}"] = f"{fn}={d['secs']}s/{d['calls']}"
        out.append(row)
    return pd.DataFrame(out)


def cpu_sweep_table(phases_df):
    """From the cpusweep_* phase runs, pivot ms/run_step by (backend,L,ncpu) and
    flag the optimal cpu count per (backend,L)."""
    if phases_df.empty or "tag" not in phases_df:
        return phases_df.iloc[0:0], phases_df.iloc[0:0]
    sw = phases_df[phases_df["tag"].astype(str).str.startswith("cpusweep_")].copy()
    if sw.empty:
        return sw, sw
    sw = sw[["backend", "L", "ncpu", "ms_per_run_step"]].sort_values(["backend", "L", "ncpu"])
    # optimal cpu per (backend, L) = argmin ms/run_step
    idx = sw.groupby(["backend", "L"])["ms_per_run_step"].idxmin()
    best = sw.loc[idx].rename(columns={"ncpu": "optimal_ncpu", "ms_per_run_step": "best_ms_per_step"})
    best = best[["backend", "L", "optimal_ncpu", "best_ms_per_step"]].reset_index(drop=True)
    return sw, best


def concurrency_table(records):
    """Aggregate per-process concurrency records into per-(K,mps,L) throughput."""
    rows = []
    for r in records:
        tag = r.get("tag", "")
        m = re.match(r"K(\d+)_(mps|nomps)_L(\d+)_p(\d+)", tag)
        if not m:
            continue
        K, mps, L, p = int(m[1]), m[2], int(m[3]), int(m[4])
        run_step_ms = r.get("derived", {}).get("ms_per_run_step")
        rows.append({"K": K, "mps": mps, "L": L, "proc": p, "ms_per_run_step": run_step_ms,
                     "meas": r.get("meas_steps")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, df
    # per-process latency + aggregate throughput (sum of 1/latency over the K procs)
    df["steps_per_s_proc"] = 1e3 / df["ms_per_run_step"]
    agg = df.groupby(["L", "mps", "K"]).agg(
        n_proc=("proc", "count"),
        mean_ms_per_step=("ms_per_run_step", "mean"),
        max_ms_per_step=("ms_per_run_step", "max"),
        aggregate_steps_per_s=("steps_per_s_proc", "sum"),
    ).reset_index().sort_values(["L", "mps", "K"])
    return df, agg


def parse_slurm_logs(base):
    """Pull wall clock + Max RSS from GNU time and manager's internal timer."""
    rows = []
    for path in glob.glob(os.path.join(base, "**", "*.out"), recursive=True):
        txt = open(path, errors="ignore").read()
        wall = re.search(r"Elapsed \(wall clock\) time.*?:\s*([\d:.]+)", txt)
        rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", txt)
        internal = re.search(r"[Ss]imulation took\s*([\d.]+)", txt)
        job = re.search(r"tag=(\S+)", txt)
        rows.append({
            "log": os.path.basename(path),
            "tag": job[1] if job else None,
            "wall_clock": wall[1] if wall else None,
            "max_rss_kb": int(rss[1]) if rss else None,
            "internal_sim_sec": float(internal[1]) if internal else None,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".", help="dir containing pulled profiling artifacts")
    args = ap.parse_args()

    phases = phases_table(load_jsonl(os.path.join(args.base, "phases.jsonl")))
    cpu_sweep, cpu_best = cpu_sweep_table(phases)
    conc_proc, conc_agg = concurrency_table(load_jsonl(os.path.join(args.base, "concurrency.jsonl")))
    logs = parse_slurm_logs(args.base)

    for name, df in [("phases", phases), ("cpu_sweep", cpu_sweep), ("cpu_optimal", cpu_best),
                     ("concurrency_per_proc", conc_proc),
                     ("concurrency_aggregate", conc_agg), ("slurm_logs", logs)]:
        if isinstance(df, pd.DataFrame) and not df.empty:
            out = os.path.join(args.base, f"summary_{name}.csv")
            df.to_csv(out, index=False)
            print(f"\n===== {name}  ->  {out} =====")
            with pd.option_context("display.max_columns", None, "display.width", 200):
                print(df.to_string(index=False))
        else:
            print(f"\n(no data for {name})")


if __name__ == "__main__":
    main()
