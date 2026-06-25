import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

METRICS = [
    "psnr",
    "rmse_celsius",
    "mae_celsius",
    "max_error_celsius",
    "smape_percent",
    "compression_factor",
    "compression_time_seconds",
    "decompression_time_seconds",
    "best_iter",
    "wall_seconds",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gan", choices=["gan", "no_gan"])
    ap.add_argument("--results_root", default=None)
    ap.add_argument("--out_summary", default=None)
    ap.add_argument("--out_runs", default=None)
    args = ap.parse_args()

    if args.results_root is None:
        args.results_root = os.path.join(HERE, f"outputs_{args.tag}")
    if args.out_summary is None:
        args.out_summary = os.path.join(HERE, f"summary_{args.tag}.csv")
    if args.out_runs is None:
        args.out_runs = os.path.join(HERE, f"runs_{args.tag}.csv")

    runs = []
    for path in sorted(glob.glob(os.path.join(args.results_root, "*", "p*", "seed*", "result.json"))):
        with open(path) as f:
            runs.append(json.load(f))

    if not runs:
        raise SystemExit(f"brak wynikow w {args.results_root}")

    with open(args.out_runs, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0].keys()))
        w.writeheader()
        w.writerows(runs)

    groups = defaultdict(list)
    for r in runs:
        groups[(r["network"], r["config_idx"])].append(r)

    rows = []
    for (net, ci), grp in sorted(groups.items()):
        n = len(grp)
        sqrt_n = math.sqrt(n)
        row = {
            "network": net,
            "config_idx": ci,
            "source_trial": grp[0]["source_trial"],
            "n_seeds": n,
            "generator_params": grp[0]["generator_params"],
        }
        for k in METRICS:
            vals = np.array([g[k] for g in grp], dtype=float)
            std = float(np.nanstd(vals, ddof=1)) if n > 1 else 0.0
            row[f"{k}_mean"] = float(np.nanmean(vals))
            row[f"{k}_std"] = std
            row[f"{k}_se"] = std / sqrt_n
            row[f"{k}_min"] = float(np.nanmin(vals))
            row[f"{k}_max"] = float(np.nanmax(vals))
        rows.append(row)

    with open(args.out_summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{len(runs)} przebiegow, {len(rows)} konfiguracji")
    for r in rows:
        print(f"  {r['network']} p{r['config_idx']}: "
              f"PSNR={r['psnr_mean']:.3f}+-{r['psnr_std']:.3f} (SE={r['psnr_se']:.3f}) "
              f"CR={r['compression_factor_mean']:.2f}")
    print(f"-> {args.out_summary}")
    print(f"-> {args.out_runs}")


if __name__ == "__main__":
    main()
