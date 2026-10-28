"""Practicality: batch audits, DuckDB scale-up, and incremental maintenance (each step in a fresh process).

  python run_scaling.py batch [--datasets ...]          certify-then-request vs exact on all 75 cells x 9 budgets
  python run_scaling.py scale-up --replicas 1 2 4 8     imdb_cast_full copies; the paper adds 18 and 36
  python run_scaling.py incremental [--datasets ...]    single-node moves, exact HGP and U_P
  python run_scaling.py smoke                           a one-minute check of all three on OA-DB
Outputs go to outputs/scaling/. The batch step needs outputs/cells.csv from run_cells.py.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from ahc_audit.data import DATASETS
from options import dataset_name

HERE = Path(__file__).resolve().parent
S = HERE / "scaling"


def run(script, *args):
    cmd = [sys.executable, str(S / script), *map(str, args)]
    print(">", " ".join(cmd[1:]), flush=True)
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    print(f"  ({time.perf_counter() - t0:.0f}s)", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("step", choices=["batch", "scale-up", "incremental", "smoke"])
    p.add_argument("--datasets", type=dataset_name, nargs="+", default=DATASETS)
    p.add_argument("--replicas", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--source", default="imdb_cast_full", help="dataset that is copied for the scale-up")
    p.add_argument("--mem", default="4GB")
    p.add_argument("--data-root", default=None)
    p.add_argument("--moves", type=int, default=100000)
    a = p.parse_args()
    dr = ["--data-root", a.data_root] if a.data_root else []
    if a.step == "batch":
        for ds in a.datasets:
            run("batch_audit.py", ds, "--reps", 3, *dr)
        run("batch_summary.py")
    elif a.step == "scale-up":
        run("scale_up.py", f"{a.source}_x1", "ref", *dr)
        for r in a.replicas:
            cfg = f"{a.source}_x{r}"
            phases = ["load", "exact", "release"] + (["fullP"] if r <= 4 else []) + ["scan"]
            for ph in phases:
                run("scale_up.py", cfg, ph, "--mem", a.mem, *dr)
        run("scale_up_summary.py")
    elif a.step == "incremental":
        for ds in a.datasets:
            run("incremental.py", ds, "--moves", a.moves, "--trace-moves", a.moves // 5, "--checkpoint", a.moves // 10, *dr)
    else:
        run("batch_audit.py", "openalex_db", "--reps", 1, "--no-growth", *dr)
        run("batch_summary.py")
        run("scale_up.py", "openalex_db_x1", "ref", *dr)
        for r in (1, 4):
            for ph in ("load", "exact", "release", "fullP", "scan"):
                run("scale_up.py", f"openalex_db_x{r}", ph, "--mem", "1GB", *dr)
        run("scale_up_summary.py")
        run("incremental.py", "openalex_db", "--moves", 2000, "--trace-moves", 500, "--checkpoint", 1000, *dr)


if __name__ == "__main__":
    main()
