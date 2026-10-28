"""Rerun the clusterings of the 57 seed cells with the four further seeds 1, 2, 3, 4.

The 57 cells are the 23 method-dataset pairs below (2 or 3 audited attributes
each). Every run is the command of the audited partition with only --seed and
--out changed. The audited partition stands for the stored seed; the other
pairs (AHRC on Pokec and IMDb-Cast, GRAC on Pokec, CAHC on OA-CS-3+, Pokec,
IMDb-Cast and ACS-PUMS-HH) were not rerun.

    python baselines/run_seeds.py --dry-run          # list the 92 commands
    python baselines/run_seeds.py --jobs 2           # run them
    python baselines/run_seeds.py --methods ahrc --datasets openalex_db

Output: outputs/partitions_seeds/<method>/<dataset>/seed<k>.csv
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import common

SCRIPT = {
    "ahrc": "run_ahrc.py",
    "ahcka": "run_ahcka.py",
    "grac": "run_grac.py",
    "cahc": "run_cahc.py",
    "ce_fair_louvain": "run_ce_fair_louvain.py",
}

PAIRS = {
    "ahrc": ["movielens_1m", "openalex_db", "openalex_cs_size3plus", "acs_pums_household"],
    "ahcka": list(common.DATASETS),
    "grac": ["movielens_1m", "openalex_db", "openalex_cs_size3plus", "imdb_cast", "acs_pums_household"],
    "cahc": ["movielens_1m", "openalex_db"],
    "ce_fair_louvain": list(common.DATASETS),
}


def commands(methods, datasets, seeds, out_root, data_root):
    out = []
    for m in methods:
        for d in PAIRS[m]:
            if datasets and d not in datasets:
                continue
            for s in seeds:
                dest = out_root / m / d / f"seed{s}.csv"
                argv = [sys.executable, str(common.BASELINES_DIR / SCRIPT[m]), "--dataset", d,
                        "--seed", str(s), "--out", str(dest)]
                if data_root is not None:
                    argv += ["--data-dir", str(data_root / d)]
                out.append((m, d, s, argv, dest))
    return out


def run(job) -> None:
    m, d, s, argv, dest = job
    t0 = time.time()
    rc = subprocess.run(argv).returncode
    print(f"[seeds] {m}/{d}/seed{s} rc={rc} {time.time() - t0:.0f} s", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="+", default=list(PAIRS), choices=list(PAIRS))
    p.add_argument("--datasets", nargs="+", default=None, choices=common.DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--jobs", type=int, default=1,
                   help="parallel runs for AHRC, AHCKA, GRAC and CE+Fair-Louvain; CAHC always runs alone")
    p.add_argument("--out-root", type=Path, default=common.REPO_ROOT / "outputs" / "partitions_seeds")
    p.add_argument("--data-root", type=Path, default=None, help="folder of the dataset folders (default: data)")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    jobs = commands(a.methods, a.datasets, a.seeds, a.out_root, a.data_root)
    if a.skip_existing:
        jobs = [j for j in jobs if not j[4].exists()]
    if a.dry_run:
        rel = lambda x: os.path.relpath(x, common.REPO_ROOT) if os.path.isabs(x) else x
        for m, d, s, argv, _ in jobs:
            print("python " + " ".join(rel(x) for x in argv[1:]))
        print(f"{len(jobs)} runs")
        return 0
    light = [j for j in jobs if j[0] != "cahc"]
    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        list(ex.map(run, light))
    for j in jobs:
        if j[0] == "cahc":
            run(j)
    return 0


if __name__ == "__main__":
    sys.exit(main())
