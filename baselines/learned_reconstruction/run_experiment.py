"""Tune beta and run every SHyRe reconstruction of one dataset.

    python baselines/learned_reconstruction/run_experiment.py --dataset openalex_db --python <SHyRe python>

Steps (after prepare_runs.py):
1. Tune the sampling budget beta on the run <dataset>_tune (seed 0) over the
   grid 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000. The beta with
   the highest validation Jaccard wins (ties go to the smaller beta). A grid
   point slower than 30 minutes ends the grid. The result is stored in
   <work>/beta.json and reused.
2. Run <dataset>_s<s>_ab, _ba and _aG for seeds 0, 1, 2 with that beta.
3. For openalex_db and openalex_cs_size3plus, tune beta on openalex_bio_tune
   the same way and run bio_to_<dataset> for seeds 0, 1, 2.
--beta and --bio-beta skip the tuning (the audited runs found 300 on both
OpenAlex datasets, 3000 on Pokec, and 1000000 on openalex_bio).
Each run writes <work>/out/<run>__b<beta>__s<seed>.pkl; finished runs are skipped.
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_WORK = HERE.parents[1] / "outputs" / "learned_reconstruction" / "work"
GRID = [300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]
TUNE_LIMIT_S = 30 * 60
RUN_LIMIT_S = 4 * 3600


def log(work: Path, msg: str) -> None:
    print(msg, flush=True)
    with open(work / "runs.log", "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def one(py: str, work: Path, run: str, beta: int, seed: int, limit: int):
    out = work / "out" / f"{run}__b{beta}__s{seed}.pkl"
    if out.exists():
        return pickle.load(open(out, "rb"))
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    try:
        p = subprocess.run([py, "-W", "ignore", str(HERE / "run_shyre.py"), "--work", str(work / "runs" / run),
                            "--dataset", run, "--beta", str(beta), "--seed", str(seed), "--out", str(out)],
                           capture_output=True, text=True, timeout=limit)
    except subprocess.TimeoutExpired:
        log(work, f"{run} beta={beta} seed={seed} TIMEOUT {limit}s")
        return None
    if p.returncode != 0 or not out.exists():
        log(work, f"{run} beta={beta} seed={seed} FAILED rc={p.returncode}: {p.stderr[-1500:]}")
        return None
    r = pickle.load(open(out, "rb"))
    log(work, f"{run} beta={beta} seed={seed} J={r['jaccard']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} "
              f"candidates={r['n_candidates']} t={r['t_total_s']:.0f}s rss={r['peak_rss_GB']:.2f}GB "
              f"wall={time.perf_counter() - t0:.0f}s")
    return r


def tuned_beta(py: str, work: Path, run: str, given: int | None = None) -> int:
    sel = work / "beta.json"
    d = json.loads(sel.read_text()) if sel.exists() else {}
    if given is not None:  # recorded so that audit_reconstruction.py finds the runs
        d[run] = dict(beta=int(given), given=True)
        sel.write_text(json.dumps(d, indent=1))
        return int(given)
    if run in d:
        return int(d[run]["beta"])
    best, rows = None, []
    for b in GRID:
        r = one(py, work, run, b, 0, TUNE_LIMIT_S)
        if r is None:
            log(work, f"{run}: grid stops at beta={b}")
            break
        rows.append((b, r["jaccard"]))
        if best is None or r["jaccard"] > best[1]:
            best = (b, r["jaccard"])
    if best is None:
        raise SystemExit(f"tuning of {run} produced no result")
    log(work, f"TUNED {run}: beta={best[0]} J={best[1]:.4f} grid={rows}")
    d[run] = dict(beta=best[0], jaccard=best[1], grid=rows)
    sel.write_text(json.dumps(d, indent=1))
    return int(best[0])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["openalex_db", "openalex_cs_size3plus", "pokec_hobby"])
    p.add_argument("--python", default=sys.executable, help="interpreter of the SHyRe environment")
    p.add_argument("--work", type=Path, default=DEFAULT_WORK)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--beta", type=int, default=None)
    p.add_argument("--bio-beta", type=int, default=None)
    p.add_argument("--no-transfer", action="store_true", help="skip the openalex_bio runs (P4)")
    a = p.parse_args()
    if not (a.work / "runs").exists():
        raise SystemExit(f"{a.work}/runs not found; run prepare_runs.py first")
    ds = a.dataset
    beta = tuned_beta(a.python, a.work, f"{ds}_tune", a.beta)
    for s in a.seeds:
        for k in ["ab", "ba", "aG"]:
            one(a.python, a.work, f"{ds}_s{s}_{k}", beta, s, RUN_LIMIT_S)
    if ds.startswith("openalex") and not a.no_transfer:
        bio_beta = tuned_beta(a.python, a.work, "openalex_bio_tune", a.bio_beta)
        for s in a.seeds:
            one(a.python, a.work, f"bio_to_{ds}", bio_beta, s, RUN_LIMIT_S)
    log(a.work, f"DONE {ds} beta={beta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
