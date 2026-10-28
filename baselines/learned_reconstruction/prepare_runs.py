"""Write the SHyRe training and query files for every run of the learned reconstruction.

    python baselines/learned_reconstruction/prepare_runs.py

Units are the rows of incidence.npz (duplicates kept). For seed s in {0, 1, 2}
the rows are permuted with numpy default_rng(s) and split into halves a (first
floor(m/2)) and b. Runs per dataset and seed:
    <ds>_s<s>_ab   train on half a, query the projection of half b   (P1, P2)
    <ds>_s<s>_ba   train on half b, query the projection of half a   (P1)
    <ds>_s<s>_aG   train on half a, query the projection of all units (P3)
    bio_to_<ds>    train on all units of openalex_bio, query all units (P4, OpenAlex only)
    <ds>_tune      90 / 10 split of half a of seed 0 (default_rng(1000)) to tune beta
    openalex_bio_tune  90 / 10 split of openalex_bio to tune beta for P4
Each run folder holds data/<run>/{train,test}.txt with compact node ids and
maps.npz (train_ids, test_ids: dataset node id of each compact id).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

DATASETS = ["openalex_db", "openalex_cs_size3plus", "pokec_hobby"]
SEEDS = [0, 1, 2]
DEFAULT_WORK = common.REPO_ROOT / "outputs" / "learned_reconstruction" / "work"


def units(data_root: Path, ds: str) -> list[np.ndarray]:
    H = sp.load_npz(data_root / ds / "incidence.npz").tocsr()
    return [np.sort(H.indices[H.indptr[r]:H.indptr[r + 1]]).astype(np.int64) for r in range(H.shape[0])]


def write(path: Path, us: list[np.ndarray]) -> np.ndarray:
    ids = np.unique(np.concatenate(us))
    loc = {int(v): i for i, v in enumerate(ids)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for u in us:
            f.write(" ".join(str(loc[int(v)]) for v in u) + "\n")
    return ids


def make_run(work: Path, name: str, train: list, test: list) -> None:
    d = work / "runs" / name
    a = write(d / "data" / name / "train.txt", train)
    b = write(d / "data" / name / "test.txt", test)
    np.savez(d / "maps.npz", train_ids=a, test_ids=b)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=common.REPO_ROOT / "data")
    p.add_argument("--work", type=Path, default=DEFAULT_WORK)
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument("--no-transfer", action="store_true", help="skip the openalex_bio runs (P4)")
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    bio = None if a.no_transfer else units(a.data_root, "openalex_bio")
    for ds in a.datasets:
        us = units(a.data_root, ds)
        m = len(us)
        for s in SEEDS:
            perm = np.random.default_rng(s).permutation(m)
            ia, ib = np.sort(perm[:m // 2]), np.sort(perm[m // 2:])
            np.savez(a.work / f"splits_{ds}_s{s}.npz", a=ia, b=ib)
            A, B = [us[i] for i in ia], [us[i] for i in ib]
            make_run(a.work, f"{ds}_s{s}_ab", A, B)
            make_run(a.work, f"{ds}_s{s}_ba", B, A)
            make_run(a.work, f"{ds}_s{s}_aG", A, us)
            if s == 0:
                q = np.random.default_rng(1000).permutation(len(A))
                k = int(round(0.9 * len(A)))
                make_run(a.work, f"{ds}_tune", [A[i] for i in np.sort(q[:k])], [A[i] for i in np.sort(q[k:])])
        if bio is not None and ds.startswith("openalex"):
            make_run(a.work, f"bio_to_{ds}", bio, us)
    if bio is not None:
        q = np.random.default_rng(1000).permutation(len(bio))
        k = int(round(0.9 * len(bio)))
        make_run(a.work, "openalex_bio_tune", [bio[i] for i in np.sort(q[:k])], [bio[i] for i in np.sort(q[k:])])
    print("runs:", " ".join(sorted(x.name for x in (a.work / "runs").iterdir())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
