"""Audit the SHyRe reconstructions: HGP of each audited partition on the reconstructed units.

    python baselines/learned_reconstruction/audit_reconstruction.py --budgets outputs/budgets.csv

Reconstructed hypergraphs (sets of distinct node sets) per dataset and seed:
    P1  R(ab) | R(ba)            each half reconstructed by the model trained on the other half
    P2  H_a | R(ab)              training half known exactly, the other half reconstructed
    P3  R(aG)                    model trained on half a, query = projection of all units
    P4  R(bio_to_<ds>)           model trained on openalex_bio, query = projection of all units
For every audited partition (data/partitions/<dataset>/<method>.csv.gz, or a rerun in
outputs/partitions/<method>/<dataset>.csv) and every
audited attribute, HGP is computed on the true units (rows of incidence.npz,
duplicates kept) and on each reconstruction, with the same integer kernel as
the auditor. A cell passes when HGP <= threshold. Verdicts use the mean of
HGP over the three seeds; the per-seed range of the counts is reported too.
Thresholds: the shared budget B of each dataset-attribute pair (--budgets, a CSV with columns dataset,
attribute, B, such as outputs/budgets.csv of run_cells.py) and the fixed tolerance 0.1.

Writes <out>/cells_learned.csv, <out>/verdicts_learned.csv, <out>/recon_quality.csv.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

DATASETS = ["openalex_db", "openalex_cs_size3plus", "pokec_hobby"]
FOLDER = {"OA-DB": "openalex_db", "OA-CS-3+": "openalex_cs_size3plus", "Pokec": "pokec_hobby"}


def find_partition(root: Path, method: str, ds: str) -> Path:
    stored = root / ds / f"{method}.csv.gz"           # data/partitions layout
    return stored if stored.exists() else root / method / f"{ds}.csv"
SEEDS = [0, 1, 2]


# ---- HGP on a list of units (integer kernel) ----
def compact(x):
    _, inv = np.unique(x, return_inverse=True)
    return inv.astype(np.int64), int(inv.max() + 1)


def onehot_int(lab, k):
    n = lab.size
    return sp.csr_matrix((np.ones(n, dtype=np.int64), (np.arange(n), lab)), shape=(n, k))


def unit_counts(H, cl, g):
    """Per unit e: size s_e and Y_e = sum over (group, cluster) cells of max(0, s_e m_egc - m_eg m_ec)."""
    cl, k = compact(cl)
    g, G = compact(g)
    s = np.diff(H.indptr).astype(np.int64)
    Mg = (H @ onehot_int(g, G)).tocsr()
    Mc = (H @ onehot_int(cl, k)).tocsr()
    Mp = (H @ onehot_int(g * k + cl, G * k)).tocoo()
    e, cell, obs = Mp.row, Mp.col, Mp.data.astype(np.int64)
    gg, cc = cell // k, cell % k
    meg = np.asarray(Mg[e, gg]).ravel().astype(np.int64)
    mec = np.asarray(Mc[e, cc]).ravel().astype(np.int64)
    pos = np.maximum(s[e] * obs - meg * mec, 0)
    Y = np.bincount(e, weights=pos.astype(np.float64), minlength=H.shape[0]).round().astype(np.int64)
    return s, Y


def build_H(units, n):
    sizes = np.array([len(u) for u in units], dtype=np.int64)
    indptr = np.concatenate([[0], np.cumsum(sizes)])
    idx = np.concatenate([np.asarray(u, dtype=np.int64) for u in units])
    H = sp.csr_matrix((np.ones(idx.size, dtype=np.int64), idx, indptr), shape=(len(units), n))
    H.sort_indices()
    return H


def hgp(units, n, cl, g) -> float:
    s, Y = unit_counts(build_H(list(units), n), cl, g)
    return float((Y / s).sum() / s.sum())


# ---- inputs ----
def read_partition(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    return df.sort_values(df.columns[0])["cluster"].to_numpy(dtype=np.int64)


def load_out(work: Path, run: str, beta: int, seed: int):
    p = work / "out" / f"{run}__b{beta}__s{seed}.pkl"
    if not p.exists():
        return None
    r = pickle.load(open(p, "rb"))
    ids = np.load(work / "runs" / run / "maps.npz")["test_ids"]
    to_ds = lambda S: set(tuple(sorted(int(ids[v]) for v in c)) for c in S)
    return dict(R=to_ds(r["reconstruction"]), meta=r)


def quality(R, T):
    hit = len(R & T)
    P = hit / len(R) if R else 0.0
    Rc = hit / len(T)
    return dict(precision=P, recall=Rc, f1=2 * P * Rc / (P + Rc) if P * Rc > 0 else 0.0,
                jaccard=hit / len(R | T), n_reconstructed=len(R), n_true=len(T))


def counts(true, aud, thr):
    is_fail = true > thr
    aud_pass = aud <= thr
    return dict(fail=int(is_fail.sum()), misses=int((is_fail & aud_pass).sum()),
                passes=int((~is_fail).sum()), false_alarms=int((~is_fail & ~aud_pass).sum()))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", type=Path, default=common.REPO_ROOT / "outputs" / "learned_reconstruction" / "work")
    p.add_argument("--data-root", type=Path, default=common.REPO_ROOT / "data")
    p.add_argument("--partitions", type=Path, default=common.REPO_ROOT / "data" / "partitions",
                   help="data/partitions layout (<dataset>/<method>.csv.gz) or a rerun folder (<method>/<dataset>.csv)")
    p.add_argument("--budgets", type=Path, default=None, help="CSV with columns dataset, attribute, B")
    p.add_argument("--tolerance", type=float, default=0.1)
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument("--beta", type=int, default=None, help="override the tuned beta of every dataset")
    p.add_argument("--bio-beta", type=int, default=None)
    p.add_argument("--out", type=Path, default=common.REPO_ROOT / "outputs" / "learned_reconstruction")
    a = p.parse_args()
    betas = json.loads((a.work / "beta.json").read_text()) if (a.work / "beta.json").exists() else {}
    budgets = None
    if a.budgets is not None:
        b = pd.read_csv(a.budgets)
        budgets = {(FOLDER.get(r.dataset, r.dataset), r.attribute): float(r.B) for r in b.itertuples()}

    cell_rows, qual_rows = [], []
    for ds in a.datasets:
        H = common.load_incidence(a.data_root / ds)
        m, n = H.shape
        true_units = [tuple(sorted(H.indices[H.indptr[r]:H.indptr[r + 1]].tolist())) for r in range(m)]
        T = set(true_units)
        sens = common.load_table(a.data_root / ds, "sensitive_attrs.csv", n)
        beta = a.beta or betas[f"{ds}_tune"]["beta"]
        bio_beta = a.bio_beta or betas.get("openalex_bio_tune", {}).get("beta")
        recs = {}
        for s in SEEDS:
            split = np.load(a.work / f"splits_{ds}_s{s}.npz")
            Ha = set(true_units[i] for i in split["a"])
            ab = load_out(a.work, f"{ds}_s{s}_ab", beta, s)
            ba = load_out(a.work, f"{ds}_s{s}_ba", beta, s)
            aG = load_out(a.work, f"{ds}_s{s}_aG", beta, s)
            if ab and ba:
                recs[("P1", s)] = (ab["R"] | ba["R"], beta)
            if ab:
                recs[("P2", s)] = (Ha | ab["R"], beta)
            if aG:
                recs[("P3", s)] = (aG["R"], beta)
            if ds.startswith("openalex") and bio_beta is not None:
                tr = load_out(a.work, f"bio_to_{ds}", bio_beta, s)
                if tr:
                    recs[("P4", s)] = (tr["R"], bio_beta)
        for (reg, s), (R, bb) in sorted(recs.items()):
            qual_rows.append(dict(dataset=ds, regime=reg, seed=s, beta=bb, **quality(R, T)))
        for method in common.METHODS:
            path = find_partition(a.partitions, method, ds)
            if not path.exists():
                print(f"[audit] missing partition {path}, skipped", flush=True)
                continue
            cl = read_partition(path)
            for attr in common.SENSITIVE_COLUMNS[ds]:
                g = common.factorize(sens[attr])
                h_true = hgp(true_units, n, cl, g)
                for (reg, s), (R, bb) in sorted(recs.items()):
                    h = hgp(R, n, cl, g)
                    cell_rows.append(dict(method=method, dataset=ds, attribute=attr, regime=reg, seed=s, beta=bb,
                                          HGP_true=h_true, HGP_learned=h, error=h - h_true,
                                          B=(budgets or {}).get((ds, attr), np.nan)))
        print(f"[audit] {ds}: {len(recs)} reconstructions", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    cells = pd.DataFrame(cell_rows)
    cells.to_csv(a.out / "cells_learned.csv", index=False)
    pd.DataFrame(qual_rows).to_csv(a.out / "recon_quality.csv", index=False)

    key = ["method", "dataset", "attribute"]
    rows = []
    for (ds, reg), sub in cells.groupby(["dataset", "regime"], sort=False):
        mean = sub.groupby(key, sort=False).agg(HGP_true=("HGP_true", "first"),
                                                HGP_learned=("HGP_learned", "mean"), B=("B", "first"))
        for thr_name in ["B", "tolerance"]:
            if thr_name == "B" and mean.B.isna().any():
                continue
            thr = mean.B.to_numpy() if thr_name == "B" else a.tolerance
            c = counts(mean.HGP_true.to_numpy(), mean.HGP_learned.to_numpy(), thr)
            per_seed = []
            for s, ss in sub.groupby("seed"):
                ss = ss.set_index(key).loc[mean.index]
                t = ss.B.to_numpy() if thr_name == "B" else a.tolerance
                per_seed.append(counts(ss.HGP_true.to_numpy(), ss.HGP_learned.to_numpy(), t))
            rows.append(dict(dataset=ds, regime=reg, threshold="B" if thr_name == "B" else f"{a.tolerance:g}",
                             cells=len(mean), **c,
                             misses_seed_range=f"{min(x['misses'] for x in per_seed)}-"
                                                     f"{max(x['misses'] for x in per_seed)}",
                             false_alarms_seed_range=f"{min(x['false_alarms'] for x in per_seed)}-"
                                                     f"{max(x['false_alarms'] for x in per_seed)}",
                             mean_abs_error=float((mean.HGP_learned - mean.HGP_true).abs().mean())))
    verdicts = pd.DataFrame(rows)
    verdicts.to_csv(a.out / "verdicts_learned.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(verdicts.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
