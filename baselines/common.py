"""Shared settings and input/output helpers for the baseline run scripts.

Every path is relative to the repository root, which is the parent of this folder.
A dataset folder holds
    incidence.npz        scipy sparse matrix, units x nodes
    node_features.npz    scipy sparse matrix, nodes x features
    sensitive_attrs.csv  node_id plus one column per audited attribute
    labels.csv           node_id plus the class label column `label`
A partition file is a CSV with the columns node_id,cluster.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

BASELINES_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASELINES_DIR.parent
THIRD_PARTY = BASELINES_DIR / "third_party"

DATASETS = (
    "movielens_1m",
    "openalex_db",
    "openalex_cs_size3plus",
    "pokec_hobby",
    "imdb_cast",
    "acs_pums_household",
)

# Audited attributes of each dataset, as columns of sensitive_attrs.csv.
# Only CE+Fair-Louvain reads them: it penalizes every column listed here
# (on ACS-PUMS-HH this includes race and disability, which the paper audits
# but screens out of the 75 cells). The other methods never see them.
SENSITIVE_COLUMNS = {
    "movielens_1m": ["gender", "age_bucket"],
    "openalex_db": ["country", "institution_type", "career_stage"],
    "openalex_cs_size3plus": ["country", "institution_type", "career_stage"],
    "pokec_hobby": ["gender", "AGE", "region_kraj"],
    "imdb_cast": ["birth_cohort", "career_stage"],
    "acs_pums_household": ["sex", "age_bucket", "race", "disability"],
}

# Number of clusters K of each audited partition (Table 3 of the paper).
# AHCKA, GRAC and CAHC take K as input. AHRC and CE+Fair-Louvain return
# their natural number of clusters, and K is only checked after the run.
K_TABLE = {
    "ahrc": {"movielens_1m": 11, "openalex_db": 8, "openalex_cs_size3plus": 34,
             "pokec_hobby": 34, "imdb_cast": 485, "acs_pums_household": 26},
    "ahcka": {"movielens_1m": 18, "openalex_db": 11, "openalex_cs_size3plus": 26,
              "pokec_hobby": 6, "imdb_cast": 23, "acs_pums_household": 9},
    "grac": {"movielens_1m": 18, "openalex_db": 11, "openalex_cs_size3plus": 26,
             "pokec_hobby": 6, "imdb_cast": 23, "acs_pums_household": 9},
    "cahc": {"movielens_1m": 18, "openalex_db": 11, "openalex_cs_size3plus": 26,
             "pokec_hobby": 34, "imdb_cast": 23, "acs_pums_household": 9},
    "ce_fair_louvain": {"movielens_1m": 1118, "openalex_db": 135,
                        "openalex_cs_size3plus": 2478, "pokec_hobby": 5733,
                        "imdb_cast": 6658, "acs_pums_household": 22706},
}

# Seed of each audited partition.
SEEDS = {
    "ahrc": {"movielens_1m": 42, "openalex_db": 2026, "openalex_cs_size3plus": 42,
             "pokec_hobby": 42, "imdb_cast": 42, "acs_pums_household": 2026},
    "ahcka": {d: 2026 for d in DATASETS},
    "grac": {d: 2026 for d in DATASETS},
    "cahc": {d: 42 for d in DATASETS},
    "ce_fair_louvain": {d: 42 for d in DATASETS},
}

METHODS = tuple(K_TABLE)


def default_out(method: str, dataset: str) -> Path:
    return REPO_ROOT / "outputs" / "partitions" / method / f"{dataset}.csv"


def add_common_args(parser: argparse.ArgumentParser, method: str) -> None:
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="dataset folder (default: data/<dataset> in the repository)")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed (default: the seed of the audited partition)")
    parser.add_argument("--k", type=int, default=None,
                        help="number of clusters (default: K of Table 3)")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"output CSV (default: outputs/partitions/{method}/<dataset>.csv)")


def resolve_common_args(args: argparse.Namespace, method: str) -> argparse.Namespace:
    if args.data_dir is None:
        args.data_dir = REPO_ROOT / "data" / args.dataset
    if args.seed is None:
        args.seed = SEEDS[method][args.dataset]
    if args.k is None:
        args.k = K_TABLE[method][args.dataset]
    if args.out is None:
        args.out = default_out(method, args.dataset)
    return args


def load_incidence(data_dir: Path) -> sp.csr_matrix:
    """Incidence matrix, units x nodes, in CSR format."""
    return sp.load_npz(Path(data_dir) / "incidence.npz").tocsr()


def load_features(data_dir: Path) -> sp.csr_matrix:
    return sp.load_npz(Path(data_dir) / "node_features.npz").tocsr()


def load_table(data_dir: Path, name: str, n: int) -> pd.DataFrame:
    """Read a per-node CSV and check that row i is node i."""
    df = pd.read_csv(Path(data_dir) / name)
    if len(df) != n:
        raise SystemExit(f"{name} has {len(df)} rows, expected {n}")
    if "node_id" in df.columns and not np.array_equal(df["node_id"].to_numpy(), np.arange(n)):
        raise SystemExit(f"{name} is not sorted by node_id 0..n-1")
    return df


def factorize(series: pd.Series) -> np.ndarray:
    """Integer codes in order of first appearance; missing values form one group."""
    values = series.astype(object).where(series.notna(), "__missing__")
    codes, _ = pd.factorize(values)
    return np.asarray(codes, dtype=np.int64)


def units_from_incidence(H: sp.csr_matrix) -> list[list[int]]:
    """Member lists of the units, in row order of the incidence matrix."""
    return [H.indices[H.indptr[r]:H.indptr[r + 1]].tolist() for r in range(H.shape[0])]


def write_partition(path: Path, clusters: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clusters = np.asarray(clusters, dtype=np.int64).ravel()
    pd.DataFrame({"node_id": np.arange(clusters.size), "cluster": clusters}).to_csv(path, index=False)


def report(method: str, dataset: str, clusters: np.ndarray, k_expected: int, path: Path) -> None:
    k = int(np.unique(clusters).size)
    note = "" if k == k_expected else f"  (Table 3 lists K = {k_expected})"
    print(f"[{method}] {dataset}: {clusters.size} nodes, {k} clusters{note} -> {path}", flush=True)
