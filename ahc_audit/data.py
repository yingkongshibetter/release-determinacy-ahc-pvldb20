"""Datasets, audited attributes, class labels, and the 30 audited partitions.

A dataset folder `data/<name>/` holds
    incidence.npz        scipy.sparse matrix, units x nodes (one row per unit)
    sensitive_attrs.csv  node_id plus one column per audited attribute
    labels.csv           node_id plus the class label `label` (the budget strata)
A partition file `data/partitions/<name>/<method>.csv.gz` has the columns node_id,cluster.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parent.parent

# Audit settings (configs/audit.json). Order matters: the shared budget replays one random stream over the
# datasets and attributes in exactly this order (see budgets.py).
CONFIG = json.loads((REPO_ROOT / "configs" / "audit.json").read_text(encoding="utf-8"))
DATASETS = list(CONFIG["datasets"])
DISPLAY = {k: v["paper_name"] for k, v in CONFIG["datasets"].items()}
INTERNAL = {v: k for k, v in DISPLAY.items()}
ATTRIBUTES = {k: v["attributes"] for k, v in CONFIG["datasets"].items()}
# Race and disability of ACS-PUMS-HH have less than 30% of their memberships in units with two or more
# groups and are screened out of the 75 cells.
SCREENED = {(k, a) for k, v in CONFIG["datasets"].items() for a in v["screened"]}
SCREEN_SHARE = CONFIG["screen_mixed_membership_share"]
UNIT_SIZES = {k: tuple(v["declared_unit_sizes"]) for k, v in CONFIG["datasets"].items()}
OVERLAPPING = [k for k, v in CONFIG["datasets"].items() if not v["units_declared_disjoint"]]
METHODS = CONFIG["methods"]
METHOD_FILE = {"AHRC": "ahrc", "AHCKA": "ahcka", "GRAC": "grac", "CAHC": "cahc",
               "CE+Fair-Louvain": "ce_fair_louvain"}


def data_root(path: str | os.PathLike | None = None) -> Path:
    """Folder that holds the dataset folders (default: <repo>/data, or $AHC_DATA)."""
    if path:
        return Path(path)
    return Path(os.environ.get("AHC_DATA", REPO_ROOT / "data"))


def partition_root(path: str | os.PathLike | None = None) -> Path:
    if path:
        return Path(path)
    return Path(os.environ.get("AHC_PARTITIONS", REPO_ROOT / "data" / "partitions"))


def factor(series: pd.Series) -> np.ndarray:
    """Integer group codes; a missing value forms one additional group."""
    codes, _ = pd.factorize(series.fillna("__missing__"))
    return np.asarray(codes, dtype=np.int64)


def compact(x: np.ndarray) -> tuple[np.ndarray, int]:
    _, inv = np.unique(x, return_inverse=True)
    inv = inv.ravel().astype(np.int64)
    return inv, int(inv.max() + 1)


def read_partition(path: str | os.PathLike) -> np.ndarray:
    """Cluster id of every node, ordered by node_id."""
    df = pd.read_csv(path)
    col = "cluster" if "cluster" in df.columns else df.columns[1]
    return df.sort_values(df.columns[0])[col].to_numpy(dtype=np.int64)


def partition_path(dataset: str, method: str, root=None) -> Path:
    return partition_root(root) / dataset / f"{METHOD_FILE[method]}.csv.gz"


def load_incidence(dataset: str, root=None) -> sp.csr_matrix:
    """0/1 int64 CSR matrix, units x nodes, with sorted column indices."""
    H = sp.load_npz(data_root(root) / dataset / "incidence.npz").tocsr()
    H = H.copy()
    H.data[:] = 1
    H = H.astype(np.int64)
    H.sort_indices()
    return H


@dataclass
class Dataset:
    name: str
    H: sp.csr_matrix
    attrs: pd.DataFrame
    labels: np.ndarray
    partitions_dir: Path | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def display(self) -> str:
        return DISPLAY[self.name]

    @property
    def n(self) -> int:
        return self.H.shape[1]

    @property
    def m(self) -> int:
        return self.H.shape[0]

    @property
    def sizes(self) -> np.ndarray:
        return np.diff(self.H.indptr).astype(np.int64)

    @property
    def vol(self) -> int:
        return int(self.H.nnz)

    def group(self, attribute: str) -> np.ndarray:
        return factor(self.attrs[attribute])

    def partition(self, method: str) -> np.ndarray:
        key = ("partition", method)
        if key not in self._cache:
            cl = read_partition(partition_path(self.name, method, self.partitions_dir))
            assert cl.shape[0] == self.n, (self.name, method, cl.shape[0], self.n)
            self._cache[key] = cl
        return self._cache[key]


def load_dataset(name: str, root=None, partitions=None) -> Dataset:
    d = data_root(root) / name
    H = load_incidence(name, root)
    attrs = pd.read_csv(d / "sensitive_attrs.csv")
    assert len(attrs) == H.shape[1] and (attrs.iloc[:, 0].to_numpy() == np.arange(H.shape[1])).all()
    lab = pd.read_csv(d / "labels.csv")
    assert len(lab) == H.shape[1] and (lab.iloc[:, 0].to_numpy() == np.arange(H.shape[1])).all()
    # Codes in order of first appearance; the shared budget's random stream depends on this order.
    return Dataset(name=name, H=H, attrs=attrs, labels=factor(lab["label"]), partitions_dir=partitions)


def cells(datasets=None, include_screened=False):
    """The (dataset, method, attribute) cells in the paper's order: dataset, then attribute, then method.

    The position of a cell in cells() over all datasets is its cell index, which seeds the searches."""
    out = []
    for ds in datasets or DATASETS:
        for a in ATTRIBUTES[ds]:
            if (ds, a) in SCREENED and not include_screened:
                continue
            for meth in METHODS:
                out.append((ds, meth, a))
    return out


def cell_index(dataset, method, attribute) -> int:
    return cells().index((dataset, method, attribute))


def cell_index_in_dataset(dataset, method, attribute) -> int:
    return cells([dataset]).index((dataset, method, attribute))
