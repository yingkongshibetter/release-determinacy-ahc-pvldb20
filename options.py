"""Command-line arguments shared by the run_*.py scripts."""
from __future__ import annotations

import argparse
from pathlib import Path

from ahc_audit.data import DATASETS, DISPLAY, INTERNAL, METHODS, REPO_ROOT

OUT_DEFAULT = REPO_ROOT / "outputs"


def dataset_name(x: str) -> str:
    """Accept the folder name (movielens_1m) or the paper name (ML-1M)."""
    if x in DATASETS:
        return x
    if x in INTERNAL:
        return INTERNAL[x]
    raise argparse.ArgumentTypeError(f"unknown dataset {x}; use one of {DATASETS} or {list(DISPLAY.values())}")


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data-root", type=Path, default=None,
                   help="folder holding the dataset folders (default: data/, or $AHC_DATA)")
    p.add_argument("--partitions", type=Path, default=None,
                   help="folder holding <dataset>/<method>.csv.gz (default: data/partitions/)")
    p.add_argument("--out", type=Path, default=OUT_DEFAULT, help="output folder (default: outputs/)")
    return p


def add_datasets(p: argparse.ArgumentParser, default=None):
    p.add_argument("--datasets", type=dataset_name, nargs="+", default=default or DATASETS,
                   help="datasets to run (default: all six)")


def add_cell(p: argparse.ArgumentParser):
    p.add_argument("--dataset", type=dataset_name, required=True)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--attribute", required=True)
    p.add_argument("--budget", type=float, default=None,
                   help="budget B on HGP (default: replay the shared budget of the paper)")
    p.add_argument("--tau", type=float, default=None, help="budget on the dissimilarity scale Dbar")
