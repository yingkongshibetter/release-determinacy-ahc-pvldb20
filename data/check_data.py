"""Check processed datasets against the files used in the paper.

For each dataset in configs/datasets.json this verifies
  * the sha256 of incidence.npz, node_features.npz, sensitive_attrs.csv and
    labels.csv (node_mapping.csv and hyperedge_mapping.csv are reported but
    a mismatch there is only a warning),
  * the counts of nodes, units and memberships (Table 3 of the paper),
  * the unit size range, binary incidence, and that sensitive_attrs.csv and
    labels.csv have one row per node with the audited attribute columns.

If an npz file differs only in its zip container (another zlib build wrote
it), the arrays are compared instead and the result is "ok (same arrays)".
If a CSV differs only in CRLF/LF line endings (for example after a git
checkout with core.autocrlf), the result is "ok (line endings)". With
--strict both count as failures.

Usage:
    python data/check_data.py                    # every dataset found under data/
    python data/check_data.py --dataset imdb_cast movielens_1m
    python data/check_data.py --data-root DIR    # DIR/<folder>/ instead of data/<folder>/
Exit status is 1 if a checked dataset fails or a requested one is missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "datasets.json"
CORE = ("incidence.npz", "node_features.npz", "sensitive_attrs.csv", "labels.csv")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def npz_content_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            h.update(name.encode("utf-8") + b"\0")
            h.update(z.read(name))
    return h.hexdigest()


def check_file(path: Path, ref: dict, strict: bool) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    data = path.read_bytes()
    if sha256_bytes(data) == ref["sha256"]:
        return True, "ok"
    if path.suffix == ".npz" and "content_sha256" in ref:
        try:
            same = npz_content_sha256(path) == ref["content_sha256"]
        except zipfile.BadZipFile:
            same = False
        if same:
            return (not strict), "ok (same arrays, different zip container)"
    if path.suffix == ".csv" and "sha256_lf" in ref:
        if sha256_bytes(data.replace(b"\r\n", b"\n")) == ref["sha256_lf"]:
            return (not strict), "ok (line endings differ)"
    return False, "sha256 differs"


def check_dataset(name: str, spec: dict, folder: Path, strict: bool) -> tuple[bool, list[str]]:
    lines, ok = [], True
    for fn, ref in spec["files"].items():
        good, msg = check_file(folder / fn, ref, strict)
        if fn in CORE:
            ok &= good
            lines.append(f"    {fn:22s} {msg}")
        elif not good:
            lines.append(f"    {fn:22s} {msg} (warning only)")
        else:
            lines.append(f"    {fn:22s} {msg}")

    exp = spec["expected"]
    try:
        H = sp.load_npz(folder / "incidence.npz").tocsr()
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        return False, lines + [f"    cannot read incidence.npz: {exc}"]
    n_nodes, n_units, n_mem = H.shape[1], H.shape[0], int(H.nnz)
    counts_ok = (n_nodes, n_units, n_mem) == (exp["nodes"], exp["units"], exp["memberships"])
    ok &= counts_ok
    lines.append(f"    counts nodes={n_nodes:,} units={n_units:,} memberships={n_mem:,} "
                 + ("ok" if counts_ok else
                    f"MISMATCH (expected {exp['nodes']:,} / {exp['units']:,} / {exp['memberships']:,})"))
    sizes = np.diff(H.indptr)
    lo, hi = spec["unit_size_range"]
    size_ok = n_units > 0 and sizes.min() >= lo and sizes.max() <= hi
    binary_ok = bool(np.all(H.data == 1))
    ok &= size_ok and binary_ok
    lines.append(f"    unit sizes {sizes.min() if n_units else 0}..{sizes.max() if n_units else 0} "
                 f"(allowed {lo}..{hi}) {'ok' if size_ok else 'OUT OF RANGE'}; "
                 f"binary {'ok' if binary_ok else 'NO'}")

    try:
        sens = pd.read_csv(folder / "sensitive_attrs.csv", low_memory=False)
        labels = pd.read_csv(folder / "labels.csv", low_memory=False)
    except (OSError, pd.errors.ParserError) as exc:
        return False, lines + [f"    cannot read CSV files: {exc}"]
    missing = [a for a in spec["attributes"] if a not in sens.columns]
    rows_ok = (len(sens) == n_nodes and len(labels) == n_nodes
               and np.array_equal(sens["node_id"].to_numpy(), np.arange(n_nodes))
               and spec["label_column"] in labels.columns)
    ok &= rows_ok and not missing
    lines.append(f"    attributes {', '.join(spec['attributes'])} "
                 + ("ok" if not missing else f"MISSING {missing}")
                 + f"; one row per node {'ok' if rows_ok else 'NO'}")
    return ok, lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", nargs="+", default=None,
                    help="dataset folder names (default: all in configs/datasets.json)")
    ap.add_argument("--data-root", type=Path, default=None,
                    help="directory that holds the dataset folders (default: data/)")
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--strict", action="store_true",
                    help="require byte-identical files (no array or line-ending fallback)")
    a = ap.parse_args()

    cfg = json.loads(a.config.read_text(encoding="utf-8"))["datasets"]
    names = a.dataset or list(cfg)
    unknown = [n for n in names if n not in cfg]
    if unknown:
        print(f"unknown dataset(s): {unknown}; choose from {list(cfg)}")
        return 2
    failed, missing, passed = [], [], []
    for name in names:
        spec = cfg[name]
        folder = (a.data_root / name) if a.data_root else (REPO_ROOT / spec["folder"])
        if not folder.is_dir():
            missing.append(name)
            hint = ("shipped with the repository" if spec["shipped"] else
                    f"build it: {spec['download']} ; {spec['build']}")
            print(f"[missing] {name} ({spec['paper_name']}): no folder {folder.name}/ ({hint})")
            continue
        ok, lines = check_dataset(name, spec, folder, a.strict)
        print(f"[{'PASS' if ok else 'FAIL'}] {name} ({spec['paper_name']})")
        print("\n".join(lines))
        (passed if ok else failed).append(name)
    print(f"\nsummary: {len(passed)} passed, {len(failed)} failed, {len(missing)} missing")
    if failed:
        print("failed: " + ", ".join(failed))
    requested_missing = missing if a.dataset else []
    return 1 if failed or requested_missing else 0


if __name__ == "__main__":
    sys.exit(main())
