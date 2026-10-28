"""Shared helpers for the dataset builders.

Every builder writes the same layout into ``data/<folder>/``:

    incidence.npz          scipy.sparse CSR, units x nodes, binary
    node_features.npz      scipy.sparse CSR, nodes x features (not for imdb_cast_full)
    sensitive_attrs.csv    node_id + audited attributes
    labels.csv             node_id, label
    node_mapping.csv       node_id -> source identifier
    hyperedge_mapping.csv  unit id -> source identifier
    metadata.json          construction rules, filters, sha256 of each file
    diagnostics.json       size and attribute summaries
    README.md              short description

Two details make the outputs byte-identical to the files used in the paper:

* CSV line endings. Some of the paper files were written on Linux ("\\n")
  and some on Windows ("\\r\\n"). Each builder passes the line ending of the
  corresponding paper file explicitly, so the result does not depend on the
  operating system.
* The npz container. ``scipy.sparse.save_npz`` stores the operating system in
  the zip headers and compresses with whatever zlib the Python build ships.
  ``save_npz`` below fixes the zip header field. The compressed bytes still
  depend on the deflate implementation (standard zlib versus zlib-ng), so
  ``check_data.py`` also compares a hash of the decompressed arrays.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.lib import format as npy_format

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

RANDOM_SEED = 2026

LF = "\n"
CRLF = "\r\n"

# zip "version made by" host system: 0 = MS-DOS/Windows, 3 = Unix.
ZIP_HOST_WINDOWS = 0
ZIP_HOST_UNIX = 3

DATA_FILES = (
    "incidence.npz",
    "node_features.npz",
    "sensitive_attrs.csv",
    "labels.csv",
    "node_mapping.csv",
    "hyperedge_mapping.csv",
)


# ---------------------------------------------------------------- hashing


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_lf(path: Path) -> str:
    """sha256 of a text file after converting CRLF line endings to LF."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def npz_content_sha256(path: Path) -> str:
    """sha256 over the decompressed .npy members of an npz file.

    Independent of the zip container (compression level, zlib build, host
    system field), so two files with the same arrays get the same value.
    """
    h = hashlib.sha256()
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            h.update(name.encode("utf-8") + b"\0")
            h.update(z.read(name))
    return h.hexdigest()


# ---------------------------------------------------------------- writers


def save_npz(path: Path, matrix: sp.spmatrix, *, zip_host: int) -> None:
    """Write a CSR matrix in the layout of ``scipy.sparse.save_npz``.

    Same members, member order, .npy headers and deflate settings as
    ``save_npz(compressed=True)``; the zip host-system field is set
    explicitly so the file does not depend on the operating system.
    """
    m = matrix.tocsr()
    arrays = {
        "indices": m.indices,
        "indptr": m.indptr,
        "format": np.array(m.format.encode("ascii")),
        "shape": np.array(m.shape, dtype=np.int64),
        "data": m.data,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED,
                         allowZip64=True) as zf:
        for key, value in arrays.items():
            info = zipfile.ZipInfo(key + ".npy")  # fixed 1980-01-01 timestamp
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = zip_host
            with zf.open(info, "w", force_zip64=True) as fid:
                npy_format.write_array(fid, np.asanyarray(value), allow_pickle=False)


def write_csv(df: pd.DataFrame, path: Path, *, eol: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator=eol)


def csv_roundtrip(df: pd.DataFrame) -> pd.DataFrame:
    """Write ``df`` to CSV and read it back with pandas defaults.

    Some paper files were produced by a script that re-read an earlier CSV
    output; reproducing that step in memory keeps pandas' type inference
    (for example, empty strings becoming NaN) identical.
    """
    return pd.read_csv(io.StringIO(df.to_csv(index=False, lineterminator=LF)))


def write_json(path: Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default),
                          encoding="utf-8")


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)!r}")


def file_hashes(out_dir: Path, names: Iterable[str] = DATA_FILES) -> dict[str, str]:
    return {n: sha256_file(out_dir / n) for n in names if (out_dir / n).exists()}


def basic_diagnostics(incidence: sp.spmatrix, sensitive: pd.DataFrame,
                      labels: pd.DataFrame) -> dict:
    H = incidence.tocsr()
    sizes = np.diff(H.indptr)
    degrees = np.bincount(H.indices, minlength=H.shape[1])
    out = {
        "nodes": int(H.shape[1]),
        "units": int(H.shape[0]),
        "memberships": int(H.nnz),
        "unit_size": {"min": int(sizes.min()), "median": float(np.median(sizes)),
                      "mean": float(sizes.mean()), "max": int(sizes.max())},
        "node_degree": {"min": int(degrees.min()), "median": float(np.median(degrees)),
                        "mean": float(degrees.mean()), "max": int(degrees.max())},
        "label_classes": int(labels["label"].nunique(dropna=False)),
        "attributes": {},
    }
    for col in sensitive.columns:
        if col == "node_id":
            continue
        vc = sensitive[col].astype("object").fillna("<missing>").value_counts()
        out["attributes"][col] = {"classes": int(len(vc)),
                                  "largest_share": float(vc.iloc[0] / vc.sum())}
    return out


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finish(out_dir: Path, metadata: dict, readme: str, diagnostics: dict) -> None:
    """Add file hashes to ``metadata`` and write metadata, diagnostics, README."""
    metadata = dict(metadata)
    metadata["file_sha256"] = file_hashes(out_dir)
    write_json(out_dir / "metadata.json", metadata)
    write_json(out_dir / "diagnostics.json", diagnostics)
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def require(path: Path, hint: str) -> Path:
    if not Path(path).exists():
        raise SystemExit(f"missing input: {path}\n{hint}")
    return Path(path)
