"""Shared helpers of the systems experiments (Practicality): DuckDB connection, memory sampling, inputs.

DuckDB runs in-process with 8 threads, preserve_insertion_order=false and an on-disk database under
outputs/scaling/db/ with a spill directory next to it. Exact HGP is a Fraction from integer per-size sums.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ahc_audit import data as dmod  # noqa: E402
from ahc_audit.hgp import unit_counts  # noqa: E402

OUT = REPO / "outputs" / "scaling"
DBD = OUT / "db"
clk = time.perf_counter
MARGIN = 1e-9          # certify iff bound <= budget - MARGIN
FACTORS = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]
DATASETS = dmod.DATASETS
SHORT = dmod.DISPLAY


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


class Con:
    """Always fetches the result of execute() (DuckDB 2.0 keeps a pending result otherwise)."""

    def __init__(self, raw):
        self.raw = raw

    def execute(self, sql, params=None):
        r = self.raw.execute(sql, params) if params is not None else self.raw.execute(sql)
        try:
            return _Rows(r.fetchall())
        except duckdb.InvalidInputException:
            return _Rows([])

    def register(self, name, df):
        self.raw.register(name, df)

    def unregister(self, name):
        self.raw.unregister(name)

    def close(self):
        self.raw.close()


def connect(path=":memory:", threads=8, memory_limit=None, temp=None):
    con = Con(duckdb.connect(str(path)))
    con.execute(f"SET threads={int(threads)}")
    if memory_limit:
        con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute("SET preserve_insertion_order=false")
    t = Path(temp) if temp else DBD / "tmp"
    t.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{t.as_posix()}'")
    return con


def settings(con):
    t, mem, tmp, ver = con.execute("SELECT current_setting('threads'), current_setting('memory_limit'),"
                                   " current_setting('temp_directory'), version()").fetchone()
    return {"threads": int(t), "memory_limit": mem, "duckdb_version": ver}


def load_incidence(ds, data_root=None):
    """(H csr units x nodes, unit ids, node ids) sorted by (unit, node)."""
    H = dmod.load_incidence(ds, data_root)
    unit = np.repeat(np.arange(H.shape[0], dtype=np.int64), np.diff(H.indptr))
    return H, unit, H.indices.astype(np.int64)


def load_attr_codes(ds, attrs, n, data_root=None):
    df = pd.read_csv(dmod.data_root(data_root) / ds / "sensitive_attrs.csv")
    assert len(df) == n and (df.iloc[:, 0].to_numpy() == np.arange(n)).all()
    return {a: dmod.factor(df[a]) for a in attrs}


def exact_bysize(H, cl, g):
    """numpy reference: per-size (k, sum Y, sum X, units); HGP = sum (sum Y)/k / vol, U_P = sum (sum X)/k / vol."""
    s, X, Y, _, _ = unit_counts(H, cl, g)
    return [(int(k), int(Y[s == k].sum()), int(X[s == k].sum()), int((s == k).sum())) for k in np.unique(s)]


def frac_from_bysize(rows, col):
    vol = sum(r[0] * r[-1] for r in rows)
    return sum((Fraction(int(r[col]), int(r[0])) for r in rows), Fraction(0)) / vol


class PeakRSS:
    """Samples the process RSS (DuckDB runs in-process) every 10 ms in a background thread."""

    def __init__(self, interval=0.01):
        import psutil
        self.p = psutil.Process(); self.interval = interval; self.peak = 0
        self._stop = threading.Event()

    def __enter__(self):
        self.peak = self.p.memory_info().rss
        self._t = threading.Thread(target=self._run, daemon=True); self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, self.p.memory_info().rss)
            except Exception:
                break
            time.sleep(self.interval)

    def __exit__(self, *a):
        self._stop.set(); self._t.join(timeout=2)
        self.peak = max(self.peak, self.p.memory_info().rss)


def dir_size(p):
    p = Path(p)
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def environment(con=None):
    import psutil
    meta = {"processor": platform.processor(), "logical_cpus": psutil.cpu_count(logical=True),
            "ram_total_gb": round(psutil.virtual_memory().total / 1024 ** 3, 2), "os": platform.platform(),
            "python": platform.python_version(), "duckdb": duckdb.__version__, "numpy": np.__version__,
            "pandas": pd.__version__}
    if con is not None:
        meta["duckdb_settings"] = settings(con)
    return meta


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)


def append_csv(path, row: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8")


def natural_cells(cells_csv, ds):
    """The five natural partitions and the shared budgets of one dataset, from outputs/cells.csv."""
    c = pd.read_csv(cells_csv)
    c = c[(c.ds == ds) & ~c.apply(lambda r: (r.ds, r.attribute) in dmod.SCREENED, axis=1)]
    assert len(c), f"no cells of {ds} in {cells_csv}; run run_cells.py first"
    return c
