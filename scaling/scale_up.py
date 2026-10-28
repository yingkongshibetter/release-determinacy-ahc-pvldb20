"""Scale-up of exact HGP and of the U_P scan on DuckDB with a memory limit below the data size (Practicality).

usage: python scaling/scale_up.py <config> <phase> [--mem 4GB] [--threads 8] [--k 8]
  config  <dataset>_x<R>: R disjoint copies of data/<dataset> (node and unit ids offset per copy, the same
          attribute value and cluster per copy, so HGP and U_P do not change). The paper uses imdb_cast_full
          (films and TV series, 5.70M memberships) with R = 1, 2, 4, 8, 18, 36; x36 has 2.05e8 memberships.
  phase   ref     numpy reference on R = 1 (exact HGP and U_P)
          load    build the on-disk DuckDB tables Incidence, Attribute, Partition
          exact   the exact SQL query (per-size counts and sums), two runs
          release producer: degrees and T pruned to the k largest entries per row, in SQL chunks
          fullP   exact P and the U_P scan (small configs only)
          scan    auditor: symmetric closure of the pruned release and the bound of Prop. 5(b)
Each phase runs in its own process and appends one row to outputs/scaling/scale_up/<phase>.csv.
Partition: imdb_cast_full uses data/partitions/imdb_cast_full/ahrc.npz (AHRC); the six paper datasets use
their AHRC partition.
"""
from __future__ import annotations

import argparse
import re
import sys
import threading
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import clk, np, pd  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("phase")
ap.add_argument("--mem", default="4GB")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--chunk-pairs", type=float, default=4e7)
ap.add_argument("--tag", default="")
ap.add_argument("--data-root", default=None)
args = ap.parse_args()
cfg = args.config
mo = re.match(r"(.+)_x(\d+)$", cfg)
src, R = mo.group(1), int(mo.group(2))
ATTR = {"imdb_cast_full": "gender", "movielens_1m": "gender", "openalex_db": "country",
        "openalex_cs_size3plus": "country", "pokec_hobby": "gender", "imdb_cast": "birth_cohort",
        "acs_pums_household": "sex"}[src]
OUT = C.OUT / "scale_up"
OUT.mkdir(parents=True, exist_ok=True)
C.DBD.mkdir(parents=True, exist_ok=True)
dbf = C.DBD / f"scale_{cfg}.duckdb"
tmpd = C.DBD / f"tmp_scale_{cfg}"
K = args.k


def partition_of(ds, n):
    if ds == "imdb_cast_full":
        cl = np.load(C.dmod.partition_root() / "imdb_cast_full" / "ahrc.npz")["cluster"].astype(np.int64)
    else:
        cl = C.dmod.read_partition(C.dmod.partition_path(ds, "AHRC"))
    assert cl.size == n
    return cl


class TempWatch:
    """Max size of the DuckDB temp (spill) directory, sampled every 0.5 s."""

    def __init__(self, d):
        self.d = Path(d); self.peak = 0; self._s = threading.Event()

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True); self._t.start(); return self

    def _run(self):
        while not self._s.is_set():
            try:
                self.peak = max(self.peak, C.dir_size(self.d))
            except Exception:
                pass
            self._s.wait(0.5)

    def __exit__(self, *a):
        self._s.set(); self._t.join(timeout=5)


def record(phase, info):
    phase = phase + args.tag
    row = {"config": cfg, "source": src, "replicas": R, "phase": phase, "mem_limit": args.mem,
           "threads": args.threads, **info}
    C.append_csv(OUT / f"{phase}.csv", row)
    C.write_json(OUT / f"{cfg}_{phase}.json", row)
    print(row, flush=True)


def timed(con, sql, params=None):
    with C.PeakRSS() as pr, TempWatch(tmpd) as tw:
        t0 = clk(); r = con.execute(sql, params).fetchall(); dt = clk() - t0
    return dt, r, pr.peak, tw.peak


SQL_L = """CREATE OR REPLACE TABLE L AS SELECT a.node_id AS node_id, a.g AS g, p.c AS c
  FROM Attribute a JOIN Partition p ON p.node_id = a.node_id"""
SQL_EX = """WITH s AS MATERIALIZED (SELECT i.unit_id AS e, L.g AS g, L.c AS c, COUNT(*) AS n
      FROM Incidence i JOIN L ON L.node_id = i.node_id GROUP BY 1, 2, 3),
u AS (SELECT e, SUM(n) AS k FROM s GROUP BY e),
gm AS (SELECT e, g, SUM(n) AS ng FROM s GROUP BY e, g),
cm AS (SELECT e, c, SUM(n) AS nc FROM s GROUP BY e, c)
SELECT u.k, COUNT(DISTINCT s.e), SUM(CASE WHEN u.k*s.n > gm.ng*cm.nc THEN u.k*s.n - gm.ng*cm.nc ELSE 0 END)
FROM s JOIN u USING(e) JOIN gm USING(e, g) JOIN cm USING(e, c) GROUP BY u.k"""


def hgp_of(rows):
    rows = sorted((int(k), int(d), int(u)) for k, u, d in rows)
    acc = sum((Fraction(d, k) for k, d, _ in rows), Fraction(0))
    vol = sum(k * u for k, _, u in rows)
    return acc / vol, vol, rows


phase = args.phase
if phase == "ref":
    # numpy reference on the original (R=1) data: exact HGP and exact U_P as fractions
    H, unit, node = C.load_incidence(src, args.data_root)
    n = H.shape[1]
    g = C.load_attr_codes(src, [ATTR], n, args.data_root)[ATTR]
    cl = partition_of(src, n)
    t0 = clk(); ref = C.exact_bysize(H, cl, g); dt = clk() - t0
    h = C.frac_from_bysize(ref, 1); up = C.frac_from_bysize(ref, 2)
    C.write_json(OUT / f"ref_{src}.json", {"source": src, "attr": ATTR, "HGP": float(h), "U_P": float(up),
                                           "HGP_frac": f"{h.numerator}/{h.denominator}",
                                           "U_P_frac": f"{up.numerator}/{up.denominator}",
                                           "bysize": ref, "t_numpy_s": dt, "memberships": int(unit.size),
                                           "n": int(n), "m": int(H.shape[0])})
    print("ref", src, float(h), float(up), dt)
    sys.exit(0)

if phase == "load":
    for f in (dbf, Path(str(dbf) + ".wal")):
        if f.exists():
            f.unlink()
H, unit, node = (None, None, None)
con = C.connect(dbf, threads=args.threads, memory_limit=args.mem, temp=tmpd)

if phase == "load":
    H, unit, node = C.load_incidence(src, args.data_root)
    m, n = H.shape
    g = C.load_attr_codes(src, [ATTR], n, args.data_root)[ATTR]
    cl = partition_of(src, n)
    t0 = clk()
    con.register("inc_df", pd.DataFrame({"unit_id": unit.astype(np.int32), "node_id": node.astype(np.int32)}))
    con.register("lab_df", pd.DataFrame({"node_id": np.arange(n, dtype=np.int32), "g": g.astype(np.int32),
                                         "c": cl.astype(np.int32)}))
    con.execute(f"""CREATE TABLE Incidence AS SELECT (unit_id + r.range * {m})::INTEGER AS unit_id,
                    (node_id + r.range * {n})::INTEGER AS node_id FROM inc_df, range({R}) r
                    ORDER BY unit_id, node_id""")
    con.execute(f"""CREATE TABLE Attribute AS SELECT (node_id + r.range * {n})::INTEGER AS node_id, g
                    FROM lab_df, range({R}) r ORDER BY node_id""")
    con.execute(f"""CREATE TABLE Partition AS SELECT (node_id + r.range * {n})::INTEGER AS node_id, c
                    FROM lab_df, range({R}) r ORDER BY node_id""")
    con.unregister("inc_df"); con.unregister("lab_df")
    con.execute("CHECKPOINT")
    dt = clk() - t0
    con.close()
    record("load", {"t_s": dt, "memberships": int(unit.size) * R, "n": n * R, "m": m * R,
                    "sum_s2": int((np.diff(H.indptr).astype(np.int64) ** 2).sum()) * R,
                    "db_bytes": dbf.stat().st_size, "raw_incidence_bytes": int(unit.size) * R * 8})
    sys.exit(0)

memb, n_all = (int(x) for x in con.execute("SELECT COUNT(*), (SELECT COUNT(*) FROM Attribute) FROM Incidence").fetchone())

if phase == "exact":
    out = {"memberships": memb, "n": n_all}
    for rep in range(2):
        dl, _, _, _ = timed(con, SQL_L)
        out[f"t_label_s_run{rep}"] = dl
        dt, rows, peak, tpk = timed(con, SQL_EX)
        h, vol, br = hgp_of(rows)
        out[f"t_s_run{rep}"] = dt; out[f"peak_rss_run{rep}"] = peak; out[f"peak_temp_run{rep}"] = tpk
    out["t_s"] = out["t_s_run0"]; out["peak_rss"] = out["peak_rss_run0"]; out["peak_temp"] = out["peak_temp_run0"]
    out["HGP"] = float(h)
    ref = OUT / f"ref_{src}.json"
    if ref.exists():
        import json
        rj = json.load(open(ref, encoding="utf-8"))
        a, b = rj["HGP_frac"].split("/")
        out["HGP_equals_reference"] = int(h == Fraction(int(a), int(b)))
    record("exact", out)

elif phase == "fullP":
    con.execute("DROP TABLE IF EXISTS P")
    dt, _, peak, tpk = timed(con, """CREATE TABLE P AS
      WITH z AS (SELECT unit_id, COUNT(*) AS k FROM Incidence GROUP BY unit_id)
      SELECT a.node_id AS u, b.node_id AS v, SUM(1.0::DOUBLE / z.k) AS w
      FROM Incidence a JOIN Incidence b ON a.unit_id = b.unit_id AND a.node_id <= b.node_id
           JOIN z ON z.unit_id = a.unit_id GROUP BY 1, 2""")
    con.execute("CHECKPOINT")
    rows = int(con.execute("SELECT COUNT(*) FROM P").fetchone()[0])
    # U_P scan on the exact release (two runs)
    con.execute("CREATE OR REPLACE TABLE Lab AS SELECT a.node_id, a.g, p.c FROM Attribute a JOIN Partition p USING(node_id)")
    q = """SELECT SUM(CASE WHEN lu.g <> lv.g AND lu.c <> lv.c THEN P.w ELSE 0 END)
           FROM P JOIN Lab lu ON lu.node_id = P.u JOIN Lab lv ON lv.node_id = P.v"""
    ts = []
    for rep in range(2):
        d2, r, pk2, tp2 = timed(con, q); ts.append(d2)
    up = float(r[0][0] or 0.0) * 2.0 / memb
    record("fullP", {"memberships": memb, "n": n_all, "t_s": dt, "peak_rss": peak, "peak_temp": tpk,
                     "P_rows_upper": rows, "bytes_P": rows * 16, "t_scan_s": ts[0], "t_scan_s_run1": ts[1],
                     "peak_rss_scan": pk2, "U_P": up})
    con.execute("DROP TABLE P"); con.execute("CHECKPOINT")

elif phase == "release":
    for t in (f"T{K}", "IncN", "Z", "Deg"):
        con.execute(f"DROP TABLE IF EXISTS {t}")
    info = {"memberships": memb, "n": n_all, "k": K}
    with C.PeakRSS() as pr, TempWatch(tmpd) as tw:
        t0 = clk()
        con.execute("CREATE TABLE Z AS SELECT unit_id, COUNT(*)::INTEGER AS k FROM Incidence GROUP BY 1")
        con.execute("CREATE TABLE Deg AS SELECT node_id AS u, COUNT(*)::INTEGER AS d FROM Incidence GROUP BY 1")
        con.execute("CREATE TABLE IncN AS SELECT node_id, unit_id FROM Incidence ORDER BY node_id")
        info["t_prep_s"] = clk() - t0
        # row work per node = sum of sizes of its units (ordered pairs before merging); chunk boundaries are
        # computed in SQL (running sum over node order), only one row per chunk reaches Python
        target = int(args.chunk_pairs)
        ch = con.execute(f"""WITH w AS (SELECT i.node_id, SUM(z.k) AS wk FROM IncN i JOIN Z z ON z.unit_id = i.unit_id
                               GROUP BY 1),
                             c AS (SELECT node_id, (SUM(wk) OVER (ORDER BY node_id) - 1) // {target} AS cid FROM w)
                             SELECT cid, MIN(node_id), MAX(node_id) FROM c GROUP BY cid ORDER BY cid""").fetchall()
        info["t_chunking_s"] = clk() - t0 - info["t_prep_s"]
        info["chunks"] = len(ch)
        con.execute(f"CREATE TABLE T{K}(u INTEGER, v INTEGER, t DOUBLE)")
        for ci, (_, lo, hi) in enumerate(ch):
            lo = int(lo); hi = int(hi)
            con.execute(f"""INSERT INTO T{K}
              SELECT r.u, r.v, r.w / d.d FROM
                (SELECT a.node_id AS u, b.node_id AS v, SUM(1.0::DOUBLE / z.k) AS w
                 FROM IncN a JOIN Z z ON z.unit_id = a.unit_id JOIN Incidence b ON b.unit_id = a.unit_id
                 WHERE a.node_id BETWEEN {lo} AND {hi} GROUP BY 1, 2) r
                JOIN Deg d ON d.u = r.u
              QUALIFY row_number() OVER (PARTITION BY r.u ORDER BY r.w DESC, r.v) <= {K}""")
            if ci % 5 == 0:
                print(f"chunk {ci+1}/{len(ch)} t={clk()-t0:.1f}s", flush=True)
        con.execute("CHECKPOINT")
        info["t_s"] = clk() - t0
    info["peak_rss"] = pr.peak; info["peak_temp"] = tw.peak
    info["T_rows"] = int(con.execute(f"SELECT COUNT(*) FROM T{K}").fetchone()[0])
    info["bytes_release_T_deg"] = info["T_rows"] * 16 + n_all * 4
    con.execute("DROP TABLE IncN"); con.execute("CHECKPOINT")
    record("release", info)

elif phase == "scan":
    info = {"memberships": memb, "n": n_all, "k": K}
    # auditor label table (attribute-side, joined once): node, g, c
    con.execute("DROP TABLE IF EXISTS Lab")
    con.execute("DROP TABLE IF EXISTS S")
    t0 = clk()
    con.execute("CREATE TABLE Lab AS SELECT a.node_id, a.g, p.c FROM Attribute a JOIN Partition p USING(node_id)")
    info["t_lab_s"] = clk() - t0
    # (i) one-time attribute-free preprocessing: symmetric closure of the listed entries, exact values d_u*t
    dt, _, pk, tp = timed(con, f"""CREATE TABLE S AS SELECT LEAST(t.u, t.v) AS a, GREATEST(t.u, t.v) AS b,
                                    MAX(t.t * d.d) AS w FROM T{K} t JOIN Deg d ON d.u = t.u GROUP BY 1, 2""")
    info["t_closure_s"] = dt; info["peak_rss_closure"] = pk; info["peak_temp_closure"] = tp
    info["S_rows"] = int(con.execute("SELECT COUNT(*) FROM S").fetchone()[0])
    vol = int(con.execute("SELECT SUM(d) FROM Deg").fetchone()[0])
    assert vol == memb
    q = """SELECT SUM(CASE WHEN lu.g <> lv.g AND lu.c <> lv.c THEN 0 WHEN S.a = S.b THEN S.w ELSE 2 * S.w END)
           FROM S JOIN Lab lu ON lu.node_id = S.a JOIN Lab lv ON lv.node_id = S.b"""
    for rep in range(2):
        d2, r, pk2, tp2 = timed(con, q)
        info[f"t_scan_s_run{rep}"] = d2; info[f"peak_rss_scan_run{rep}"] = pk2; info[f"peak_temp_scan_run{rep}"] = tp2
    info["t_s"] = info["t_scan_s_run0"]; info["peak_rss"] = info["peak_rss_scan_run0"]
    info["peak_temp"] = info["peak_temp_scan_run0"]
    info["bound_T"] = 1.0 - float(r[0][0]) * (1 - 1e-12) / vol
    # (ii) direct scan of the released rows without the closure (sound, looser): listed ordered entries only
    q2 = f"""SELECT SUM(CASE WHEN lu.g <> lv.g AND lu.c <> lv.c THEN 0 ELSE t.t * lu.d END)
             FROM T{K} t JOIN (SELECT Lab.*, Deg.d FROM Lab JOIN Deg ON Deg.u = Lab.node_id) lu
             ON lu.node_id = t.u JOIN Lab lv ON lv.node_id = t.v"""
    d3, r3, pk3, tp3 = timed(con, q2)
    info["t_scan_direct_s"] = d3; info["bound_T_direct"] = 1.0 - float(r3[0][0]) * (1 - 1e-12) / vol
    ref = OUT / f"ref_{src}.json"
    if ref.exists():
        import json
        rj = json.load(open(ref, encoding="utf-8"))
        info["U_P_ref"] = rj["U_P"]; info["HGP_ref"] = rj["HGP"]
        info["bound_ge_UP"] = int(info["bound_T"] >= rj["U_P"] - 1e-12 and info["bound_T_direct"] >= info["bound_T"] - 1e-12)
    record("scan", info)
con.close()
