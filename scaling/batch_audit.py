"""Batch auditing on DuckDB: certify-then-request against exact evaluation of every audit (Practicality).

usage: python scaling/batch_audit.py <dataset> [--reps 3] [--k 8,32] [--no-growth]

Tables (one on-disk DuckDB file per dataset, rebuilt from scratch):
  producer : Incidence(unit_id, node_id)
  auditor  : Attribute(node_id, attr_id, g), Partition(part_id, node_id, c)
  release  : P(u, v, w), u <= v, w = sum over shared units of 1/|e| (random-walk release, symmetric storage);
             Tk(u, v, t), the k largest entries of each row of T = D^-1 P (diagonal included), and Deg(u, d)
Workload: (attribute, partition) pairs x the nine budget factors 0.6..2.0 of the shared budget B. Base pairs
are the five natural partitions; growth pairs add perturbations of each natural partition (a fraction t of
nodes reassigned to a random existing cluster, t in {.01, .02, .05, .1, .2}, seeds 0-2).
(a) = certify-then-request: U_P scan of P for every pair, exact evaluation only for pairs left open.
(b) = exact evaluation of every pair. Every certified audit is checked against exact HGP.
Inputs: the dataset folder, the partitions, and outputs/cells.csv (budgets B, HGP and U_P of the natural cells).
Outputs: outputs/scaling/batch/<dataset>_{pairs,audits,batch,growth}.csv and <dataset>_meta.json.
"""
from __future__ import annotations

import argparse
import math
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import clk, np, pd  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ds")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--k", default="8,32")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--no-growth", action="store_true")
ap.add_argument("--data-root", default=None)
ap.add_argument("--partitions", default=None)
ap.add_argument("--cells", default=str(C.REPO / "outputs" / "cells.csv"))
args = ap.parse_args()
ds = args.ds
KS = [int(x) for x in args.k.split(",")]
OUT = C.OUT / "batch"
OUT.mkdir(parents=True, exist_ok=True)
for suf in ("_pairs.csv", "_audits.csv", "_batch.csv", "_growth.csv"):
    p = OUT / f"{ds}{suf}"
    if p.exists():
        p.unlink()
(C.OUT / "logs").mkdir(parents=True, exist_ok=True)
log = open(C.OUT / "logs" / f"batch_{ds}.log", "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")
    log.flush()


# ---------------------------------------------------------------- inputs
base = C.natural_cells(args.cells, ds)
attrs = list(dict.fromkeys(base.attribute.tolist()))
Bof = {a: float(base[base.attribute == a].B.iloc[0]) for a in attrs}
methods = [m for m in C.dmod.METHODS if m in set(base.method)]
ref_hgp = {(r.method, r.attribute): float(r.HGP) for r in base.itertuples()}
ref_up = {(r.method, r.attribute): float(r.U_P) for r in base.itertuples()}

H, unit, node = C.load_incidence(ds, args.data_root)
m, n = H.shape
vol = int(unit.size)
sizes = np.diff(H.indptr)
nsizes = int(np.unique(sizes).size)
G = C.load_attr_codes(ds, attrs, n, args.data_root)
P_list = []   # (part_id, method, t, seed, clusters)
for mi, meth in enumerate(methods):
    cl = C.dmod.read_partition(C.dmod.partition_path(ds, meth, args.partitions))
    assert cl.size == n, (meth, cl.size, n)
    P_list.append((len(P_list), meth, 0.0, -1, cl))
if not args.no_growth:
    for seed in (0, 1, 2):
        for t in (0.01, 0.02, 0.05, 0.1, 0.2):
            for mi, meth in enumerate(methods):
                cl = P_list[mi][4]
                rng = np.random.default_rng(1_000_003 * seed + int(round(t * 1000)) * 101 + mi)
                sel = rng.choice(n, size=max(1, int(round(t * n))), replace=False)
                cl2 = cl.copy()
                cl2[sel] = rng.choice(np.unique(cl), size=sel.size)
                P_list.append((len(P_list), meth, t, seed, cl2))
pairs = [(pid, ai) for pid, *_ in P_list for ai in range(len(attrs))]
nbase = len(methods) * len(attrs)
say(f"{ds}: n={n} m={m} vol={vol} sizes={nsizes} attrs={attrs} methods={methods} "
    f"partitions={len(P_list)} pairs={len(pairs)} base_pairs={nbase}")

# ---------------------------------------------------------------- DuckDB
C.DBD.mkdir(parents=True, exist_ok=True)
dbf = C.DBD / f"batch_{ds}.duckdb"
for f in (dbf, Path(str(dbf) + ".wal")):
    if f.exists():
        f.unlink()
con = C.connect(dbf, threads=args.threads, memory_limit="8GB")
meta = {"dataset": ds, "n": n, "m": m, "vol": vol, "n_sizes": nsizes, "attrs": attrs, "B": Bof,
        "methods": methods, "partitions": len(P_list), "pairs": len(pairs), "base_pairs": nbase,
        "env": C.environment(con), "margin": C.MARGIN, "factors": C.FACTORS, "k": KS}

t0 = clk()
con.register("inc_df", pd.DataFrame({"unit_id": unit.astype(np.int32), "node_id": node.astype(np.int32)}))
con.execute("CREATE TABLE Incidence AS SELECT * FROM inc_df")
con.unregister("inc_df")
att = pd.concat([pd.DataFrame({"node_id": np.arange(n, dtype=np.int32), "attr_id": np.int32(ai),
                               "g": G[a].astype(np.int32)}) for ai, a in enumerate(attrs)])
con.register("att_df", att)
con.execute("CREATE TABLE Attribute AS SELECT * FROM att_df")
con.unregister("att_df")
par = pd.concat([pd.DataFrame({"part_id": np.int32(pid), "node_id": np.arange(n, dtype=np.int32),
                               "c": cl.astype(np.int32)}) for pid, _, _, _, cl in P_list])
con.register("par_df", par)
con.execute("CREATE TABLE Partition AS SELECT * FROM par_df")
con.unregister("par_df")
del att, par
con.execute("CHECKPOINT")
meta["t_load_tables_s"] = clk() - t0

# ---------------------------------------------------------------- producer: release
t0 = clk()
con.execute("""CREATE TABLE P AS
  WITH z AS (SELECT unit_id, COUNT(*) AS k FROM Incidence GROUP BY unit_id)
  SELECT a.node_id AS u, b.node_id AS v, SUM(1.0::DOUBLE / z.k) AS w
  FROM Incidence a JOIN Incidence b ON a.unit_id = b.unit_id AND a.node_id <= b.node_id
       JOIN z ON z.unit_id = a.unit_id
  GROUP BY 1, 2""")
con.execute("CHECKPOINT")
meta["t_build_P_s"] = clk() - t0
nP = con.execute("SELECT COUNT(*), SUM(CASE WHEN u=v THEN 1 ELSE 0 END) FROM P").fetchone()
meta["P_rows_upper"] = int(nP[0]); meta["P_rows_diag"] = int(nP[1])
meta["P_entries_ordered"] = int(2 * nP[0] - nP[1])
t0 = clk()
con.execute("CREATE TABLE Deg AS SELECT node_id AS u, COUNT(*)::INTEGER AS d FROM Incidence GROUP BY 1")
meta["t_build_deg_s"] = clk() - t0
for k in KS:
    t0 = clk()
    con.execute(f"""CREATE TABLE T{k} AS
      SELECT r.u, r.v, r.w / d.d AS t FROM
        (SELECT u, v, w FROM P UNION ALL SELECT v AS u, u AS v, w FROM P WHERE u <> v) r
        JOIN Deg d ON d.u = r.u
      QUALIFY row_number() OVER (PARTITION BY r.u ORDER BY r.w DESC, r.v) <= {k}""")
    con.execute("CHECKPOINT")
    meta[f"t_build_T{k}_from_P_s"] = clk() - t0
    meta[f"T{k}_rows"] = int(con.execute(f"SELECT COUNT(*) FROM T{k}").fetchone()[0])
    # auditor preprocessing (once per release, attribute-free): symmetric closure with exact-entry recovery
    t0 = clk()
    con.execute(f"""CREATE TABLE S{k} AS
      SELECT LEAST(t.u, t.v) AS a, GREATEST(t.u, t.v) AS b, MAX(t.t * d.d) AS w
      FROM T{k} t JOIN Deg d ON d.u = t.u GROUP BY 1, 2""")
    meta[f"t_aud_prep_S{k}_s"] = clk() - t0
    meta[f"S{k}_rows"] = int(con.execute(f"SELECT COUNT(*) FROM S{k}").fetchone()[0])
    meta[f"T{k}_mass_share"] = float(con.execute(
        f"SELECT SUM(CASE WHEN a=b THEN w ELSE 2*w END) FROM S{k}").fetchone()[0]) / vol
vol_P = float(con.execute("SELECT SUM(CASE WHEN u=v THEN w ELSE 2*w END) FROM P").fetchone()[0])
vol_T = int(con.execute("SELECT SUM(d) FROM Deg").fetchone()[0])
meta["vol_from_P"] = vol_P
meta["vol_from_deg"] = vol_T
assert abs(vol_P - vol) < 1e-6 * vol and vol_T == vol

# bytes (fixed-width binary rows: int32 ids, float64 weights, int32 codes, int64 aggregates)
meta["bytes_incidence"] = vol * 8
meta["bytes_attr_col"] = n * 4
meta["bytes_part_col"] = n * 4
meta["bytes_release_P"] = meta["P_rows_upper"] * 16
for k in KS:
    meta[f"bytes_release_T{k}_deg"] = meta[f"T{k}_rows"] * 16 + n * 4
meta["bytes_aggregates_per_pair"] = nsizes * 16
say("release:", {k: v for k, v in meta.items() if k.startswith(("t_", "P_", "T", "S", "bytes"))})

# ---------------------------------------------------------------- SQL
SQL_L = """CREATE OR REPLACE TEMP TABLE L AS SELECT a.node_id AS node_id, a.g AS g, p.c AS c
  FROM Attribute a JOIN Partition p ON p.node_id = a.node_id WHERE a.attr_id = $a AND p.part_id = $p"""
# the crossing test sits in a CASE, not in WHERE: with a WHERE predicate the optimizer may join L with L
# on the inequalities first (a near cross product when the partition has many clusters).
SQL_UP = """SELECT SUM(CASE WHEN lu.g <> lv.g AND lu.c <> lv.c THEN P.w ELSE 0 END)
FROM P JOIN L lu ON lu.node_id = P.u JOIN L lv ON lv.node_id = P.v"""
SQL_UT = """SELECT SUM(CASE WHEN lu.g <> lv.g AND lu.c <> lv.c THEN 0 WHEN S.a = S.b THEN S.w ELSE 2 * S.w END)
FROM S{k} S JOIN L lu ON lu.node_id = S.a JOIN L lv ON lv.node_id = S.b"""
SQL_EX = """WITH s AS MATERIALIZED (SELECT i.unit_id AS e, L.g AS g, L.c AS c, COUNT(*) AS n
      FROM Incidence i JOIN L ON L.node_id = i.node_id GROUP BY 1, 2, 3),
u AS (SELECT e, SUM(n) AS k FROM s GROUP BY e),
gm AS (SELECT e, g, SUM(n) AS ng FROM s GROUP BY e, g),
cm AS (SELECT e, c, SUM(n) AS nc FROM s GROUP BY e, c)
SELECT u.k, COUNT(DISTINCT s.e), SUM(CASE WHEN u.k*s.n > gm.ng*cm.nc THEN u.k*s.n - gm.ng*cm.nc ELSE 0 END)
FROM s JOIN u USING(e) JOIN gm USING(e, g) JOIN cm USING(e, c) GROUP BY u.k"""
_curL = [None]


def make_L(pid, ai):
    """Label table of one (attribute, partition) pair: built by whoever evaluates the pair (both plans)."""
    con.execute(SQL_L, {"a": ai, "p": pid})
    _curL[0] = (pid, ai)


def run_up(pid, ai):
    make_L(pid, ai)
    return float(con.execute(SQL_UP).fetchone()[0] or 0.0) * 2.0 / vol_P


def run_ut(pid, ai, k):
    if _curL[0] != (pid, ai):
        make_L(pid, ai)
    ml = float(con.execute(SQL_UT.replace("{k}", str(k))).fetchone()[0] or 0.0)
    return 1.0 - ml * (1.0 - 1e-12) / vol_T


def run_ex(pid, ai, reuse=False):
    if not (reuse and _curL[0] == (pid, ai)):
        make_L(pid, ai)
    rows = sorted((int(k), int(d), int(u)) for k, u, d in con.execute(SQL_EX).fetchall())
    acc = sum((Fraction(d, k) for k, d, _ in rows), Fraction(0))
    return acc / sum(k * u for k, _, u in rows), rows


def build_lab(pids, ais):
    df = pd.DataFrame({"node_id": np.arange(n, dtype=np.int32)})
    for ai in ais:
        df[f"g{ai}"] = G[attrs[ai]].astype(np.int32)
    for pid in pids:
        df[f"c{pid}"] = P_list[pid][4].astype(np.int32)
    con.execute("DROP TABLE IF EXISTS Lab")
    con.register("lab_df", df)
    con.execute("CREATE TABLE Lab AS SELECT * FROM lab_df")
    con.unregister("lab_df")


def run_fused(plist):
    """One scan of P joined twice with a wide label table answers U_P for every pair in plist."""
    pids = sorted({p for p, _ in plist}); ais = sorted({a for _, a in plist})
    build_lab(pids, ais)
    aggs = ",\n".join(f"SUM(CASE WHEN lu.g{a}<>lv.g{a} AND lu.c{p}<>lv.c{p} THEN P.w ELSE 0 END)"
                      for p, a in plist)
    r = con.execute(f"SELECT {aggs} FROM P JOIN Lab lu ON lu.node_id=P.u JOIN Lab lv ON lv.node_id=P.v"
                    ).fetchone()
    return [float(x or 0.0) * 2.0 / vol_P for x in r]


def run_ex_shared_build(pids, ais):
    build_lab(pids, ais)
    con.execute("DROP TABLE IF EXISTS J")
    con.execute("CREATE TABLE J AS SELECT i.unit_id AS e, Lab.* EXCLUDE(node_id) FROM Incidence i "
                "JOIN Lab ON Lab.node_id = i.node_id")


def run_ex_shared(pid, ai):
    q = f"""WITH s AS MATERIALIZED (SELECT e, g{ai} AS g, c{pid} AS c, COUNT(*) AS n FROM J GROUP BY 1,2,3),
u AS (SELECT e, SUM(n) AS k FROM s GROUP BY e),
gm AS (SELECT e, g, SUM(n) AS ng FROM s GROUP BY e, g),
cm AS (SELECT e, c, SUM(n) AS nc FROM s GROUP BY e, c)
SELECT u.k, COUNT(DISTINCT s.e), SUM(CASE WHEN u.k*s.n > gm.ng*cm.nc THEN u.k*s.n - gm.ng*cm.nc ELSE 0 END)
FROM s JOIN u USING(e) JOIN gm USING(e, g) JOIN cm USING(e, c) GROUP BY u.k"""
    rows = sorted((int(k), int(d), int(u)) for k, u, d in con.execute(q).fetchall())
    acc = sum((Fraction(d, k) for k, d, _ in rows), Fraction(0))
    return acc / sum(k * u for k, _, u in rows)


# ---------------------------------------------------------------- per-pair pass (all pairs)
# every pair: U_P scan, T_k bound scans, exact; numpy reference; verdicts for every budget.
pair_info = {}
for idx, (pid, ai) in enumerate(pairs):
    _, meth, t, seed, cl = P_list[pid]
    a = attrs[ai]
    t0 = clk(); up = run_up(pid, ai); t_up = clk() - t0
    uts, t_uts = {}, {}
    for k in KS:
        t0 = clk(); uts[k] = run_ut(pid, ai, k); t_uts[k] = clk() - t0
    t0 = clk(); hgp, rows = run_ex(pid, ai); t_ex = clk() - t0
    ref = C.exact_bysize(H, cl, G[a])
    assert [(k, y, u) for k, y, _, u in ref] == rows, "SQL per-size rows differ from numpy reference"
    hgp_ref = C.frac_from_bysize(ref, 1)
    up_ref = C.frac_from_bysize(ref, 2)
    assert hgp_ref == hgp
    assert up_ref >= hgp, "Prop. 2(a) violated"
    up_err = abs(up - float(up_ref))
    for k in KS:
        assert uts[k] >= float(up_ref) - 1e-12, ("pruned bound below U_P", k)
    if t == 0.0:
        assert abs(float(hgp) - ref_hgp[(meth, a)]) < 1e-9, ("HGP differs from panel", meth, a)
        assert abs(up - ref_up[(meth, a)]) < 1e-6, ("U_P differs from panel", meth, a, up, ref_up[(meth, a)])
    row = {"dataset": ds, "pair_idx": idx, "part_id": pid, "method": meth, "t": t, "seed": seed,
           "attribute": a, "is_base": int(t == 0.0), "B": Bof[a], "HGP": float(hgp),
           "HGP_exact": f"{hgp.numerator}/{hgp.denominator}" if hgp.denominator < 10 ** 40 else "",
           "U_P": up, "U_P_abs_err_vs_exact": up_err, "t_scan_P_s": t_up, "t_exact_s": t_ex}
    for k in KS:
        row[f"U_T{k}"] = uts[k]; row[f"t_scan_T{k}_s"] = t_uts[k]
    open_any = False
    for fi, f in enumerate(C.FACTORS):
        tau = Bof[a] * f
        pass_ex = hgp <= Fraction(tau)
        cert = up <= tau - C.MARGIN
        assert (not cert) or pass_ex, ("FALSE CERTIFICATE", pid, a, f)
        certT = {k: uts[k] <= tau - C.MARGIN for k in KS}
        for k in KS:
            assert (not certT[k]) or pass_ex, ("FALSE CERTIFICATE (T)", pid, a, f, k)
        verdict_a = True if cert else pass_ex
        assert verdict_a == pass_ex
        open_any |= not cert
        arow = {"dataset": ds, "pair_idx": idx, "audit_idx": idx * len(C.FACTORS) + fi, "factor": f,
                "budget": tau, "is_base": int(t == 0.0), "HGP": float(hgp), "U_P": up,
                "pass_exact": int(pass_ex), "cert_P": int(cert), "verdict_a": int(verdict_a),
                "verdict_b": int(pass_ex)}
        for k in KS:
            arow[f"cert_T{k}"] = int(certT[k])
        C.append_csv(OUT / f"{ds}_audits.csv", arow)
    row["open_P"] = int(open_any)
    C.append_csv(OUT / f"{ds}_pairs.csv", row)
    pair_info[(pid, ai)] = row
    if idx % 20 == 0 or idx < nbase:
        say(f"[{idx+1}/{len(pairs)}] {meth} t={t} s={seed} {a}: HGP={float(hgp):.6f} U_P={up:.6f} "
            f"U_T{KS[0]}={uts[KS[0]]:.4f} scan={t_up*1e3:.1f}ms exact={t_ex*1e3:.1f}ms")

# ---------------------------------------------------------------- base batch, timed end to end
basepairs = pairs[:nbase]
open_base = [pa for pa in basepairs if pair_info[pa]["open_P"]]
say(f"base: {len(basepairs)} pairs, {len(open_base)} open under U_P")
for rep in range(args.reps):
    res = {"dataset": ds, "rep": rep}
    # (a) per-pair scans + exact for open pairs
    t0 = clk()
    opened = []
    for pid, ai in basepairs:
        up = run_up(pid, ai)
        tau_min = Bof[attrs[ai]] * min(C.FACTORS)
        if not up <= tau_min - C.MARGIN:
            opened.append((pid, ai))
    t_a_scan = clk() - t0
    for pid, ai in opened:
        run_ex(pid, ai)
    res["a_scan_s"] = t_a_scan; res["a_total_s"] = clk() - t0; res["a_opened"] = len(opened)
    # (a') fused single scan + exact for open pairs
    t0 = clk()
    ups = run_fused(basepairs)
    t_f = clk() - t0
    opened_f = [pa for pa, up in zip(basepairs, ups) if not up <= Bof[attrs[pa[1]]] * min(C.FACTORS) - C.MARGIN]
    assert opened_f == opened
    for pid, ai in opened_f:
        run_ex(pid, ai)
    res["af_scan_s"] = t_f; res["af_total_s"] = clk() - t0
    # (a, T_k) per-pair pruned-bound scans + exact for open pairs
    for k in KS:
        t0 = clk(); op = []
        for pid, ai in basepairs:
            make_L(pid, ai)
            if not run_ut(pid, ai, k) <= Bof[attrs[ai]] * min(C.FACTORS) - C.MARGIN:
                op.append((pid, ai))
        res[f"aT{k}_scan_s"] = clk() - t0
        for pid, ai in op:
            run_ex(pid, ai)
        res[f"aT{k}_total_s"] = clk() - t0; res[f"aT{k}_opened"] = len(op)
    # (b) exact everywhere
    t0 = clk()
    for pid, ai in basepairs:
        run_ex(pid, ai)
    res["b_total_s"] = clk() - t0
    # (b') exact everywhere with one shared incidence-label join
    t0 = clk()
    run_ex_shared_build(sorted({p for p, _ in basepairs}), sorted({a for _, a in basepairs}))
    tj = clk() - t0
    for pid, ai in basepairs:
        h = run_ex_shared(pid, ai)
        assert float(h) == pair_info[(pid, ai)]["HGP"]
    res["bs_join_s"] = tj; res["bs_total_s"] = clk() - t0
    C.append_csv(OUT / f"{ds}_batch.csv", res)
    say("batch rep", rep, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()})

# ---------------------------------------------------------------- growth: fused scan at prefix sizes
if not args.no_growth:
    nf = len(C.FACTORS)
    for N in (1, 10, 100, 1000, len(pairs) * nf):
        if N > len(pairs) * nf:
            continue
        npairs = math.ceil(N / nf)
        plist = pairs[:npairs]
        ts = []
        for rep in range(args.reps):
            t0 = clk(); ups = run_fused(plist); ts.append(clk() - t0)
        for pa, up in zip(plist, ups):
            assert abs(up - pair_info[pa]["U_P"]) < 1e-9
        C.append_csv(OUT / f"{ds}_growth.csv", {"dataset": ds, "N": N, "pairs": npairs,
                                               "t_fused_scan_s": float(np.median(ts))})
        say("growth fused", N, npairs, round(float(np.median(ts)), 4))

C.write_json(OUT / f"{ds}_meta.json", meta)
con.close()
say("done", ds)
