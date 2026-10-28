"""Audit one cell (method, dataset, attribute): every auditor, every certificate, and certify-then-request.

Examples:
  python run_audit.py --dataset OA-DB --method AHRC --attribute country
  python run_audit.py --dataset ML-1M --method CAHC --attribute gender --budget 0.19
  python run_audit.py --dataset Pokec --method GRAC --attribute AGE --tau 0.6
Without --budget, the shared budget B of the paper is replayed (about 10 s to 2 min, depending on the dataset).
The exact HGP is also evaluated with the SQL query on SQLite (and DuckDB with --duckdb).
"""
from __future__ import annotations

import time
from fractions import Fraction

from ahc_audit import data as dmod
from ahc_audit import sql
from ahc_audit.budgets import shared_budget_stream
from ahc_audit.cells import Context
from ahc_audit.certificates import nc_constant
from options import add_cell, base_parser


def main():
    p = base_parser(__doc__.splitlines()[0])
    add_cell(p)
    p.add_argument("--duckdb", action="store_true", help="also run the exact query on DuckDB")
    a = p.parse_args()
    ds = a.dataset
    if a.attribute not in dmod.ATTRIBUTES[ds]:
        raise SystemExit(f"attribute must be one of {dmod.ATTRIBUTES[ds]}")
    D = dmod.load_dataset(ds, a.data_root, a.partitions)
    ctx = Context(D, a.out / "reconstruction")
    t0 = time.perf_counter()
    r = ctx.cell(a.method, a.attribute)
    print(f"{D.display} / {a.method} / {a.attribute}: n={D.n} units={D.m} memberships={D.vol} K={r['K']} "
          f"({time.perf_counter() - t0:.2f}s)")
    if a.budget is None:
        cache = {}

        def load(name):
            if name not in cache:
                cache.clear(); cache[name] = D if name == ds else dmod.load_dataset(name, a.data_root, a.partitions)
            return cache[name]
        print("replaying the shared budget ...")
        rows = list(shared_budget_stream(load, lambda X, attr: Context(X, a.out / "reconstruction", with_P=False)
                                         .reference_stats(attr), datasets=[ds],
                                         include_screened=(ds, a.attribute) in dmod.SCREENED, log=lambda m: None))
        b = [x for x in rows if x["attribute"] == a.attribute][0]
        B = b["B"]
        print(f"  shared B = {B:.6f} (95% interval {b['B_CI_lo']:.6f} to {b['B_CI_hi']:.6f}); self-calibrated "
              f"dyadic {b['B_sur']:.6f}, normalized clique {b['B_ncl']:.6f}")
    else:
        B = a.budget
    Bx = Fraction(B)
    h = Fraction(r["HGP_exact"])
    print(f"\nHGP_true = {r['HGP']:.6f}  ->  {'PASS' if h <= Bx else 'FAIL'} at budget {B:.6f}")
    rows = [("dyadic surrogate (unweighted clique)", Fraction(r["HGP_sur_exact"]), "auditor"),
            ("normalized-clique surrogate", Fraction(r["U_Q_exact"]) / 2, "auditor")]
    if r["HGP_rec_exact"]:
        rows.append(("maximal-clique reconstruction", Fraction(r["HGP_rec_exact"]), "auditor"))
    rows += [("U_P (random-walk release)", Fraction(r["U_P_exact"]), "certificate"),
             ("U_Q (normalized clique)", Fraction(r["U_Q_exact"]), "certificate"),
             ("U_Q,50 (declared max size 50)", Fraction(r["U_Q_exact"]) * Fraction(49, 50), "certificate")]
    for name, v, kind in rows:
        ok = v <= Bx
        if kind == "auditor":
            verdict = "pass" if ok else "fail"
            err = "" if ok == (h <= Bx) else ("  <- MISS" if ok else "  <- FALSE ALARM")
        else:
            verdict = "certified" if ok else "undetermined -> request per-size counts and sums"
            err = "  <- FALSE CERTIFICATE" if ok and h > Bx else ""
        print(f"  {name:38s} {float(v):.6f}  {verdict}{err}")
    if a.tau is not None:
        t = Fraction(a.tau)
        db = Fraction(r["Dbar_exact"])
        print(f"\nDbar = {float(db):.6f}  ->  {'PASS' if db <= t else 'FAIL'} at tau {a.tau}")
        for name, v in (("dyadic surrogate", Fraction(r["D_sur_exact"])), ("normalized-clique surrogate", Fraction(r["D_ncl_exact"])),
                        ("U_P / Ibar (certificate)", Fraction(r["D_UP_exact"])),
                        ("c W_x/W_xi, declared sizes (certificate)", nc_constant(int(r["rmin_declared"]), 50) * Fraction(r["D_ncl_exact"]))):
            print(f"  {name:42s} {float(v):.6f}  {'<= tau' if v <= t else '> tau'}")
    print("\nexact evaluation with the SQL query:")
    for engine in ["sqlite"] + (["duckdb"] if a.duckdb else []):
        t0 = time.perf_counter()
        rows = sql.per_size_counts_and_sums(engine, D.H, D.attrs[a.attribute], D.partition(a.method))
        hv = sql.hgp_from_rows(rows)
        print(f"  {engine}: {len(rows)} unit sizes, HGP = {float(hv):.6f}, equal to the exact value: {hv == h} "
              f"({time.perf_counter() - t0:.2f}s)")


if __name__ == "__main__":
    main()
