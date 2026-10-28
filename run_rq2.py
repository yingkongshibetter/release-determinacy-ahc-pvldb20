"""RQ2 (certification from a held release): every RQ2 number, Table 7, and the request counts of Practicality.

Needs outputs/cells.csv (run_cells.py; with --screened for the screened rows of Table 7).
Steps with their own cache under outputs/rq2/:
  inexact   certificates on rounded and pruned releases (Prop. 5)            about 1 min
  lp        linear relaxation of Prop. 4 on 540 complete sets of compatible inputs on five to seven nodes
            (exact rational simplex)                                       about 1 min
Writes outputs/table7.csv and outputs/rq2.txt.
"""
from __future__ import annotations

import time
from fractions import Fraction

import numpy as np
import pandas as pd

from ahc_audit import data as dmod
from ahc_audit import small_instances as SI
from ahc_audit.budgets import TAUS
from ahc_audit.certificates import nc_constant
from ahc_audit.inexact import Releases
from ahc_audit.verdicts import Cells, no_false_certificate
from options import base_parser

LINES = []


def say(label, value=""):
    s = f"{label}: {value}" if value != "" else label
    print(s, flush=True)
    LINES.append(s)


def cached(path, fn, force=False):
    if path.exists() and not force:
        return pd.read_csv(path)
    t0 = time.perf_counter()
    df = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  wrote {path} ({time.perf_counter() - t0:.0f}s)", flush=True)
    return df


def inexact_step(a, cells):
    rows = []
    for ds in dmod.DATASETS:
        sub = cells[cells.ds == ds]
        if sub.empty:
            continue
        D = dmod.load_dataset(ds, a.data_root, a.partitions)
        t0 = time.perf_counter()
        R = Releases(D.H)
        for _, c in sub.iterrows():
            r = R.cell(D.partition(c.method), D.group(c.attribute), Fraction(float(c.B)), Fraction(c.HGP_exact),
                       Fraction(c.U_P_exact), Fraction(c.U_Q_exact))
            rows.append(dict(method=c.method, dataset=c.dataset, attribute=c.attribute, **r))
        print(f"  inexact {D.display} ({time.perf_counter() - t0:.0f}s); top-50 rows keep "
              f"{100 * R.top[50]['kept_share']:.1f}% of the mass", flush=True)
    return pd.DataFrame(rows)


def lp_step():
    rows = []
    for name in SI.CONFIGS:
        _, r = SI.lp_check(name)
        rows.extend(r)
    return pd.DataFrame(rows)


def coverage(C, mask):
    idx = [i for i in range(C.n) if mask[i]]
    p = [i for i in idx if C.h[i] <= C.B[i]]
    return (len(idx), len(idx) - len(p), sum(C.up[i] <= C.B[i] for i in p), sum(C.uq50[i] <= C.B[i] for i in p), len(p))


def main():
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--skip", nargs="*", default=[], choices=["inexact", "lp"])
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    out = a.out
    allc = pd.read_csv(out / "cells.csv")
    scr = allc.apply(lambda r: (r.ds, r.attribute) in dmod.SCREENED, axis=1)
    cells = allc[~scr].reset_index(drop=True)
    C = Cells(cells)
    rq = out / "rq2"
    d = cells.dataset
    passB = [C.h[i] <= C.B[i] for i in range(C.n)]
    P = [i for i in range(C.n) if passB[i]]

    # ---------------- RQ2 bullet 1
    say("RQ2-1 U_P certifies passing cells at B", f"{sum(C.up[i] <= C.B[i] for i in P)} of {len(P)}")
    say("RQ2-1 U_Q,50 / U_Q certify passing cells at B",
        f"{sum(C.uq50[i] <= C.B[i] for i in P)} / {sum(C.uq[i] <= C.B[i] for i in P)} of {len(P)}")
    Po = [i for i in P if cells.method[i] != "CE+Fair-Louvain"]
    say("RQ2-1 U_P outside CE+Fair-Louvain", f"{sum(C.up[i] <= C.B[i] for i in Po)} of {len(Po)}")
    for tau in ("0.3", "0.6"):
        t = Fraction(tau)
        pt = [i for i in range(C.n) if C.dbar[i] <= t]
        up = sum(C.d_up[i] <= t for i in pt)
        ncs = []
        for lab, fn in (("no declaration (c = 2)", lambda i: nc_constant()),
                        ("max size 50 (c = 1.96)", lambda i: nc_constant(None, 50)),
                        ("declared sizes (c = 1.47; 1.96 on ACS-PUMS-HH)", lambda i: nc_constant(int(cells.rmin_declared[i]), 50))):
            ncs.append(f"{lab} {sum(fn(i) * C.d_ncl[i] <= t for i in pt)}")
        say(f"RQ2-1 tau = {tau}: U_P/Ibar certifies {up} of {len(pt)} passing; normalized clique", "; ".join(ncs))
    ok, bad = no_false_certificate(C)
    say("RQ2-1 no certificate of Props. 2 and 3 certifies a failing cell at any budget", f"{ok} ({bad} violations)")

    # ---------------- RQ2 bullet 2: requests
    t01 = Fraction(1, 10)
    unc = [sum(C.h[i] <= C.B[i] and not C.up[i] <= C.B[i] for i in range(C.n)),
           sum(C.h[i] <= t01 and not C.up[i] <= t01 for i in range(C.n))]
    for tau in ("0.3", "0.6"):
        t = Fraction(tau)
        unc.append(sum(C.dbar[i] <= t and not C.d_up[i] <= t for i in range(C.n)))
    say("RQ2-2 passing cells that U_P leaves uncertified at B, 0.1, tau = 0.3, tau = 0.6", unc)
    U = [i for i in P if not C.up[i] <= C.B[i]]
    Cc = [i for i in P if C.up[i] <= C.B[i]]
    mu = [float((C.B[i] - C.h[i]) / C.B[i]) for i in U]
    mc = [float((C.B[i] - C.h[i]) / C.B[i]) for i in Cc]
    say("RQ2-2 uncertified passing cells at B: HGP below B by", f"{100 * min(mu):.0f}% to {100 * max(mu):.0f}%")
    say("RQ2-2 median margin of the certified passing cells", f"{100 * np.median(mc):.0f}%")

    # ---------------- RQ2 bullet 3: tightness
    r = np.array([float(C.up[i] / C.h[i]) for i in range(C.n)])
    r50 = np.array([float(C.uq50[i] / C.h[i]) for i in range(C.n)])
    say("RQ2-3 median U_P / HGP and U_Q,50 / HGP", f"{np.median(r):.2f} / {np.median(r50):.2f} (U_P/HGP p90 {np.quantile(r, 0.9):.2f})")
    if "lp" not in a.skip:
        lp = cached(rq / "lp_small_instances.csv", lp_step, a.force)
        eq = lp.apply(lambda x: Fraction(x.U_LP) == Fraction(x.h_max), axis=1)
        chain = lp.apply(lambda x: Fraction(x.h_min) <= Fraction(x.h_max) <= Fraction(x.U_LP) <= Fraction(x.U_cert), axis=1)
        isP = lp.release == "P"
        say("RQ2-3 linear relaxation equals h_max", f"{int(eq.sum())} of {len(lp)} complete sets, including "
            f"{int(eq[isP].sum())} of {int(isP.sum())} for P; h_max <= U_LP <= certificate on {int(chain.sum())}")
        say("RQ2-3 median time per set: linear relaxation / certificate",
            f"{1e3 * lp.t_lp_s.median():.1f} ms / {1e6 * lp.t_cert_s.median():.0f} us")

    # ---------------- RQ2 bullet 4: rounded and pruned releases
    if "inexact" not in a.skip:
        iv = cached(rq / "inexact.csv", lambda: inexact_step(a, cells), a.force)
        ip = iv[iv.pass_B.astype(bool)]
        ifl = iv[~iv.pass_B.astype(bool)]
        n = len(ip)
        cnt = lambda col: f"{int(ip[col].astype(bool).sum())} of {n}"
        say("RQ2-4 exact release: U_Q / U_Q,50 / U_P", f"{cnt('cert_U_Q')} / {cnt('cert_U_Q50')} / {cnt('cert_U_P')}")
        say("RQ2-4 normalized clique, 6 significant digits: U_Q / U_Q,50", f"{cnt('Q-sig6_UQ_cert')} / {cnt('Q-sig6_UQ50_cert')}")
        say("RQ2-4 normalized clique, 3 decimals: U_Q / U_Q,50", f"{cnt('Q-3dp_UQ_cert')} / {cnt('Q-3dp_UQ50_cert')}")
        say("RQ2-4 U_P on T in double precision with degrees", cnt("P-f64_UP_cert"))
        say("RQ2-4 U_P, T rounded to 4 decimals, zero entries dropped, degrees (Prop. 5(b))", cnt("T-4dp_UP_cert"))
        say("RQ2-4 U_P, rows pruned to 200 / 50 largest entries, degrees (Prop. 5(b))",
            f"{cnt('top200_UP_cert')} / {cnt('top50_UP_cert')}")
        sound = [c for c in iv.columns if c.endswith("_cert") and "plugin" not in c]
        say("RQ2-4 failing cells certified by any sound certificate", int(ifl[sound].astype(bool).any(axis=1).sum()))
        pl = ifl[ifl.top50_plugin_cert.astype(bool)]
        say("RQ2-4 failing cells certified by the crossing share among kept top-50 entries (not sound)",
            f"{len(pl)} ({sorted(set(pl.dataset))})")

    # ---------------- Practicality: pairs that need no request
    pairs = cells.groupby(["dataset", "attribute"]).groups
    noreq_B = sum(all(C.up[i] <= C.B[i] for i in idx) for idx in pairs.values())
    t6 = Fraction(3, 5)
    noreq_t = sum(all(C.d_up[i] <= t6 for i in idx) for idx in pairs.values())
    say("Practicality dataset-attribute pairs that need no request under B / under tau = 0.6",
        f"{noreq_B} / {noreq_t} of {len(pairs)}")

    # ---------------- Table 7 (Appendix B)
    rows = []

    def add(label, mask, frame=C):
        n, f, up, u50, np_ = coverage(frame, mask)
        rows.append(dict(set=label, cells=n, fail=f, UP_certifies=f"{up} of {np_}", UQ50_certifies=f"{u50} of {np_}"))
    add("All 75 cells", [True] * C.n)
    init = (cells.dataset == "IMDb-Cast") & (cells.method == "CAHC")
    add("Without CAHC-init", (~init).tolist())
    if scr.any():
        S = Cells(allc[scr].reset_index(drop=True))
        add("Screened race and disability", [True] * S.n, S)
        sdf = allc[scr]
        say("Appendix B screened cells: dyadic / normalized-clique errors at B",
            f"{sum((Fraction(h) <= Fraction(float(b))) != (Fraction(s) <= Fraction(float(b))) for h, s, b in zip(sdf.HGP_exact, sdf.HGP_sur_exact, sdf.B))} / "
            f"{sum((Fraction(h) <= Fraction(float(b))) != (Fraction(q) / 2 <= Fraction(float(b))) for h, q, b in zip(sdf.HGP_exact, sdf.U_Q_exact, sdf.B))}; "
            f"median U_P/HGP {np.median([float(Fraction(u) / Fraction(h)) for u, h in zip(sdf.U_P_exact, sdf.HGP_exact)]):.3f}")
    for meth in dmod.METHODS:
        add(meth, (cells.method == meth).tolist())
    t7 = pd.DataFrame(rows)
    t7.to_csv(out / "table7.csv", index=False)
    say("Table 7", "\n" + t7.to_string(index=False))
    without = cells[~init]
    W = Cells(without)
    f = [i for i in range(W.n) if W.h[i] > W.B[i]]
    say("Appendix B without CAHC-init: dyadic / normalized-clique misses, requests at B",
        f"{sum(W.sur[i] <= W.B[i] for i in f)} / {sum(W.nc[i] <= W.B[i] for i in f)} of {len(f)}, "
        f"{sum(not W.up[i] <= W.B[i] for i in range(W.n))}")

    (out / "rq2.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"wrote {out / 'rq2.txt'}")


if __name__ == "__main__":
    main()
