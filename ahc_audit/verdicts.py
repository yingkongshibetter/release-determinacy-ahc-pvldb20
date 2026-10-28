"""Verdicts of every auditor and certificate on the cells, under every budget of Table 4.

Input: the table written by run_cells.py (outputs/cells.csv). Every comparison is exact: statistics are
read from their rational strings and float budgets are converted exactly to Fractions.
Auditors: Dy = dyadic surrogate, NC = normalized-clique surrogate (U_Q/2), Rec = reconstruction auditor
(not run on ML-1M, whose enumeration exceeds the cap). Certificates: U_P, U_Q, U_Q,50, and on the scale of
Dbar U_P/Ibar and c * W_x/W_xi (Prop. 3(b)).
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd

from .budgets import FACTORS, TAUS
from .certificates import nc_constant
from .data import DISPLAY, OVERLAPPING

DS_ORDER = [DISPLAY[d] for d in ["movielens_1m", "openalex_db", "openalex_cs_size3plus", "pokec_hobby", "imdb_cast",
                                 "acs_pums_household"]]
OVERLAP = [DISPLAY[d] for d in OVERLAPPING]


def F(x):
    if isinstance(x, Fraction):
        return x
    if isinstance(x, str):
        return Fraction(x)
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return Fraction(float(x))


def le(a, b):
    return None if a is None or b is None else a <= b


class Cells:
    """Exact statistics of the cells, with helper predicates."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        c = self.df
        self.h = [F(x) for x in c.HGP_exact]
        self.sur = [F(x) for x in c.HGP_sur_exact]
        self.uq = [F(x) for x in c.U_Q_exact]
        self.nc = [q / 2 for q in self.uq]
        self.uq50 = [q * Fraction(49, 50) for q in self.uq]
        self.up = [F(x) for x in c.U_P_exact]
        self.rec = [F(x) if isinstance(x, str) and x else None for x in c.HGP_rec_exact.fillna("")]
        self.B = [F(x) for x in c.B]
        self.dbar = [F(x) for x in c.Dbar_exact]
        self.d_sur = [F(x) for x in c.D_sur_exact]
        self.d_ncl = [F(x) for x in c.D_ncl_exact]
        self.d_up = [F(x) for x in c.D_UP_exact]
        self.d_rec = [F(x) if isinstance(x, str) and x else None for x in c.D_rec_exact.fillna("")]
        self.d_cgp = [F(x) for x in c.D_CGP_exact]
        self.cgp = [F(x) for x in c.CGP_exact]
        self.B_sur = [F(x) for x in c.B_sur]
        self.B_ncl = [F(x) for x in c.B_ncl]
        self.B_rec = [F(x) for x in c.B_rec] if "B_rec" in c else [None] * len(c)
        self.B_pm = [F(x) for x in c.B_pm] if "B_pm" in c else None
        self.n = len(c)

    def subset(self, mask):
        return Cells(self.df[np.asarray(mask)])

    @property
    def dataset(self):
        return self.df.dataset.tolist()


def row_counts(C: Cells, stat, budget, label):
    """Misses and false alarms of the pairwise auditors and U_P at one budget (a list of thresholds per cell)."""
    p = [h <= b for h, b in zip(stat["true"], budget)]
    out = dict(budget=label, cells=C.n, fail=sum(not x for x in p))
    for k in ("Dy", "NC", "Rec", "U_P"):
        s = stat.get(k)
        if s is None:
            out[f"miss_{k}"] = None; out[f"fa_{k}"] = None
            continue
        ev = [i for i in range(C.n) if s[i] is not None and budget[i] is not None]
        fails = [i for i in ev if not p[i]]
        passes = [i for i in ev if p[i]]
        out[f"miss_{k}"] = sum(s[i] <= budget[i] for i in fails) if fails else None
        out[f"fa_{k}"] = sum(s[i] > budget[i] for i in passes) if (passes and k != "U_P") else None
        out[f"eval_{k}"] = len(ev)
    if stat.get("U_P") is not None:
        passes = [i for i in range(C.n) if p[i]]
        cert = sum(stat["U_P"][i] <= budget[i] for i in passes)
        out["UP_certifies"] = f"{cert} of {len(passes)}"
        out["UP_cert_n"] = cert
        out["requests"] = sum(not (stat["U_P"][i] <= budget[i]) for i in range(C.n))
    return out


def table6(C: Cells):
    rows = []
    base = dict(true=C.h, Dy=C.sur, NC=C.nc, Rec=C.rec, U_P=C.up)
    rows.append(row_counts(C, base, C.B, "Shared B"))
    # self-calibrated: each auditor against its own budget; the true verdict stays at B
    sc = {}
    p = [h <= b for h, b in zip(C.h, C.B)]
    sc_row = dict(budget="Self-calibrated", cells=C.n, fail=sum(not x for x in p))
    for k, s, b in (("Dy", C.sur, C.B_sur), ("NC", C.nc, C.B_ncl), ("Rec", C.rec, C.B_rec)):
        ev = [i for i in range(C.n) if s[i] is not None and b[i] is not None]
        fails = [i for i in ev if not p[i]]; passes = [i for i in ev if p[i]]
        sc_row[f"miss_{k}"] = sum(s[i] <= b[i] for i in fails) if fails else None
        sc_row[f"fa_{k}"] = sum(s[i] > b[i] for i in passes) if passes else None
        sc_row[f"eval_{k}"] = len(ev)
    sc_row["miss_U_P"] = None
    rows.append(sc_row)
    t01 = [Fraction(1, 10)] * C.n
    rows.append(row_counts(C, base, t01, "Fixed 0.1"))
    for tau in ("0.3", "0.6"):
        t = [Fraction(tau)] * C.n
        dstat = dict(true=C.dbar, Dy=C.d_sur, NC=C.d_ncl, Rec=C.d_rec, U_P=C.d_up)
        rows.append(row_counts(C, dstat, t, f"tau = {tau} on Dbar"))
    return pd.DataFrame(rows)


def table5(C: Cells):
    rows = []
    d = C.df
    for ds in DS_ORDER + ["All"]:
        idx = list(range(C.n)) if ds == "All" else [i for i in range(C.n) if d.dataset[i] == ds]
        if not idx:
            continue
        gap = np.array([float(C.h[i] - C.sur[i]) for i in idx])
        rows.append(dict(dataset=ds if ds != "All" else f"All {len(idx)} cells",
                         under=f"{sum(C.h[i] > C.sur[i] for i in idx)} of {len(idx)}",
                         median_gap=round(float(np.median(gap)), 3),
                         fail_B=sum(C.h[i] > C.B[i] for i in idx),
                         fail_01=f"{sum(C.h[i] > Fraction(1, 10) for i in idx)} of {len(idx)}"))
    out = pd.DataFrame(rows)
    per = out[~out.dataset.str.startswith("All")].sort_values("median_gap", ascending=False, kind="stable")
    return pd.concat([per, out[out.dataset.str.startswith("All")]]).reset_index(drop=True)


def figure4(C: Cells):
    """Per dataset and tau: failing cells, cells certified by U_P/Ibar, requests, misses and false alarms."""
    rows = []
    for ds in DS_ORDER:
        idx = [i for i in range(C.n) if C.df.dataset[i] == ds]
        for tau in TAUS:
            t = Fraction(tau)
            p = {i: C.dbar[i] <= t for i in idx}
            r = dict(dataset=ds, tau=float(t), cells=len(idx), failing=sum(not p[i] for i in idx),
                     certified_UP_over_Ibar=sum(p[i] and C.d_up[i] <= t for i in idx))
            r["requests"] = sum(not (C.d_up[i] <= t) for i in idx)
            for k, s in (("dyadic", C.d_sur), ("normalized_clique", C.d_ncl), ("reconstruction", C.d_rec)):
                ev = [i for i in idx if s[i] is not None]
                r[f"{k}_misses"] = sum((not p[i]) and s[i] <= t for i in ev) if ev else None
                r[f"{k}_false_alarms"] = sum(p[i] and s[i] > t for i in ev) if ev else None
            r["UP_false_certificates"] = sum((not p[i]) and C.d_up[i] <= t for i in idx)
            rows.append(r)
    return pd.DataFrame(rows)


def scaled(C: Cells, factor: str):
    """Verdicts at the shared budget scaled by `factor`."""
    f = Fraction(factor)
    b = [f * x for x in C.B]
    return row_counts(C, dict(true=C.h, Dy=C.sur, NC=C.nc, Rec=C.rec, U_P=C.up), b, f"B x {factor}")


def no_false_certificate(C: Cells, rmin_col="rmin_declared"):
    """True if no certificate of Props. 2 and 3 certifies a failing cell at any budget of Table 4."""
    bad = 0
    for f in FACTORS:
        fr = Fraction(f)
        for i in range(C.n):
            B = fr * C.B[i]
            if C.h[i] > B:
                bad += (C.up[i] <= B) + (C.uq[i] <= B) + (C.uq50[i] <= B)
    for i in range(C.n):
        t = Fraction(1, 10)
        if C.h[i] > t:
            bad += (C.up[i] <= t) + (C.uq[i] <= t) + (C.uq50[i] <= t)
    for tau in TAUS:
        t = Fraction(tau)
        for i in range(C.n):
            if C.dbar[i] > t:
                rmin = int(C.df[rmin_col][i])
                for c in (nc_constant(), nc_constant(None, 50), nc_constant(rmin, 50)):
                    bad += c * C.d_ncl[i] <= t
                bad += C.d_up[i] <= t
    return bad == 0, bad
