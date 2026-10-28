"""Conservative certificates (Propositions 2 and 3).

    U_P       share of the mass of P on ordered crossing pairs (no cap on unit size)
    U_Q       weighted share of crossing pairs in the normalized clique
    U_Q,rbar  (rbar - 1)/rbar * U_Q when no unit has more than rbar members
    U_P/Ibar  the certificate on the scale of Dbar (Prop. 3(a))
    c * W_x / W_xi  the normalized-clique certificate on the scale of Dbar (Prop. 3(b))
A conservative certificate U returns "certified" when U <= budget and "undetermined" otherwise.
The exact values come from per-unit integers (hgp.CellStats); the release scans below give the same
values in float64 from the released matrices only.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from .hgp import CellStats


def crossing(g, cl, u, v):
    return (g[u] != g[v]) & (cl[u] != cl[v])


# ---------- exact values (from the incidence; equal to the release scans) ----------
def u_p(st: CellStats) -> Fraction:
    return st.SX / st.vol


def u_q(st: CellStats) -> Fraction:
    return st.SX1 / st.vol


def u_q_cap(st: CellStats, rbar: int) -> Fraction:
    assert st.max_size <= rbar, "U_Q,rbar needs every unit to have at most rbar members"
    return Fraction(rbar - 1, rbar) * u_q(st)


def u_p_over_ibar(st: CellStats) -> Fraction:
    """Prop. 3(a): Dbar <= U_P / Ibar = sum P chi / sum P xi."""
    return st.SX / st.SZ if st.SZ else Fraction(0)


def nc_dissimilarity(st: CellStats) -> Fraction:
    """W_x / W_xi of the normalized clique: crossing weight over cross-group weight."""
    return st.SX1 / st.SZ1 if st.SZ1 else Fraction(0)


def nc_constant(rmin: int | None = None, rbar: int | None = None) -> Fraction:
    """c of Prop. 3(b): 2 without declaration; (rbar-1)/rbar * rmin/(rmin-1) under declared sizes."""
    if rmin is None and rbar is None:
        return Fraction(2)
    rmin = 2 if rmin is None else rmin
    top = Fraction(1) if rbar is None else Fraction(rbar - 1, rbar)
    return top * Fraction(rmin, rmin - 1)


# ---------- scans of released matrices (float64) ----------
def u_p_scan(P, g, cl) -> float:
    """One pass over the entries of P (full symmetric matrix with diagonal)."""
    msk = crossing(g, cl, P.row, P.col)
    return float(P.data[msk].sum() / P.data.sum())


def u_p_scan_T(T, d, g, cl) -> float:
    """U_P from T and the degrees, as P = diag(d) T."""
    Tc = T.tocoo()
    w = d[Tc.row] * Tc.data
    msk = crossing(g, cl, Tc.row, Tc.col)
    return float(w[msk].sum() / w.sum())


def u_q_scan(W, g, cl) -> float:
    """One pass over the upper-triangle entries of the normalized clique."""
    msk = crossing(g, cl, W.row, W.col)
    return float(W.data[msk].sum() / W.data.sum())


def ibar_scan(P, g) -> float:
    """Prop. 3(a): Ibar = sum P xi / sum P over all ordered pairs."""
    msk = g[P.row] != g[P.col]
    return float(P.data[msk].sum() / P.data.sum())
