"""Certificates from rounded and pruned releases (Proposition 5; RQ2, last bullet).

The auditor receives an interval [l_o, h_o] that contains the exact value of each listed entry o. With N_h the
sum of h_o over listed crossing entries and M_l the sum of l_o over the other listed entries:
  (a) every positive entry listed:  U <= N_h / (N_h + M_l)  for P and the normalized clique (U_Q, U_Q,rbar);
  (b) degrees released:             U_P <= 1 - M_l / vol, whichever entries of P are missing.
Releases evaluated (dataset_releases):
  Q-sig6 / Q-3dp   normalized clique rounded to 6 significant digits (the cond-mat file) / to 3 decimals
  P-f64            T in float64 (the AHRC/AHCKA code path) with integer degrees; relative error at most
                   k 2^-53 / (1 - k 2^-53), k = d_max + 2
  T-6dp/4dp/3dp    T rounded to p decimals, entries that round to zero dropped, with degrees (part (b))
  T-top200/top50   float64 T pruned to the k largest entries of each row, with degrees (part (b)); the auditor
                   lists the transpose of each listed entry (symmetric closure)
Rounded releases are decided in exact rational arithmetic; float64 releases certify only if the bound is at
most the budget minus 1e-9.
"""
from __future__ import annotations

import math
from fractions import Fraction as Fr

import numpy as np

from .releases import ExactEntries, normalized_clique, random_walk_P, top_k_per_row, transition_T

U64 = 2.0 ** -53
MARGIN = 1e-9
RBAR = 50


def gamma(k):
    return k * U64 / (1 - k * U64)


def decade_exact(x: Fr) -> int:
    k = math.floor(math.log10(x.numerator) - math.log10(x.denominator))
    while Fr(10) ** k > x:
        k -= 1
    while Fr(10) ** (k + 1) <= x:
        k += 1
    return k


def quantize(vals, rows, cols, exact_fn, mode, digits):
    """Round exact values (given by float approximations vals plus an exact fallback) to `digits` decimals
    (mode 'dp') or `digits` significant digits (mode 'sig'). Returns integer mantissa M and exponent E with
    obs = M * 10^E, and the number of entries resolved in exact arithmetic."""
    vals = np.asarray(vals, dtype=np.float64)
    n = vals.size
    if mode == "dp":
        E = np.full(n, -digits, dtype=np.int64)
        flag_dec = np.zeros(n, dtype=bool)
    else:
        lg = np.log10(vals)
        k = np.floor(lg).astype(np.int64)
        # within 1e-9 (relative) of a power of ten: whatever the exact decade, %g gives exactly 10^p
        near = np.abs(lg - np.rint(lg)) < 1e-9
        k = np.where(near, np.rint(lg).astype(np.int64), k)
        E = k - (digits - 1)
    x = vals * np.power(10.0, -E.astype(np.float64))
    fr = x - np.floor(x)
    flag = (np.abs(fr - 0.5) < 1e-4)
    if mode == "sig":
        flag &= ~near
    M = np.rint(x).astype(np.int64)
    if mode == "sig":
        M = np.where(near, 10 ** (digits - 1), M)
    idx = np.nonzero(flag)[0]
    for i in idx:
        ex = exact_fn(int(rows[i]), int(cols[i]))
        if mode == "sig":
            E[i] = decade_exact(ex) - (digits - 1)
            if round(ex * Fr(10) ** int(-E[i])) >= 10 ** digits:
                pass                          # rounds up into the next decade; value is still M * 10^E
        xe = ex * Fr(10) ** int(-E[i])
        M[i] = round(xe)                      # exact, ties to even
    return M, E, len(idx)


def sig_delta_units(M, E, digits):
    """Auditor's rounding radius for %g with `digits` significant digits, from the observation only:
    delta = 0.5 * 10^(floor(log10 obs) - digits + 1). In units of 10^E this is 0.5, or 5 when M = 10^digits."""
    return np.where(M >= 10 ** digits, 5.0, 0.5)


def to_scaled_int(M, E, dU):
    """Return integer arrays lo2, hi2 in the common unit 10^Emin / 2:  lo = max(M - dU, 0), hi = M + dU."""
    Emin = int(E.min())
    f = np.power(10, (E - Emin)).astype(np.int64)
    twoM = 2 * M
    d2 = np.rint(2 * dU).astype(np.int64)
    hi2 = (twoM + d2) * f
    lo2 = np.maximum(twoM - d2, 0) * f
    return lo2, hi2, Emin


def closure_lower(u, v, lo, n):
    """Symmetric closure of listed ordered entries (u, v) with lower bounds lo: per entry max(lo_uv, lo_vu)
    (lo_vu = 0 if (v, u) is unlisted), and the mask of entries whose transpose is unlisted (recovered by symmetry)."""
    key = u.astype(np.int64) * n + v
    keyT = v.astype(np.int64) * n + u
    order = np.argsort(key)
    ks = key[order]
    pos = np.minimum(np.searchsorted(ks, keyT), ks.size - 1)
    hasT = ks[pos] == keyT
    loT = np.where(hasT, lo[order][pos], 0)
    return np.maximum(lo, loT), ~hasT & (u != v)


class Releases:
    """The rounded and pruned releases of one dataset (attribute-free, built once)."""

    def __init__(self, H, tops=(200, 50)):
        H = H.tocsr()
        self.n = H.shape[1]
        s = np.diff(H.indptr).astype(np.int64)
        d = np.asarray(H.sum(axis=0)).ravel().astype(np.int64)
        self.vol = int(s.sum())
        ex = ExactEntries(H)
        W = normalized_clique(H)
        self.wu, self.wv, wf = W.row, W.col, W.data
        self.Q = {}
        for nm, mode, dg in [("Q-sig6", "sig", 6), ("Q-3dp", "dp", 3)]:
            Mq, Eq, _ = quantize(wf, self.wu, self.wv, ex.w, mode, dg)
            dU = sig_delta_units(Mq, Eq, dg) if mode == "sig" else np.full(Mq.size, 0.5)
            lo2, hi2, _ = to_scaled_int(Mq, Eq, dU)
            assert (lo2 > 0).all(), "a normalized-clique weight rounds to 0"
            self.Q[nm] = (lo2, hi2)
        T = transition_T(H).tocoo()
        self.tu, self.tv, tf = T.row, T.col, T.data
        self.eps = gamma(int(d.max()) + 2)
        df = d.astype(np.float64)
        Pobs = df[self.tu] * tf
        self.Plo = Pobs / (1 + self.eps) * (1 - 2 * U64)
        self.Phi = Pobs / (1 - self.eps) * (1 + 2 * U64)
        self.Tr = {}
        for dg in (6, 4, 3):
            Mt, Et, _ = quantize(tf, self.tu, self.tv, ex.t, "dp", dg)
            keep = Mt > 0
            lo_t = np.where(keep, d[self.tu] * np.maximum(2 * Mt - 1, 0), 0)   # unit 0.5e-dg on P
            closed, _ = closure_lower(self.tu, self.tv, lo_t, self.n)
            self.Tr[f"T-{dg}dp"] = (closed, dg)
        Tcsr = transition_T(H)
        self.top = {}
        for k in tops:
            Tp = top_k_per_row(Tcsr, k).tocoo()
            ku, kv, kf = Tp.row, Tp.col, Tp.data
            Pk = df[ku] * kf
            lo = Pk / (1 + self.eps) * (1 - 2 * U64)
            closed, rec = closure_lower(ku, kv, lo, self.n)
            self.top[k] = dict(ku=ku, kv=kv, lo=lo, closed=closed, rec=rec, Pk=Pk,
                               kept_share=float(Pk.sum() / self.vol))

    def cell(self, cl, g, B: Fr, HGP: Fr, UP: Fr, UQ: Fr):
        Bf = float(B)
        vol = self.vol
        row = dict(pass_B=HGP <= B, cert_U_P=UP <= B, cert_U_Q=UQ <= B, cert_U_Q50=UQ * Fr(RBAR - 1, RBAR) <= B)
        cq = (g[self.wu] != g[self.wv]) & (cl[self.wu] != cl[self.wv])
        for nm, (lo2, hi2) in self.Q.items():                               # Prop. 5(a), exact
            Nh = int(hi2[cq].sum()); Ml = int(lo2[~cq].sum())
            Ub = Fr(Nh, Nh + Ml)
            assert Ub >= UQ
            row[f"{nm}_UQ_cert"] = Ub <= B
            row[f"{nm}_UQ50_cert"] = Ub * Fr(RBAR - 1, RBAR) <= B
        ct = (g[self.tu] != g[self.tv]) & (cl[self.tu] != cl[self.tv])
        for nm, (closed_t, dg) in self.Tr.items():                         # Prop. 5(b), exact
            U3 = 1 - Fr(int(closed_t[~ct].sum()), 2 * 10 ** dg * vol)
            assert U3 >= UP
            row[f"{nm}_UP_cert"] = U3 <= B
        Nh_f = math.fsum(self.Phi[ct]); Ml_f = math.fsum(self.Plo[~ct])    # float64 T with degrees
        for nm, U in [("P-f64-box_UP", Nh_f / (Nh_f + Ml_f) * (1 + 4 * U64)), ("P-f64_UP", (1 - Ml_f / vol) * (1 + 4 * U64))]:
            assert U >= float(UP) * (1 - 1e-14)
            row[f"{nm}_cert"] = U <= Bf - MARGIN
        for k, q in self.top.items():                                      # pruned rows, Prop. 5(b)
            ku, kv = q["ku"], q["kv"]
            ck = (g[ku] != g[kv]) & (cl[ku] != cl[kv])
            Mlc = math.fsum(q["closed"][~ck]) + math.fsum(q["lo"][q["rec"] & ~ck])
            U_deg = (1 - Mlc / vol) * (1 + 4 * U64)
            assert U_deg >= float(UP) * (1 - 1e-14)
            row[f"top{k}_UP_cert"] = U_deg <= Bf - MARGIN
            Pk = q["Pk"]
            row[f"top{k}_plugin"] = float(Pk[ck].sum() / Pk.sum())          # share among kept entries (not sound)
            row[f"top{k}_plugin_cert"] = row[f"top{k}_plugin"] <= Bf
        return row
