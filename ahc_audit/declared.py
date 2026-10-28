"""Compatible inputs of the unweighted clique under declarations (RQ1, Figure 3).

Three declared classes; every compatible input keeps the node set, the attributes, the partition and the
unweighted clique (the set of node pairs covered by at least one unit):
  L0  unit sizes alone (3..50, or 3..25 on OA-CS-3+), any multiplicity. Replacing a unit of k >= 4 members
      by its C(k,3) triples keeps the unweighted clique, and the exact extremes of HGP over all subsets of
      replaced units follow from a Dinkelbach iteration in exact arithmetic (l0_extremes).
  L1  also the number of units and the size of every unit, and no new repeated unit (local search in
      declared_search.py; the witnesses found are stored and verified here).
  L2  also every node degree.
A verdict at budget B is undetermined when the compatible inputs found lie on both sides of B:
low <= B < high, with low and high the extremes found (always including the data itself).
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction

import numpy as np
import scipy.sparse as sp

from .hgp import contingency_rows, exact_sum, unit_counts
from .releases import clique_pair_keys


# ---------- unit lists ----------
def units_of(H):
    H = H.tocsr()
    return [np.sort(H.indices[H.indptr[r]:H.indptr[r + 1]]).astype(np.int64) for r in range(H.shape[0])]


_TRIU = {}


def triu(k):
    if k not in _TRIU:
        _TRIU[k] = np.triu_indices(k, 1)
    return _TRIU[k]


def unit_pair_keys(mem, n):
    iu, iv = triu(mem.size)
    return mem[iu] * n + mem[iv]


def unit_pair_index(mem, keys, n):
    pk = unit_pair_keys(mem, n)
    idx = np.searchsorted(keys, pk)
    assert (idx < keys.size).all() and (keys[idx] == pk).all()
    return idx


def pair_counts(units, keys, n):
    allk = np.concatenate([unit_pair_keys(e, n) for e in units])
    idx = np.searchsorted(keys, allk)
    assert (keys[idx] == allk).all()
    return np.bincount(idx, minlength=keys.size).astype(np.int64)


def max_multiplicity(units):
    return max(Counter(tuple(e.tolist()) for e in units).values())


def repeated_units(units):
    return len(units) - len(set(tuple(e.tolist()) for e in units))


def adjacency(keys, n):
    u = keys // n; v = keys % n
    A = sp.csr_matrix((np.ones(2 * keys.size, dtype=np.int8), (np.concatenate([u, v]), np.concatenate([v, u]))),
                      shape=(n, n))
    A.sort_indices()
    return A


def build_H(units, n):
    sizes = np.array([e.size for e in units], dtype=np.int64)
    indptr = np.concatenate([[0], np.cumsum(sizes)])
    idx = np.concatenate(units).astype(np.int64)
    H = sp.csr_matrix((np.ones(idx.size, dtype=np.int64), idx, indptr), shape=(len(units), n))
    H.sort_indices()
    return H


def Y_rows(Gr, Cr):
    """Y / multiplicity per member, summed: Y = sum over cells of max(k n_gC - n_g n_C, 0), one row per unit (float)."""
    k = Gr.shape[1]
    eqg = Gr[:, :, None] == Gr[:, None, :]
    eqc = Cr[:, :, None] == Cr[:, None, :]
    ng = eqg.sum(2); nc = eqc.sum(2); nl = (eqg & eqc).sum(2)
    return (np.maximum(k * nl - ng * nc, 0) / nl).sum(1)


def exact_hgp(H, cl, g) -> Fraction:
    s, _, Y, _, _ = unit_counts(H, cl, g)
    return exact_sum(s, Y) / int(s.sum())


# ---------- L0: exact extremes over replacing units by their triples ----------
def comb_small(x, r):
    x = x.astype(np.int64)
    return {0: np.ones_like(x), 1: x.copy(), 2: x * (x - 1) // 2, 3: x * (x - 1) * (x - 2) // 6}[r]


COMPS = [(a, b, d, 3 - a - b - d) for a in range(1, 4) for b in range(0, 4) for d in range(0, 4) if a + b + d <= 3]


def triple_comp_sum(k, A, Ng, Nc):
    """Per contingency row (g, C) of a unit of size k: sum over the unit's triples of max(3 n_tgC - n_tg n_tC, 0)."""
    Bg = Ng - A; Dc = Nc - A; R = k - Ng - Nc + A
    tot = np.zeros(A.size, dtype=np.int64)
    for a, b, d, r in COMPS:
        res = 3 * a - (a + b) * (a + d)
        if res <= 0:
            continue
        tot += res * comb_small(A, a) * comb_small(Bg, b) * comb_small(Dc, d) * comb_small(R, r)
    return tot


def dinkelbach(numN, volN, numT, volT, aint, bint, sizes_e, Y, S, mode):
    """Exact max or min of sum(num)/sum(vol) over a per-unit choice: keep the unit (N) or its triples (T)."""
    lam = float(numN.sum() / volN.sum())
    choose = None
    for _ in range(200):
        d = (numT - numN) - lam * (volT - volN)
        new = d > 0 if mode == "max" else d < 0
        if choose is not None and np.array_equal(new, choose):
            break
        choose = new
        lam = float(np.where(choose, numT, numN).sum() / np.where(choose, volT, volN).sum())

    def exact_ratio(ch):
        num = Fraction(0)
        for kk in np.unique(sizes_e[~ch]):
            num += Fraction(int(Y[(~ch) & (sizes_e == kk)].sum()), int(kk))
        num += Fraction(int(S[ch].sum()), 3)
        return num / int(np.where(ch, volT, volN).astype(np.int64).sum())

    uniq, inv = np.unique(np.stack([aint, bint], axis=1), axis=0, return_inverse=True)
    inv = inv.ravel()
    lamx = exact_ratio(choose)
    for _ in range(200):
        p, q = lamx.numerator, lamx.denominator
        sgn = [(q * int(a) - p * int(b)) for a, b in uniq.tolist()]
        pos = np.array([x > 0 for x in sgn]); neg = np.array([x < 0 for x in sgn])
        better = pos[inv] if mode == "max" else neg[inv]
        worse = neg[inv] if mode == "max" else pos[inv]
        viol = (choose & worse) | ((~choose) & better)
        if not viol.any():
            return lamx
        choose = np.where(viol, ~choose, choose)
        lamx = exact_ratio(choose)
    raise RuntimeError("exact Dinkelbach iteration did not converge")


def l0_extremes(H, cl, g):
    """(h_min, h_max) of HGP over all inputs that replace any subset of units by their triples (exact)."""
    s, e, A, Ng, Nc, _, _ = contingency_rows(H, cl, g)
    m = H.shape[0]
    pos = np.maximum(s[e] * A - Ng * Nc, 0)
    Y = np.zeros(m, dtype=np.int64); np.add.at(Y, e, pos)
    comp = triple_comp_sum(s[e], A, Ng, Nc)
    S = np.zeros(m, dtype=np.int64); np.add.at(S, e, comp)
    C3 = s * (s - 1) * (s - 2) // 6
    sf = s.astype(np.float64)
    nN = Y / sf; vN = sf; nT = S / 3.0; vT = 3.0 * C3
    aint = s * S - 3 * Y; bint = 3 * s * (3 * C3 - s)
    return (dinkelbach(nN, vN, nT, vT, aint, bint, s, Y, S, "min"),
            dinkelbach(nN, vN, nT, vT, aint, bint, s, Y, S, "max"))


# ---------- L1 / L2 witnesses ----------
def save_witness(path, best_units, units):
    """Store a witness as the units that differ from the data."""
    ch = np.array([i for i in range(len(units)) if not np.array_equal(best_units[i], units[i])], dtype=np.int64)
    mem = [best_units[i] for i in ch.tolist()]
    indptr = np.concatenate([[0], np.cumsum([x.size for x in mem])]).astype(np.int64)
    members = np.concatenate(mem).astype(np.int32) if mem else np.zeros(0, dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, changed=ch, indptr=indptr, members=members)


def load_witness(path, units):
    z = np.load(path)
    ch, ip, mem = z["changed"], z["indptr"], z["members"]
    w = list(units)
    for j, i in enumerate(ch.tolist()):
        w[i] = mem[ip[j]:ip[j + 1]].astype(np.int64)
    return w, ch


class DataStructure:
    """What every compatible input must keep, computed once per dataset."""

    def __init__(self, H, size_range):
        self.H = H
        self.m, self.n = H.shape
        self.units = units_of(H)
        self.hist = np.sort(np.array([e.size for e in self.units]))
        self.keys = clique_pair_keys(H)
        self.deg = np.bincount(np.concatenate(self.units), minlength=self.n)
        self.mult = max_multiplicity(self.units)
        self.rep = repeated_units(self.units)
        self.size_range = size_range


def verify_witness(path, ds: DataStructure, rung: str, cl, g):
    """Rebuild a witness from the data and its stored diff, check that it is compatible under the rung,
    and return (exact HGP, checks dict, verified)."""
    wu, ch = load_witness(path, ds.units)
    Hw = build_H(wu, ds.n)
    sizes = np.diff(Hw.indptr)
    chk = dict(m=Hw.shape[0] == ds.m,
               unit_sizes_each=all(wu[i].size == ds.units[i].size for i in ch.tolist()),
               histogram=bool(np.array_equal(np.sort(sizes), ds.hist)),
               size_range=bool(sizes.min() >= ds.size_range[0] and sizes.max() <= ds.size_range[1]),
               distinct_members=bool((Hw.data == 1).all()) and all(np.unique(x).size == x.size for x in wu),
               clique=bool(np.array_equal(clique_pair_keys(Hw), ds.keys)),
               max_multiplicity=max_multiplicity(wu) <= ds.mult,
               repeated_units=repeated_units(wu) <= ds.rep)
    if rung == "L2":
        chk["degrees"] = bool(np.array_equal(np.bincount(np.concatenate(wu), minlength=ds.n), ds.deg))
    return exact_hgp(Hw, cl, g), chk, all(chk.values()), int(ch.size)
