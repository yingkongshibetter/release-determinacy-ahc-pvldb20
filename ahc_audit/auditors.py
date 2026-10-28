"""Pairwise auditors (Section 5.1, Table 4).

    dyadic surrogate         HGP on one size-2 unit per pair of the unweighted clique
    normalized-clique surr.  HGP on one size-2 unit per released pair, weighted by w_uv; equals U_Q / 2
    reconstruction auditor   HGP on the maximal cliques of the unweighted clique taken as units
                             (the training-free baseline of Wang and Kleinberg, ICLR 2024)
On the scale of Dbar, each auditor divides its HGP by the Ibar of the units that it assumes.
"""
from __future__ import annotations

import json
import time
from array import array
from fractions import Fraction

import numpy as np
import scipy.sparse as sp

from .hgp import exact_sum, unit_counts
from .releases import clique_pair_keys


def dyadic(pu, pv, g, cl):
    """(HGP_sur, Dbar_sur) as exact Fractions: crossing pairs / (2 pairs) and crossing / cross-group pairs."""
    dg = g[pu] != g[pv]
    nx = int(dg.sum())
    nxc = int((dg & (cl[pu] != cl[pv])).sum())
    return Fraction(nxc, 2 * pu.size), (Fraction(nxc, nx) if nx else Fraction(0))


def nc_surrogate_scan(W, g, cl) -> float:
    """Normalized-clique surrogate from the released weights (float); equals U_Q / 2."""
    msk = (g[W.row] != g[W.col]) & (cl[W.row] != cl[W.col])
    return float(W.data[msk].sum() / (2.0 * W.data.sum()))


# ---------- maximal-clique reconstruction ----------
TIME_CAP_S = 30 * 60
MEM_CAP_BYTES = 6 * 1024 ** 3
CLIQUE_CAP = 20_000_000


def maximal_cliques(H, time_cap=TIME_CAP_S, mem_cap=MEM_CAP_BYTES, clique_cap=CLIQUE_CAP, log=print):
    """Every maximal clique (size >= 2) of the unweighted clique graph as one unit.

    Enumeration by networkx.find_cliques (Bron-Kerbosch with pivoting). Aborts at the preset caps
    (30 min, 6 GB working set, 20 million cliques). Returns (incidence or None, info dict)."""
    import networkx as nx
    import psutil
    proc = psutil.Process()
    t0 = time.perf_counter()
    m, n = H.shape
    keys = clique_pair_keys(H)
    u, v = keys // n, keys % n
    G = nx.Graph()
    G.add_edges_from(zip(u.tolist(), v.tolist()))
    del keys
    nodes = array("i"); sizes = array("i")
    status = "ok"; peak = proc.memory_info().rss
    t1 = time.perf_counter()
    for c in nx.find_cliques(G):
        nodes.extend(c); sizes.append(len(c))
        if len(sizes) % 20000 == 0:
            rss = proc.memory_info().rss; peak = max(peak, rss)
            el = time.perf_counter() - t1
            if len(sizes) % 1_000_000 == 0:
                log(f"  {len(sizes)} cliques, {el:.0f}s, rss {rss / 1024**3:.2f} GB")
            if el > time_cap:
                status = f"aborted: time limit {time_cap}s"; break
            if rss > mem_cap:
                status = f"aborted: working set {rss / 1024**3:.2f} GB"; break
            if len(sizes) >= clique_cap:
                status = f"aborted: {clique_cap} cliques"; break
    del G
    sz = np.frombuffer(sizes, dtype=np.int32).astype(np.int64)
    info = dict(status=status, pairs=int(u.size), cliques=int(sz.size), clique_memberships=int(sz.sum()),
                t_enum_s=time.perf_counter() - t1, t_total_s=time.perf_counter() - t0,
                peak_rss_GB=max(peak, proc.memory_info().rss) / 1024 ** 3)
    if status != "ok":
        return None, info
    nd = np.frombuffer(nodes, dtype=np.int32).astype(np.int64)
    indptr = np.concatenate([[0], np.cumsum(sz)])
    Hr = sp.csr_matrix((np.ones(nd.size, dtype=np.int64), nd, indptr), shape=(sz.size, n))
    Hr.sort_indices()
    return Hr, info


def save_reconstruction(path, Hr, info):
    path.parent.mkdir(parents=True, exist_ok=True)
    if Hr is not None:
        sp.save_npz(path.with_suffix(".npz"), Hr)
    path.with_suffix(".json").write_text(json.dumps(info, indent=1), encoding="utf-8")


class Reconstruction:
    """A stored reconstruction, read without its data array (all ones), so that 3e8 memberships fit in memory."""

    def __init__(self, npz_path):
        z = np.load(npz_path)
        self.indptr = z["indptr"].astype(np.int64)
        self.indices = z["indices"]
        self.m, self.n = (int(x) for x in z["shape"])
        self.nnz = int(self.indptr[-1])
        self._csr = None

    def csr(self):
        if self._csr is None:
            self._csr = sp.csr_matrix((np.ones(self.nnz, dtype=np.int64), self.indices, self.indptr),
                                      shape=(self.m, self.n))
        return self._csr


def reconstruction_sums(R: Reconstruction, cl, g, chunk=10_000_000):
    """(sum Y/s, sum Z/s, vol) of the reconstruction as exact Fractions, in chunks of about `chunk` memberships."""
    indptr, indices, m, n = R.indptr, R.indices, R.m, R.n
    SY = Fraction(0); SZ = Fraction(0); vol = 0
    a = 0
    while a < m:
        b = int(np.searchsorted(indptr, indptr[a] + chunk, side="right")) - 1
        b = min(max(b, a + 1), m)
        ip = indptr[a:b + 1] - indptr[a]
        Hc = sp.csr_matrix((np.ones(int(ip[-1]), dtype=np.int64), indices[indptr[a]:indptr[b]], ip), shape=(b - a, n))
        s, _, Y, Z, _ = unit_counts(Hc, cl, g)
        SY += exact_sum(s, Y); SZ += exact_sum(s, Z); vol += int(s.sum())
        a = b
    return SY, SZ, vol


def reconstruction_hgp_float(Hr, cl, g) -> float:
    s, _, Y, _, _ = unit_counts(Hr, cl, g)
    return float((Y / s).sum() / s.sum())
