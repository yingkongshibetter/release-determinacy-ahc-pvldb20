"""HGP, CGP, certificates and pairwise auditors against brute-force definitions on random small inputs."""
import itertools
import random
import sys
import unittest
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ahc_audit import auditors, certificates as cert, sql  # noqa: E402
from ahc_audit.hgp import (CellStats, cgp_exact, hgp_exact, hgp_from_size_sums, per_size_counts_and_sums,  # noqa: E402
                           mean_worst_gap)
from ahc_audit.releases import (clique_pair_keys, normalized_clique, random_walk_P, top_k_per_row,  # noqa: E402
                                transition_T, unweighted_clique, degrees)


def random_instance(rng, n_max=12, m_max=10, k_max=6, groups=3, clusters=3):
    n = rng.randint(4, n_max)
    units = [tuple(sorted(rng.sample(range(n), rng.randint(2, min(k_max, n))))) for _ in range(rng.randint(1, m_max))]
    g = np.array([rng.randrange(groups) for _ in range(n)])
    cl = np.array([rng.randrange(clusters) for _ in range(n)])
    idx = np.concatenate([np.array(e) for e in units])
    ptr = np.concatenate([[0], np.cumsum([len(e) for e in units])])
    H = sp.csr_matrix((np.ones(idx.size, dtype=np.int64), idx, ptr), shape=(len(units), n))
    H.sort_indices()
    return H, units, g, cl


def psi_bruteforce(e, g, cl):
    """Eq. (2): sum over present groups of p_{g|e} TV(q_{e,g}, q_e)."""
    s = len(e)
    tot = Fr(0)
    for gg in set(g[list(e)]):
        mem = [v for v in e if g[v] == gg]
        tv = sum(abs(Fr(sum(1 for v in mem if cl[v] == c), len(mem)) - Fr(sum(1 for v in e if cl[v] == c), s))
                 for c in set(cl[list(e)])) / 2
        tot += Fr(len(mem), s) * tv
    return tot


def hgp_bruteforce(units, g, cl):
    vol = sum(len(e) for e in units)
    return sum(len(e) * psi_bruteforce(e, g, cl) for e in units) / vol


class TestHGP(unittest.TestCase):
    def test_hgp_equals_definition(self):
        rng = random.Random(0)
        for _ in range(300):
            H, units, g, cl = random_instance(rng)
            self.assertEqual(hgp_exact(H, cl, g), hgp_bruteforce(units, g, cl))
            self.assertEqual(hgp_from_size_sums(per_size_counts_and_sums(H, cl, g)), hgp_bruteforce(units, g, cl))

    def test_cgp(self):
        rng = random.Random(1)
        for _ in range(200):
            _, _, g, cl = random_instance(rng)
            n = g.size
            want = Fr(0)
            for c in set(cl):
                Cn = [v for v in range(n) if cl[v] == c]
                tv = sum(abs(Fr(sum(1 for v in Cn if g[v] == gg), len(Cn)) - Fr(int((g == gg).sum()), n)) for gg in set(g)) / 2
                want += Fr(len(Cn), n) * tv
            self.assertEqual(cgp_exact(cl, g), want)

    def test_certificates_and_auditors(self):
        rng = random.Random(2)
        for _ in range(300):
            H, units, g, cl = random_instance(rng)
            st = CellStats(H, cl, g)
            h = st.hgp
            self.assertEqual(h, hgp_bruteforce(units, g, cl))
            up, uq = cert.u_p(st), cert.u_q(st)
            # Prop. 2: HGP <= U_P <= U_Q, U_P <= U_Q,rbar; equality for units of at most three members
            self.assertLessEqual(h, up)
            self.assertLessEqual(up, uq)
            self.assertLessEqual(up, cert.u_q_cap(st, st.max_size))
            if st.max_size <= 3:
                self.assertEqual(h, up)
            # release scans equal the exact values
            P = random_walk_P(H)
            self.assertAlmostEqual(cert.u_p_scan(P, g, cl), float(up), places=12)
            self.assertAlmostEqual(cert.u_p_scan_T(transition_T(H), degrees(H), g, cl), float(up), places=12)
            W = normalized_clique(H)
            if W.nnz:
                self.assertAlmostEqual(cert.u_q_scan(W, g, cl), float(uq), places=12)
                self.assertAlmostEqual(auditors.nc_surrogate_scan(W, g, cl), float(uq) / 2, places=12)
            # Prop. 3(a): Ibar from P, Dbar <= U_P / Ibar; Prop. 3(b) with c = 2
            self.assertAlmostEqual(cert.ibar_scan(P, g), float(st.ibar), places=12)
            if st.SZ:
                self.assertLessEqual(st.dbar, cert.u_p_over_ibar(st))
                self.assertLessEqual(st.dbar, 2 * cert.nc_dissimilarity(st))
            # dyadic surrogate = HGP of one size-2 unit per clique pair
            pu, pv = unweighted_clique(H)
            sur, _ = auditors.dyadic(pu, pv, g, cl)
            self.assertEqual(sur, hgp_bruteforce(list(zip(pu.tolist(), pv.tolist())), g, cl))
            # psi_e <= I_e M_e gives HGP <= mean M_e
            self.assertLessEqual(float(h), mean_worst_gap(H, cl, g) + 1e-12)

    def test_pruning_keeps_largest(self):
        rng = random.Random(3)
        H, _, _, _ = random_instance(rng, n_max=20, m_max=20)
        T = transition_T(H)
        Tk = top_k_per_row(T, 3)
        self.assertTrue((np.diff(Tk.indptr) <= 3).all())
        self.assertTrue(np.allclose(np.asarray(T.sum(axis=1)).ravel(), 1.0))

    def test_clique_keys(self):
        rng = random.Random(4)
        H, units, _, _ = random_instance(rng)
        n = H.shape[1]
        want = sorted({u * n + v for e in units for u, v in itertools.combinations(e, 2)})
        self.assertEqual(clique_pair_keys(H).tolist(), want)


class TestSQL(unittest.TestCase):
    def test_sql_equals_numpy(self):
        rng = random.Random(5)
        engines = ["sqlite"]
        try:
            import duckdb  # noqa: F401
            engines.append("duckdb")
        except ImportError:
            pass
        for _ in range(20):
            H, units, g, cl = random_instance(rng)
            vals = pd.Series([f"g{x}" if x else None for x in g])        # group 0 is a missing value
            want = per_size_counts_and_sums(H, cl, pd.factorize(vals.fillna("__missing__"))[0])
            for eng in engines:
                rows = sql.per_size_counts_and_sums(eng, H, vals, cl)
                self.assertEqual(rows, want)
                ct = sql.contingency_table(eng, H, vals, cl)
                self.assertEqual(sql.hgp_from_contingency(eng, ct), hgp_from_size_sums(want))


if __name__ == "__main__":
    unittest.main()
