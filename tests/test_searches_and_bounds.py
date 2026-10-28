"""Budget strata order, declared-size extremes, witnesses, exchanges, inexact-release bounds, the small-instance LP."""
import itertools
import random
import sys
import unittest
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ahc_audit import budgets, declared as DC, exchange as EX, inexact  # noqa: E402
from ahc_audit.hgp import CellStats, contingency_rows, hgp_exact  # noqa: E402
from ahc_audit.small_instances import lp_check  # noqa: E402
from tests.test_audits import hgp_bruteforce, random_instance  # noqa: E402


class TestBudgets(unittest.TestCase):
    def test_strata_follow_the_original_stream(self):
        """The strata and their order equal those of the pandas group-by that produced the paper's budgets."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            n = int(rng.integers(30, 200))
            labels = rng.integers(0, int(rng.integers(2, 25)), size=n)
            clusters = rng.integers(0, 7, size=n)
            key = pd.Series(labels.astype(str)) + "::" + pd.Series(np.zeros(n, dtype=int).astype(str))
            r1, r2 = np.random.default_rng(99), np.random.default_rng(99)
            out = clusters.copy()
            for _, idx in pd.Series(np.arange(n)).groupby(key).groups.items():
                idx = np.asarray(list(idx), dtype=np.int64)
                if idx.size > 1:
                    out[idx] = r1.permutation(out[idx])
            mine = budgets.reassign(clusters, budgets.strata(labels), r2)
            self.assertTrue(np.array_equal(out, mine))
            self.assertEqual(r1.bit_generator.state, r2.bit_generator.state)
            self.assertTrue(np.array_equal(np.sort(mine), np.sort(clusters)))


class TestDeclared(unittest.TestCase):
    def test_triple_sums(self):
        """S_e (sum over the triples of e of 9 psi_t) by composition counting equals explicit enumeration."""
        rng = random.Random(0)
        for _ in range(300):
            k = rng.randint(3, 9)
            gs = [rng.randrange(3) for _ in range(k)]; cs = [rng.randrange(3) for _ in range(k)]
            H = sp.csr_matrix((np.ones(k, dtype=np.int64), np.arange(k), [0, k]), shape=(1, k))
            s, e, A, Ng, Nc, _, _ = contingency_rows(H, np.array(cs), np.array(gs))
            S = int(DC.triple_comp_sum(s[e], A, Ng, Nc).sum())
            want = sum(9 * hgp_bruteforce([t], np.array(gs), np.array(cs)) for t in itertools.combinations(range(k), 3))
            self.assertEqual(Fr(S), want)

    def test_l0_extremes_bruteforce(self):
        rng = random.Random(1)
        for _ in range(40):
            n = rng.randint(5, 8)
            units = [tuple(sorted(rng.sample(range(n), rng.randint(3, min(5, n))))) for _ in range(rng.randint(1, 4))]
            g = np.array([rng.randrange(2) for _ in range(n)]); cl = np.array([rng.randrange(2) for _ in range(n)])
            idx = np.concatenate([np.array(e) for e in units]); ptr = np.concatenate([[0], np.cumsum([len(e) for e in units])])
            H = sp.csr_matrix((np.ones(idx.size, dtype=np.int64), idx, ptr), shape=(len(units), n))
            vals = []
            for choice in itertools.product([0, 1], repeat=len(units)):
                U = []
                for e, c in zip(units, choice):
                    U.extend(itertools.combinations(e, 3) if c and len(e) > 3 else [e])
                vals.append(hgp_bruteforce(U, g, cl))
            lo, hi = DC.l0_extremes(H, cl, g)
            self.assertEqual((lo, hi), (min(vals), max(vals)))

    def test_witness_roundtrip(self):
        rng = random.Random(2)
        H, units, g, cl = random_instance(rng, m_max=6, k_max=4)
        ds = DC.DataStructure(H, (2, 6))
        tmp = ROOT / "outputs" / "tests"; tmp.mkdir(parents=True, exist_ok=True)
        DC.save_witness(tmp / "w.npz", ds.units, ds.units)
        h, chk, ok, nch = DC.verify_witness(tmp / "w.npz", ds, "L2", cl, g)
        self.assertTrue(ok and nch == 0 and h == hgp_exact(H, cl, g))


class TestExchange(unittest.TestCase):
    def test_verify_keeps_cluster_audits(self):
        rng = random.Random(3)
        H, units, g, cl = random_instance(rng, n_max=14, m_max=12)
        c = cl.copy(); su, sv, tr = [], [], []
        for _ in range(30):
            u, v = rng.sample(range(g.size), 2)
            if g[u] == g[v] and c[u] != c[v]:
                c[u], c[v] = c[v], c[u]; su.append(u); sv.append(v); tr.append(float(hgp_exact(H, c, g)))
        out = EX.verify(H, g, cl, np.array(su, dtype=np.int32), np.array(sv, dtype=np.int32), np.array(tr), Fr(1, 2))
        self.assertTrue(out["moves_same_group_cross_cluster"] and out["cluster_audits_identical"])
        self.assertAlmostEqual(out["HGP_end"], float(hgp_exact(H, c, g)), places=12)


class TestInexact(unittest.TestCase):
    def test_bounds_are_conservative(self):
        rng = random.Random(4)
        for _ in range(20):
            H, units, g, cl = random_instance(rng, n_max=25, m_max=25, k_max=6)
            st = CellStats(H, cl, g)
            R = inexact.Releases(H, tops=(3,))
            up, uq = st.SX / st.vol, st.SX1 / st.vol
            for B in (st.hgp, up, uq, (st.hgp + up) / 2, Fr(1, 2)):
                r = R.cell(cl, g, B, st.hgp, up, uq)
                for k, v in r.items():
                    if k.endswith("_cert") and "plugin" not in k and v:
                        self.assertLessEqual(st.hgp, B, k)


class TestSmallInstanceLP(unittest.TestCase):
    def test_lp_chain(self):
        stats, rows = lp_check("L2", log=lambda *a: None)
        self.assertEqual(stats["chain_ok"], stats["sets"])
        self.assertEqual(stats["dual_ok"], stats["sets"])
        self.assertEqual(stats["lp_eq_hmax"], stats["sets"])     # on P the relaxation is tight on this configuration


if __name__ == "__main__":
    unittest.main()
