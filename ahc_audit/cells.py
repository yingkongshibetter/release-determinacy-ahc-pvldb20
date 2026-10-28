"""All statistics of one cell (method, dataset, audited attribute) and of reference partitions.

A cell passes a budget when HGP_true <= budget. A miss is a failing cell that an auditor certifies, and a
false alarm is a passing cell that it fails. Every comparison of an exact statistic with a budget is made
in exact rational arithmetic (the float budget is converted exactly).
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np

from . import auditors, certificates as cert
from .data import REPO_ROOT, UNIT_SIZES, Dataset
from .hgp import CellStats, cgp_exact, hgp_float, mean_worst_gap, mixed_membership_share, population_interaction, split_membership_share
from .releases import normalized_clique, random_walk_P, unweighted_clique

RECON_DIR = REPO_ROOT / "outputs" / "reconstruction"
RECON_VOLUME_CAP = 50_000_000   # above this volume the reconstruction is not self-calibrated (IMDb-Cast)
RBAR = 50


def frs(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator}"


class Context:
    """Releases of one dataset, built once and shared by all its cells."""

    def __init__(self, D: Dataset, recon_dir: Path = RECON_DIR, with_P=True):
        self.D = D
        self.pu, self.pv = unweighted_clique(D.H)
        W = normalized_clique(D.H)
        self.wu, self.wv, self.w = W.row.astype(np.int64), W.col.astype(np.int64), W.data
        assert np.array_equal(np.sort(self.wu * D.n + self.wv), self.pu * D.n + self.pv)
        self.wsum = float(self.w.sum())
        self.P = random_walk_P(D.H) if with_P else None
        self.Hr = None
        self.recon_status = "not built"
        f = Path(recon_dir) / f"{D.name}.npz"
        j = Path(recon_dir) / f"{D.name}.json"
        if f.exists():
            self.Hr = auditors.Reconstruction(f)
            self.recon_status = "ok"
        elif j.exists():
            import json
            self.recon_status = json.loads(j.read_text(encoding="utf-8"))["status"]

    @property
    def recon_self_calibrated(self) -> bool:
        return self.Hr is not None and self.Hr.nnz <= RECON_VOLUME_CAP

    def reference_stats(self, attribute):
        """Statistics of a reference partition, for the shared and self-calibrated budgets."""
        D = self.D
        g = D.group(attribute)
        gd = g[self.wu] != g[self.wv]
        npairs = self.wu.size
        Hr = self.Hr.csr() if self.recon_self_calibrated else None

        def fn(p):
            msk = gd & (p[self.wu] != p[self.wv])
            out = {"HGP": hgp_float(D.H, p, g),
                   "sur": int(msk.sum()) / (2.0 * npairs),
                   "ncl": float(self.w[msk].sum()) / (2.0 * self.wsum)}
            if Hr is not None:
                out["rec"] = auditors.reconstruction_hgp_float(Hr, p, g)
            return out
        return fn

    def cell(self, method, attribute, B=None, budgets=None):
        D = self.D
        cl = D.partition(method)
        g = D.group(attribute)
        st = CellStats(D.H, cl, g)
        hgp = st.hgp
        sur, d_sur = auditors.dyadic(self.pu, self.pv, g, cl)
        msk = (g[self.wu] != g[self.wv]) & (cl[self.wu] != cl[self.wv])
        qhalf_scan = float(self.w[msk].sum()) / (2.0 * self.wsum)
        uq = cert.u_q(st)
        assert abs(qhalf_scan - float(uq) / 2) < 1e-12, "normalized-clique surrogate != U_Q/2"
        up = cert.u_p(st)
        if self.P is not None:
            assert abs(cert.u_p_scan(self.P, g, cl) - float(up)) < 1e-12, "U_P scan != exact U_P"
            assert abs(cert.ibar_scan(self.P, g) - float(st.ibar)) < 1e-12, "Ibar scan != exact Ibar"
        rmin, rmax = UNIT_SIZES[D.name]
        assert st.min_size >= rmin and st.max_size <= rmax
        cgp = cgp_exact(cl, g)
        row = dict(method=method, dataset=D.display, ds=D.name, attribute=attribute, K=int(np.unique(cl).size),
                   HGP=float(hgp), HGP_exact=frs(hgp), HGP_sur=float(sur), HGP_sur_exact=frs(sur),
                   Qhalf=float(uq / 2), U_P=float(up), U_P_exact=frs(up), U_Q=float(uq), U_Q_exact=frs(uq),
                   U_Q50=float(Fraction(RBAR - 1, RBAR) * uq),
                   CGP=float(cgp), CGP_exact=frs(cgp),
                   Ibar=float(st.ibar), Dbar=float(st.dbar), Dbar_exact=frs(st.dbar),
                   D_sur=float(d_sur), D_sur_exact=frs(d_sur),
                   D_ncl=float(cert.nc_dissimilarity(st)), D_ncl_exact=frs(cert.nc_dissimilarity(st)),
                   D_UP=float(cert.u_p_over_ibar(st)), D_UP_exact=frs(cert.u_p_over_ibar(st)),
                   D_CGP_exact=frs(cgp / population_interaction(g)),
                   rmin_declared=rmin, contingency_rows=st.rows,
                   mixed_share=mixed_membership_share(D.H, g), split_share=split_membership_share(D.H, cl),
                   mean_worst_gap=mean_worst_gap(D.H, cl, g))
        if self.Hr is not None:
            rY, rZ, rvol = auditors.reconstruction_sums(self.Hr, cl, g)
            row.update(HGP_rec=float(rY / rvol), HGP_rec_exact=frs(rY / rvol),
                       D_rec_exact=frs(rY / rZ) if rZ else "0")
        else:
            row.update(HGP_rec=np.nan, HGP_rec_exact="", D_rec_exact="")
        row["recon_status"] = self.recon_status
        return row

