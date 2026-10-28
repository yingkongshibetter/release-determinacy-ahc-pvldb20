"""RQ1 (information lost by pairwise releases): Tables 3, 5 and 6, Figures 3 and 4, and every RQ1 number.

Needs outputs/cells.csv (run_cells.py; with --partition-calibrated for the B_pm counts of Figure 3).
Steps that read the datasets cache their per-cell results under outputs/rq1/ and are skipped when cached:
  exchange   exact replay of the stored same-group exchange sequences            about 20 s
  declared   exact extremes of HGP with unit sizes declared (L0)                 about 10 s
  witnesses  exact check of the stored compatible inputs of Figure 3 (L1, L2)    about 4 min
  policy     worst cluster-level invitation policy found (Prop. 1)               about 2 min
  components ACS-PUMS-HH: HGP of the connected components of the clique          seconds
Writes outputs/table3.csv, table5.csv, table6.csv, figure3.csv, figure4.csv and outputs/rq1.txt.
"""
from __future__ import annotations

import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from ahc_audit import data as dmod
from ahc_audit import declared as DC
from ahc_audit import exchange as EX
from ahc_audit import policy as PO
from ahc_audit.budgets import FACTORS, TAUS
from ahc_audit.data import DISPLAY, INTERNAL, METHODS, METHOD_FILE, REPO_ROOT
from ahc_audit.hgp import mixed_membership_share
from ahc_audit.releases import unweighted_clique
from ahc_audit.verdicts import DS_ORDER, OVERLAP, Cells, figure4, scaled, table5, table6
from options import base_parser

WITNESS_DIR = REPO_ROOT / "data" / "witnesses"
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


def table3(a, cells):
    rows = []
    for ds in dmod.DATASETS:
        D = dmod.load_dataset(ds, a.data_root, a.partitions)
        r = dict(dataset=D.display, nodes=D.n, units=D.m, memberships=D.vol,
                 unit_sizes=f"{int(D.sizes.min())}..{int(D.sizes.max())}")
        for meth in METHODS:
            r[f"K_{meth}"] = int(np.unique(D.partition(meth)).size)
        for attr in dmod.ATTRIBUTES[ds]:
            r[f"mixed_share_{attr}"] = round(mixed_membership_share(D.H, D.group(attr)), 3)
        rows.append(r)
    return pd.DataFrame(rows)


def declared_step(a, cells):
    rows = []
    for ds in dmod.OVERLAPPING:
        D = dmod.load_dataset(ds, a.data_root, a.partitions)
        for _, c in cells[cells.ds == ds].iterrows():
            lo, hi = DC.l0_extremes(D.H, D.partition(c.method), D.group(c.attribute))
            rows.append(dict(method=c.method, dataset=c.dataset, attribute=c.attribute, L0_low=str(lo), L0_high=str(hi)))
        print(f"  L0 {D.display} done", flush=True)
    return pd.DataFrame(rows)


def witness_step(a, cells):
    rows = []
    for ds in dmod.OVERLAPPING:
        D = dmod.load_dataset(ds, a.data_root, a.partitions)
        st = DC.DataStructure(D.H, dmod.UNIT_SIZES[ds])
        for _, c in cells[cells.ds == ds].iterrows():
            cl, g = D.partition(c.method), D.group(c.attribute)
            r = dict(method=c.method, dataset=c.dataset, attribute=c.attribute)
            for rung in ("L1", "L2"):
                for side in ("low", "high"):
                    f = WITNESS_DIR / "declared" / ds / f"{METHOD_FILE[c.method]}__{c.attribute}__{rung}_{side}.npz"
                    h, chk, ok, nch = DC.verify_witness(f, st, rung, cl, g)
                    r[f"{rung}_{side}"] = str(h); r[f"{rung}_{side}_verified"] = ok; r[f"{rung}_{side}_changed"] = nch
            rows.append(r)
        print(f"  witnesses {D.display} done", flush=True)
    return pd.DataFrame(rows)


def exchange_step(a, cells):
    rows = []
    for ds in dmod.DATASETS:
        D = dmod.load_dataset(ds, a.data_root, a.partitions)
        for _, c in cells[cells.ds == ds].iterrows():
            z = np.load(WITNESS_DIR / "exchange" / ds / f"{METHOD_FILE[c.method]}__{c.attribute}.npz")
            r = EX.verify(D.H, D.group(c.attribute), D.partition(c.method), z["swap_u"], z["swap_v"], z["traj"],
                          Fraction(float(c.B)))
            rows.append(dict(method=c.method, dataset=c.dataset, attribute=c.attribute, **r))
        print(f"  exchange {D.display} done", flush=True)
    return pd.DataFrame(rows)


def policy_step(a, cells):
    rows = []
    for di, ds in enumerate(dmod.DATASETS):
        D = dmod.load_dataset(ds, a.data_root, a.partitions)
        for mi, meth in enumerate(METHODS):
            cl = D.partition(meth)
            PI = PO.draws(int(np.unique(cl).size), di, mi)
            for _, c in cells[(cells.ds == ds) & (cells.method == meth)].iterrows():
                best_draw, gg, gex, mean = PO.worst_policy(D.H, cl, D.group(c.attribute), PI)
                rows.append(dict(method=meth, dataset=c.dataset, attribute=c.attribute, G_mean=mean, G_best_draw=best_draw,
                                 G_greedy=gg, G_exact=gex))
        print(f"  policy {D.display} done", flush=True)
    return pd.DataFrame(rows)


def components_step(a, cells):
    ds = "acs_pums_household"
    D = dmod.load_dataset(ds, a.data_root, a.partitions)
    pu, pv = unweighted_clique(D.H)
    A = sp.csr_matrix((np.ones(pu.size), (pu, pv)), shape=(D.n, D.n))
    ncomp, lab = connected_components(A, directed=False)
    Hc = sp.csr_matrix((np.ones(D.n, dtype=np.int64), (lab, np.arange(D.n))), shape=(ncomp, D.n))
    Hc = Hc[np.diff(Hc.indptr) >= 2].tocsr(); Hc.sort_indices()
    rows = []
    for _, c in cells[cells.ds == ds].iterrows():
        h = DC.exact_hgp(Hc, D.partition(c.method), D.group(c.attribute))
        rows.append(dict(method=c.method, dataset=c.dataset, attribute=c.attribute, HGP_components=str(h),
                         equals_true=h == Fraction(c.HGP_exact)))
    return pd.DataFrame(rows)


def main():
    global WITNESS_DIR
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--skip", nargs="*", default=[], choices=["declared", "witnesses", "exchange", "policy", "components"])
    p.add_argument("--force", action="store_true", help="recompute cached steps")
    p.add_argument("--witness-dir", type=Path, default=WITNESS_DIR,
                   help="witnesses to check (default: the shipped data/witnesses; run_search.py writes outputs/witnesses)")
    a = p.parse_args()
    WITNESS_DIR = a.witness_dir
    out = a.out
    allc = pd.read_csv(out / "cells.csv")
    cells = allc[~allc.apply(lambda r: (r.ds, r.attribute) in dmod.SCREENED, axis=1)].reset_index(drop=True)
    assert len(cells) == 75 or a.skip, f"expected the 75 cells, found {len(cells)}"
    C = Cells(cells)
    rq = out / "rq1"

    # ---------------- tables
    t3 = cached(out / "table3.csv", lambda: table3(a, cells), a.force)
    say("Table 3", "\n" + t3.to_string(index=False))
    t5 = table5(C); t5.to_csv(out / "table5.csv", index=False)
    say("Table 5", "\n" + t5.to_string(index=False))
    t6 = table6(C); t6.to_csv(out / "table6.csv", index=False)
    cols = ["budget", "fail", "miss_Dy", "miss_NC", "miss_Rec", "miss_U_P", "fa_Dy", "fa_NC", "fa_Rec", "UP_certifies", "requests"]
    say("Table 6", "\n" + t6[cols].to_string(index=False))
    f4 = figure4(C); f4.to_csv(out / "figure4.csv", index=False)

    d = cells.dataset
    passB = [C.h[i] <= C.B[i] for i in range(C.n)]
    fails = [i for i in range(C.n) if not passB[i]]

    # ---------------- RQ1 bullet 1: same-group exchanges
    if "exchange" not in a.skip:
        ex = cached(rq / "exchange.csv", lambda: exchange_step(a, cells), a.force)
        ok = ex.moves_same_group_cross_cluster.all() and ex.cluster_audits_identical.all() and ex.final_matches_trajectory.all()
        pb = ex[ex.pass_B]
        say("RQ1-1 witnesses replayed exactly (same group, cross cluster, every cluster-level audit unchanged)", bool(ok))
        say("RQ1-1 passing cells made to fail B by same-group exchanges", f"{int(pb.fails_B_reached.sum())} of {len(pb)}")
        say("RQ1-1 median share of nodes reassigned at the first crossing of B",
            f"{100 * pb.moved_share_B.median():.1f}%")
        say("RQ1-1 median HGP increase over all cells", round(float(ex.delta.median()), 2))

    # ---------------- RQ1 bullet 2: declared unit sizes and Figure 3
    if "declared" not in a.skip:
        l0 = cached(rq / "declared_L0.csv", lambda: declared_step(a, cells), a.force)
        m0 = cells.merge(l0, on=["method", "dataset", "attribute"])
        und = [Fraction(lo) <= Fraction(float(B)) < Fraction(hi) for lo, hi, B in zip(m0.L0_low, m0.L0_high, m0.B)]
        say("RQ1-2 undetermined with unit sizes declared (exact extremes)", f"{sum(und)} of {len(m0)}")
        say("RQ1-2 L0 median range width", round(float(np.median([float(Fraction(h) - Fraction(l)) for l, h in zip(m0.L0_low, m0.L0_high)])), 3))
    if "witnesses" not in a.skip:
        w = cached(rq / "declared_witnesses.csv", lambda: witness_step(a, cells), a.force)
        m1 = cells.merge(w, on=["method", "dataset", "attribute"])
        verified = all(m1[f"{r}_{s}_verified"].all() for r in ("L1", "L2") for s in ("low", "high"))
        say("RQ1-2 stored compatible inputs verified (m, every unit size, clique, multiplicity; degrees for L2)",
            f"{bool(verified)} ({4 * len(m1)} witnesses)")
        rows = []
        for _, r in m1.iterrows():
            h = Fraction(r.HGP_exact)
            band = {}
            for rung, src in (("L1", ("L1", "L2")), ("L2", ("L2",))):
                lo = min([h] + [Fraction(r[f"{x}_low"]) for x in src])
                hi = max([h] + [Fraction(r[f"{x}_high"]) for x in src])
                band[rung] = (lo, hi)
            B = Fraction(float(r.B))
            o = dict(method=r.method, dataset=r.dataset, attribute=r.attribute, HGP=float(h), B=float(B),
                     L1_low=float(band["L1"][0]), L1_high=float(band["L1"][1]),
                     undetermined_L1=band["L1"][0] <= B < band["L1"][1], undetermined_L2=band["L2"][0] <= B < band["L2"][1],
                     pass_B=h <= B)
            for f in FACTORS:
                o[f"undetermined_L1_x{f}"] = band["L1"][0] <= Fraction(f) * B < band["L1"][1]
            if "B_pm" in r and not pd.isna(r.B_pm):
                o["undetermined_L1_Bpm"] = band["L1"][0] <= Fraction(float(r.B_pm)) < band["L1"][1]
            rows.append(o)
        f3 = pd.DataFrame(rows); f3.to_csv(out / "figure3.csv", index=False)
        per = ", ".join(f"{ds} {int(f3[f3.dataset == ds].undetermined_L1.sum())} of {int((f3.dataset == ds).sum())}"
                        for ds in DS_ORDER if ds in set(f3.dataset))
        say("Figure 3 undetermined with number and sizes of units declared", per)
        say("RQ1-2 total (Figure 3)", f"{int(f3.undetermined_L1.sum())} of {len(f3)}")
        ff = f3[~f3.pass_B]
        say("RQ1-2 failing cells undetermined under L1 / with every degree also declared (L2)",
            f"{int(ff.undetermined_L1.sum())} / {int(ff.undetermined_L2.sum())} of {len(ff)}")
        v = [int(f3[f"undetermined_L1_x{f}"].sum()) for f in FACTORS[:6]]
        say("RQ1-2 L1 undetermined with B scaled by 0.6..1.1", f"{v} (min {min(v)})")
        if "undetermined_L1_Bpm" in f3:
            say("RQ1-2 L1 undetermined under partition-calibrated budgets", int(f3.undetermined_L1_Bpm.sum()))

    # ---------------- RQ1 bullet 3: dyadic surrogate
    say("RQ1-3 dyadic surrogate understates true HGP", f"{sum(C.h[i] > C.sur[i] for i in range(C.n))} of {C.n}, "
        f"on {len(set(d[i] for i in range(C.n) if C.h[i] > C.sur[i]))} datasets")
    if "policy" not in a.skip:
        po = cached(rq / "policy.csv", lambda: policy_step(a, cells), a.force)
        mp = cells.merge(po, on=["method", "dataset", "attribute"])
        best = np.fmax(mp.G_greedy.to_numpy(), mp.G_exact.fillna(-1).to_numpy())
        ov = mp.dataset.isin(OVERLAP).to_numpy()
        say("RQ1-3 worst invitation policy found creates an average gap above HGP_sur",
            f"{int((best > mp.HGP_sur.to_numpy())[ov].sum())} of {int(ov.sum())} cells with overlapping units")
        say("RQ1-3 Prop. 1 holds for every policy evaluated (gap <= HGP)", bool((best <= mp.HGP.to_numpy() + 1e-12).all()))
    if "components" not in a.skip:
        cc = cached(rq / "acs_components.csv", lambda: components_step(a, cells), a.force)
        say("RQ1-3 ACS-PUMS-HH: connected components of the clique give HGP exactly", f"{int(cc.equals_true.sum())} of {len(cc)}")
    say("RQ1-3 misses at B: dyadic / normalized clique", f"{t6.miss_Dy[0]} / {t6.miss_NC[0]} of {len(fails)} failing cells, "
        f"datasets {sorted(set(d[i] for i in fails))}")
    lo, hi = cells.B_CI_lo.to_numpy(), cells.B_CI_hi.to_numpy()
    above = [i for i in fails if C.h[i] > Fraction(float(hi[i])) and C.sur[i] <= Fraction(float(lo[i]))]
    say("RQ1-3 misses with HGP_true above the bootstrap interval of B and HGP_sur at or below it", f"{len(above)} of {len(fails)}")
    s06 = scaled(C, "0.6")
    f06 = [i for i in range(C.n) if C.h[i] > Fraction(3, 5) * C.B[i]]
    dy = sorted({d[i] for i in f06 if C.sur[i] <= Fraction(3, 5) * C.B[i]})
    nc = sorted({d[i] for i in f06 if C.nc[i] <= Fraction(3, 5) * C.B[i]})
    say("RQ1-3 B x 0.6: failing cells / datasets with failing cells", f"{s06['fail']} / {len({d[i] for i in f06})}")
    say("RQ1-3 B x 0.6: datasets where the dyadic / normalized-clique surrogate misses", f"{dy} / {nc}")

    # ---------------- RQ1 bullet 4: dissimilarity scale
    err_ds = set()
    for _, r in f4.iterrows():
        if any((r[f"{k}_misses"] or 0) + (r[f"{k}_false_alarms"] or 0) > 0 for k in ("dyadic", "normalized_clique", "reconstruction")):
            err_ds.add(r.dataset)
    say("RQ1-4 datasets with overlapping units where a pairwise auditor errs at some tau", f"{sorted(err_ds & set(OVERLAP))} ({len(err_ds & set(OVERLAP))} of 5)")
    tot = f4.groupby("tau")[["reconstruction_misses", "dyadic_false_alarms", "normalized_clique_false_alarms"]].sum()
    say("RQ1-4 reconstruction misses per tau 0.1..0.7", tot.reconstruction_misses.astype(int).tolist())
    say("RQ1-4 dyadic / normalized-clique false alarms per tau 0.1..0.7",
        f"{tot.dyadic_false_alarms.astype(int).tolist()} / {tot.normalized_clique_false_alarms.astype(int).tolist()}")

    # ---------------- RQ1 bullet 5: fixed 0.1
    t = Fraction(1, 10)
    f01 = [i for i in range(C.n) if C.h[i] > t]
    say("RQ1-5 failing at 0.1 / includes all failing at B", f"{len(f01)} of {C.n} / {set(fails) <= set(f01)}")
    say("RQ1-5 0.1 misses by dataset: dyadic / normalized clique / reconstruction",
        f"{sorted({d[i] for i in f01 if C.sur[i] <= t})} / {sorted({d[i] for i in f01 if C.nc[i] <= t})} / "
        f"{sorted({d[i] for i in f01 if C.rec[i] is not None and C.rec[i] <= t})}")
    if "mean_worst_gap" in cells:
        mg = cells.mean_worst_gap.to_numpy()
        say("RQ1-5 failing cells at 0.1 with a membership-weighted average of M_e above 0.1",
            f"{int(sum(mg[i] > 0.1 for i in f01))} of {len(f01)}")

    # ---------------- RQ1 bullet 6: reconstruction
    for ds in ("Pokec", "IMDb-Cast"):
        x = cells[(cells.dataset == ds) & cells.HGP_rec.notna()]
        say(f"RQ1-6 reconstruction minus true HGP on {ds}", f"{(x.HGP_rec - x.HGP).min():+.2f} to {(x.HGP_rec - x.HGP).max():+.2f}")
    x = cells[cells.dataset.isin(["Pokec", "IMDb-Cast"]) & cells.HGP_rec.notna()]
    say("RQ1-6 reconstruction minus true HGP on Pokec and IMDb-Cast", f"{(x.HGP_rec - x.HGP).min():+.2f} to {(x.HGP_rec - x.HGP).max():+.2f}")
    fa = [f"{cells.method[i]}/{d[i]}/{cells.attribute[i]}" for i in range(C.n) if passB[i] and C.rec[i] is not None and C.rec[i] > C.B[i]]
    fas = [f"{cells.method[i]}/{d[i]}/{cells.attribute[i]}" for i in range(C.n)
           if passB[i] and C.rec[i] is not None and C.B_rec[i] is not None and C.rec[i] > C.B_rec[i]]
    say("RQ1-6 reconstruction false alarms at B", fa)
    say("RQ1-6 reconstruction false alarms, self-calibrated", fas)
    say("RQ1-6 reconstruction status on ML-1M", cells[cells.dataset == "ML-1M"].recon_status.iloc[0])

    # ---------------- RQ1 bullet 7: shares of memberships in split units
    ml = cells[(cells.dataset == "ML-1M") & cells.method.isin(["GRAC", "CE+Fair-Louvain"])]
    sh = ml.groupby("method").split_share.first()
    say("RQ1-7 ML-1M share of memberships in split units, GRAC / CE+Fair-Louvain",
        f"{sh['GRAC']:.3f} / {sh['CE+Fair-Louvain']:.3f}; K {int(ml[ml.method == 'GRAC'].K.iloc[0])} / "
        f"{int(ml[ml.method == 'CE+Fair-Louvain'].K.iloc[0])}")
    for attr in ("gender", "age_bucket"):
        y = ml[ml.attribute == attr].set_index("method").HGP
        say(f"RQ1-7 ML-1M HGP ratio CE+Fair-Louvain / GRAC, {attr}", round(float(y["CE+Fair-Louvain"] / y["GRAC"]), 2))

    (out / "rq1.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"wrote {out / 'rq1.txt'}")


if __name__ == "__main__":
    main()
