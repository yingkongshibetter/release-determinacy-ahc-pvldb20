"""Compute every statistic of the 75 cells and their budgets.

Steps (each is skipped when its output exists, unless --force):
  1. maximal-clique reconstruction of each dataset   -> outputs/reconstruction/<dataset>.{npz,json}
  2. shared and self-calibrated budgets (R = 200)    -> outputs/budgets.csv
  3. partition-calibrated budgets (with --partition-calibrated) -> outputs/budgets_partition.csv
  4. per-cell statistics, merged with the budgets    -> outputs/cells.csv

Usage:
  python run_cells.py                         # all six datasets
  python run_cells.py --datasets OA-DB ACS-PUMS-HH
  python run_cells.py --screened              # also race and disability of ACS-PUMS-HH (Appendix B)
"""
from __future__ import annotations

import time

import pandas as pd

from ahc_audit import data as dmod
from ahc_audit.auditors import maximal_cliques, save_reconstruction
from ahc_audit.budgets import partition_calibrated, shared_budget_stream
from ahc_audit.cells import Context
from options import add_datasets, base_parser


def log(msg):
    print(msg, flush=True)


def main():
    p = base_parser(__doc__.splitlines()[0])
    add_datasets(p)
    p.add_argument("--screened", action="store_true", help="include race and disability of ACS-PUMS-HH")
    p.add_argument("--partition-calibrated", action="store_true", help="also compute B_pm (about 10 min)")
    p.add_argument("--skip-reconstruction", action="store_true", help="do not build maximal-clique reconstructions")
    p.add_argument("--recon-datasets", nargs="+", default=None,
                   help="datasets to reconstruct (default: all requested; ML-1M aborts at the 20M-clique cap "
                        "after about 4 min, IMDb-Cast takes about 6 min)")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    rdir = out / "reconstruction"
    cache = {}

    def load(ds):
        if ds not in cache:
            cache.clear()
            cache[ds] = dmod.load_dataset(ds, a.data_root, a.partitions)
        return cache[ds]

    # 1. reconstruction
    if not a.skip_reconstruction:
        for ds in a.recon_datasets or a.datasets:
            ds = dmod.INTERNAL.get(ds, ds)
            if (rdir / f"{ds}.json").exists() and not a.force:
                continue
            log(f"[{dmod.DISPLAY[ds]}] maximal-clique reconstruction")
            Hr, info = maximal_cliques(load(ds).H, log=log)
            save_reconstruction(rdir / ds, Hr, info)
            log(f"  {info}")

    contexts = {}

    def ctx(ds):
        if ds not in contexts:
            contexts.clear()
            contexts[ds] = Context(load(ds), rdir)
        return contexts[ds]

    # 2. shared and self-calibrated budgets
    bfile = out / "budgets.csv"
    if a.force or not bfile.exists() or not set(dmod.DISPLAY[d] for d in a.datasets) <= set(pd.read_csv(bfile).dataset):
        t0 = time.perf_counter()
        log("shared and self-calibrated budgets (one random stream, default_rng(2026))")
        rows = list(shared_budget_stream(load, lambda D, attr: ctx(D.name).reference_stats(attr),
                                         datasets=a.datasets, include_screened=a.screened, log=log))
        b = pd.DataFrame(rows)
        b["dataset"] = b.dataset.map(dmod.DISPLAY)
        b.to_csv(bfile, index=False)
        log(f"  wrote {bfile} ({time.perf_counter() - t0:.0f}s)")
    budgets = pd.read_csv(bfile)

    # 3. partition-calibrated budgets
    pfile = out / "budgets_partition.csv"
    if a.partition_calibrated and (a.force or not pfile.exists()):
        rows = []
        for ds in a.datasets:
            D = load(ds)
            attrs = [x for x in dmod.ATTRIBUTES[ds] if (ds, x) not in dmod.SCREENED]
            for meth in dmod.METHODS:
                t0 = time.perf_counter()
                for attr, v in partition_calibrated(D, meth, attrs).items():
                    rows.append(dict(method=meth, dataset=D.display, attribute=attr, **v))
                log(f"  B_pm {meth:16s} {D.display:12s} ({time.perf_counter() - t0:.0f}s)")
        pd.DataFrame(rows).to_csv(pfile, index=False)

    # 4. cells
    rows = []
    for ds, meth, attr in dmod.cells(a.datasets, include_screened=a.screened):
        t0 = time.perf_counter()
        r = ctx(ds).cell(meth, attr)
        r["t_cell_s"] = time.perf_counter() - t0
        rows.append(r)
        log(f"  {r['dataset']:12s} {meth:16s} {attr:16s} HGP {r['HGP']:.6f} sur {r['HGP_sur']:.6f} "
            f"U_P {r['U_P']:.6f} rec {r['HGP_rec']:.6f}")
    c = pd.DataFrame(rows).merge(budgets, on=["dataset", "attribute"], how="left", validate="many_to_one")
    if pfile.exists():
        c = c.merge(pd.read_csv(pfile), on=["method", "dataset", "attribute"], how="left", validate="one_to_one")
    c.to_csv(out / "cells.csv", index=False)
    log(f"wrote {out / 'cells.csv'} ({len(c)} cells)")


if __name__ == "__main__":
    main()
