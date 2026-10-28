"""The two searches of RQ1 that the paper ran for hours; their witnesses ship in data/witnesses/.

  python run_search.py declared --dataset OA-DB [--cells 0 1] [--tcap 300] [--restarts 5]
      compatible inputs under declared number and sizes of units (L1) and also degrees (L2), Figure 3.
      For every cell, rung and direction (high, low): independent hill-climbing runs with seeds
      SeedSequence(20261023, (dataset index, cell index, rung, direction, restart)) until the time cap of the
      four runs or the number of restarts is reached; the run with the most extreme HGP is kept.
      Paper setting: --tcap 300 --restarts 5 on all 65 cells, 14.5 CPU hours (about 2 h on 11 processes).
  python run_search.py exchange --dataset OA-DB [--cells 0 1] [--tcap 300]
      same-group exchanges that raise HGP and keep every cluster-level audit (RQ1, first bullet); needs numba.
      Seeds SeedSequence(20261024, (cell index,)). Paper setting: --tcap 300 on all 75 cells, about 6 CPU hours.
Witnesses are written to outputs/witnesses/{declared,exchange}/ in the format of data/witnesses/; check them
with run_rq1.py --witness-dir outputs/witnesses. Searches stop on a time cap, so a rerun on other hardware
finds other witnesses; the counts of the paper are lower bounds that the shipped witnesses certify.
"""
from __future__ import annotations

import json
import time

import numpy as np

from ahc_audit import data as dmod
from ahc_audit import declared as DC
from ahc_audit import declared_search as S
from ahc_audit import exchange as EX
from ahc_audit.data import METHOD_FILE
from ahc_audit.releases import clique_pair_keys
from options import base_parser, dataset_name


def declared(a, D):
    di = dmod.OVERLAPPING.index(D.name)
    cells = dmod.cells([D.name])
    sel = a.cells if a.cells is not None else list(range(len(cells)))
    H = D.H
    m, n = H.shape
    units = DC.units_of(H)
    keys = clique_pair_keys(H)
    cnt = DC.pair_counts(units, keys, n)
    A = DC.adjacency(keys, n)
    out = a.out / "witnesses" / "declared" / D.name
    for ci in sel:
        ds, meth, attr = cells[ci]
        gc, _ = dmod.compact(D.group(attr)); cc, _ = dmod.compact(D.partition(meth))
        for ri, rung in enumerate(["L1", "L2"]):
            for dj, d in enumerate(["high", "low"]):
                sign = 1 if d == "high" else -1
                t0 = time.perf_counter(); best = None; runs = []
                for r in range(a.restarts):
                    left = a.tcap - (time.perf_counter() - t0)
                    if r > 0 and left < 5:
                        break
                    seed = int(np.random.SeedSequence(20261023, spawn_key=(di, ci, ri, dj, r)).generate_state(1)[0])
                    st, log = S.run_search(units, n, keys, cnt, A, gc, cc, rung, d, seed, left)
                    val = st.total / st.vol
                    runs.append(dict(restart=r, seed=seed, final=val, stop=log["stop"], t=round(log["t"], 2)))
                    if best is None or sign * val > sign * best[0]:
                        best = (val, [e.copy() for e in st.units])
                path = out / f"{METHOD_FILE[meth]}__{attr}__{rung}_{d}.npz"
                DC.save_witness(path, best[1], units)
                print(json.dumps(dict(dataset=D.display, cell=ci, method=meth, attribute=attr, rung=rung, direction=d,
                                      best=best[0], runs=runs)), flush=True)


def exchange(a, D):
    cells = dmod.cells([D.name])
    sel = a.cells if a.cells is not None else list(range(len(cells)))
    out = a.out / "witnesses" / "exchange" / D.name
    out.mkdir(parents=True, exist_ok=True)
    for ci in sel:
        ds, meth, attr = cells[ci]
        idx = dmod.cell_index(ds, meth, attr)
        seed = int(np.random.SeedSequence(EX.SEED_ROOT, spawn_key=(idx,)).generate_state(1)[0])
        su, sv, tr, info = EX.search(D.H, D.group(attr), D.partition(meth), seed, a.tcap)
        np.savez_compressed(out / f"{METHOD_FILE[meth]}__{attr}.npz", swap_u=su, swap_v=sv, traj=tr)
        print(json.dumps(dict(dataset=D.display, cell=idx, method=meth, attribute=attr, **info)), flush=True)


def main():
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("search", choices=["declared", "exchange"])
    p.add_argument("--dataset", type=dataset_name, required=True)
    p.add_argument("--cells", type=int, nargs="+", default=None, help="cell indices within the dataset (paper order)")
    p.add_argument("--tcap", type=float, default=300.0, help="seconds per cell (per rung and direction for declared)")
    p.add_argument("--restarts", type=int, default=5)
    a = p.parse_args()
    D = dmod.load_dataset(a.dataset, a.data_root, a.partitions)
    if a.search == "declared":
        if a.dataset not in dmod.OVERLAPPING:
            raise SystemExit("ACS-PUMS-HH households are declared disjoint; the declared search runs on the other five")
        declared(a, D)
    else:
        exchange(a, D)


if __name__ == "__main__":
    main()
