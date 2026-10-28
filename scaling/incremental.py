"""Incremental maintenance of exact HGP (producer or trusted third party) and U_P (auditor) under node moves.

usage: python scaling/incremental.py <dataset> [--moves 100000] [--trace-moves 20000] [--checkpoint 10000]
Start: the AHRC partition of the dataset; all audited attributes at once.
(i)  producer: per-unit counts n_e(g,c), n_e(c) (n_e(g) is static); a move of v from C1 to C2 updates the counts
     of the units that contain v and recomputes Y_e = |e|^2 psi_e only there. HGP is kept exactly as
     N / (L vol), N = sum_e Y_e L/|e| (Python integer), L = lcm of the unit sizes.
(ii) auditor: holds P (CSR, float64); the crossing mass is updated from row v of P only.
Streams: random single-node moves (uniform node, uniform new cluster), and a local-moving trace (node moved to
the cluster with the largest P-weight to it). Checkpoints compare with full recomputation (exact Fraction).
Outputs: outputs/scaling/incremental/summary.csv, <dataset>_<stream>_latency.csv, <dataset>_checkpoints.csv.
"""
from __future__ import annotations

import argparse
import math
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import clk, np, pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from ahc_audit.hgp import hgp_exact  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ds")
ap.add_argument("--moves", type=int, default=100000)
ap.add_argument("--trace-moves", type=int, default=20000)
ap.add_argument("--checkpoint", type=int, default=10000)
ap.add_argument("--data-root", default=None)
args = ap.parse_args()
ds = args.ds
OUT = C.OUT / "incremental"
OUT.mkdir(parents=True, exist_ok=True)

attrs = [a for a in C.dmod.ATTRIBUTES[ds] if (ds, a) not in C.dmod.SCREENED]
H, unit, node = C.load_incidence(ds, args.data_root)
m, n = H.shape
vol = int(unit.size)
size = np.diff(H.indptr).astype(np.int64)
G = C.load_attr_codes(ds, attrs, n, args.data_root)
cl0 = C.dmod.read_partition(C.dmod.partition_path(ds, "AHRC")).astype(np.int64)
clusters = np.unique(cl0)
KC = int(cl0.max()) + 1
Lcm = 1
for s in np.unique(size):
    Lcm = Lcm * int(s) // math.gcd(Lcm, int(s))
Hc = H.tocsc()
units_of = [Hc.indices[Hc.indptr[v]:Hc.indptr[v + 1]].tolist() for v in range(n)]
deg = np.diff(Hc.indptr)
active = np.where(deg > 0)[0]
A = len(attrs)
Glist = [G[a].tolist() for a in attrs]
sz = size.tolist()
Lk = [Lcm // s for s in sz]
print(f"{ds}: n={n} m={m} vol={vol} attrs={attrs} K={clusters.size} L={Lcm}", flush=True)

# auditor's release P (symmetric CSR incl. diagonal), built from incidence (producer side, untimed here)
Hf = H.astype(np.float64)
Dinv = sp.diags(1.0 / size.astype(np.float64))
P = (Hf.T @ Dinv @ Hf).tocsr()
P.sort_indices()
Pip, Pix, Pw = P.indptr, P.indices, P.data
Pu = sp.triu(P, k=1).tocoo()
pu_r, pu_c, pu_w = Pu.row, Pu.col, Pu.data
garr = [G[a] for a in attrs]


def full_up(cl):
    out = []
    for ai in range(A):
        g = garr[ai]
        msk = (g[pu_r] != g[pu_c]) & (cl[pu_r] != cl[pu_c])
        out.append(2.0 * float(pu_w[msk].sum()) / vol)
    return out


def full_hgp(cl):
    return [hgp_exact(H, cl, garr[ai]) for ai in range(A)]


class Producer:
    def __init__(self, cl):
        self.cl = cl.tolist()
        self.nc = [dict() for _ in range(m)]
        self.cell = [[dict() for _ in range(m)] for _ in range(A)]
        self.ng = [[dict() for _ in range(m)] for _ in range(A)]
        ip, ix = H.indptr, H.indices
        for e in range(m):
            nce = self.nc[e]
            for v in ix[ip[e]:ip[e + 1]].tolist():
                c = self.cl[v]
                nce[c] = nce.get(c, 0) + 1
                for ai in range(A):
                    g = Glist[ai][v]
                    d = self.ng[ai][e]; d[g] = d.get(g, 0) + 1
                    key = g * KC + c
                    d = self.cell[ai][e]; d[key] = d.get(key, 0) + 1
        self.Y = [[self.y(ai, e) for e in range(m)] for ai in range(A)]
        self.N = [sum(self.Y[ai][e] * Lk[e] for e in range(m)) for ai in range(A)]

    def y(self, ai, e):
        s = sz[e]; nge = self.ng[ai][e]; nce = self.nc[e]; tot = 0
        for key, cnt in self.cell[ai][e].items():
            d = s * cnt - nge[key // KC] * nce[key % KC]
            if d > 0:
                tot += d
        return tot

    def move(self, v, c2):
        c1 = self.cl[v]
        self.cl[v] = c2
        for e in units_of[v]:
            nce = self.nc[e]
            x = nce[c1] - 1
            if x:
                nce[c1] = x
            else:
                del nce[c1]
            nce[c2] = nce.get(c2, 0) + 1
            for ai in range(A):
                g = Glist[ai][v]
                d = self.cell[ai][e]
                k1 = g * KC + c1; x = d[k1] - 1
                if x:
                    d[k1] = x
                else:
                    del d[k1]
                k2 = g * KC + c2; d[k2] = d.get(k2, 0) + 1
                yn = self.y(ai, e)
                Ye = self.Y[ai]
                self.N[ai] += (yn - Ye[e]) * Lk[e]
                Ye[e] = yn

    def hgp(self):
        return [Fraction(self.N[ai], Lcm * vol) for ai in range(A)]


class Auditor:
    def __init__(self, cl):
        self.cl = cl.copy()
        self.X = [ux * vol / 2.0 for ux in full_up(cl)]

    def move(self, v, c2):
        c1 = self.cl[v]
        lo, hi = Pip[v], Pip[v + 1]
        nb = Pix[lo:hi]; w = Pw[lo:hi]
        cn = self.cl[nb]
        for ai in range(A):
            g = garr[ai]
            mg = g[nb] != g[v]
            self.X[ai] += float(w[mg & (cn != c2)].sum()) - float(w[mg & (cn != c1)].sum())
        self.cl[v] = c2   # diagonal entry: g[v]==g[v], never crossing, so self-row handling is exact

    def up(self):
        return [2.0 * x / vol for x in self.X]


def make_random_stream(k, seed):
    rng = np.random.default_rng(seed)
    vs = rng.choice(active, size=k)
    cs = rng.choice(clusters, size=k)
    return vs, cs


def run_stream(name, cl_start, vs, cs, valid):
    """valid(v,c,cur_cl) True -> apply; random stream re-draws c when equal to current."""
    t0 = clk(); prod = Producer(cl_start); t_init_p = clk() - t0
    t0 = clk(); aud = Auditor(cl_start); t_init_a = clk() - t0
    cur = cl_start.copy()
    lat_p = np.zeros(len(vs)); lat_a = np.zeros(len(vs))
    tot_p = tot_a = 0.0
    rows = []
    cps = []
    for i, (v, c2) in enumerate(zip(vs, cs)):
        v = int(v); c2 = int(c2)
        t0 = clk(); prod.move(v, c2); h = prod.hgp(); t1 = clk()
        aud.move(v, c2); u = aud.up(); t2 = clk()
        lat_p[i] = t1 - t0; lat_a[i] = t2 - t1
        cur[v] = c2
        if (i + 1) % args.checkpoint == 0 or i + 1 == len(vs):
            ta = clk(); hf = full_hgp(cur); tb = clk(); uf = full_up(cur); tc = clk()
            ok_h = all(hf[ai] == h[ai] for ai in range(A))
            err_u = max(abs(uf[ai] - u[ai]) for ai in range(A))
            assert ok_h, ("incremental HGP != full recomputation", name, i)
            assert prod.cl == cur.tolist() and (aud.cl == cur).all()
            cp = {"dataset": ds, "stream": name, "moves_done": i + 1, "hgp_exact_match": int(ok_h),
                  "up_max_abs_err": err_u, "t_full_hgp_s": tb - ta, "t_full_up_s": tc - tb,
                  **{f"HGP_{a}": float(h[ai]) for ai, a in enumerate(attrs)},
                  **{f"U_P_{a}": u[ai] for ai, a in enumerate(attrs)}}
            C.append_csv(OUT / f"{ds}_checkpoints.csv", cp)
            cps.append(cp)
            print(f"  {name} {i+1}: exact match={ok_h} U_P err={err_u:.2e} full HGP {tb-ta:.3f}s "
                  f"full U_P {tc-tb:.4f}s; mean move {lat_p[:i+1].mean()*1e6:.1f}us/{lat_a[:i+1].mean()*1e6:.1f}us",
                  flush=True)
    d = deg[vs.astype(np.int64)]
    pd.DataFrame({"move": np.arange(len(vs)), "v": vs, "deg": d, "lat_hgp_us": lat_p * 1e6,
                  "lat_up_us": lat_a * 1e6}).to_csv(OUT / f"{ds}_{name}_latency.csv", index=False)
    full_h = float(np.median([c["t_full_hgp_s"] for c in cps]))
    full_u = float(np.median([c["t_full_up_s"] for c in cps]))
    summ = {"dataset": ds, "stream": name, "moves": len(vs), "attrs": A, "n": n, "m": m, "vol": vol,
            "t_init_producer_s": t_init_p, "t_init_auditor_s": t_init_a,
            "hgp_throughput_moves_per_s": len(vs) / lat_p.sum(), "up_throughput_moves_per_s": len(vs) / lat_a.sum(),
            "hgp_lat_median_us": float(np.median(lat_p) * 1e6), "hgp_lat_p99_us": float(np.quantile(lat_p, .99) * 1e6),
            "hgp_lat_max_us": float(lat_p.max() * 1e6),
            "up_lat_median_us": float(np.median(lat_a) * 1e6), "up_lat_p99_us": float(np.quantile(lat_a, .99) * 1e6),
            "up_lat_max_us": float(lat_a.max() * 1e6),
            "full_hgp_s_median": full_h, "full_up_s_median": full_u,
            "speedup_hgp_median": full_h / float(np.median(lat_p)), "speedup_up_median": full_u / float(np.median(lat_a)),
            "checkpoints": len(cps), "all_exact": int(all(c["hgp_exact_match"] for c in cps)),
            "up_max_abs_err": max(c["up_max_abs_err"] for c in cps), "deg_mean_moved": float(d.mean()),
            "deg_max_moved": int(d.max())}
    C.append_csv(OUT / "summary.csv", summ)
    print(summ, flush=True)


for f in (OUT / f"{ds}_checkpoints.csv",):
    if f.exists():
        f.unlink()

# random stream
vs, cs = make_random_stream(args.moves, 4242)
cur = cl0.copy()
for i in range(len(vs)):            # redraw targets equal to current cluster (sequentially consistent)
    if cs[i] == cur[vs[i]]:
        cs[i] = clusters[(np.searchsorted(clusters, cs[i]) + 1) % clusters.size]
    cur[vs[i]] = cs[i]
run_stream("random", cl0, vs, cs, None)

# local-moving trace (generated first, untimed)
if args.trace_moves > 0:
    rng = np.random.default_rng(777)
    cur = cl0.copy(); tv, tc = [], []
    tries = 0
    while len(tv) < args.trace_moves and tries < 50 * args.trace_moves:
        tries += 1
        v = int(rng.choice(active))
        lo, hi = Pip[v], Pip[v + 1]
        nb = Pix[lo:hi]; w = Pw[lo:hi]
        keep = nb != v
        if not keep.any():
            continue
        cw = np.bincount(cur[nb[keep]], weights=w[keep], minlength=KC)
        c2 = int(np.argmax(cw))
        if c2 != cur[v]:
            tv.append(v); tc.append(c2); cur[v] = c2
    print(f"trace: {len(tv)} effective moves from {tries} tries", flush=True)
    run_stream("localmove", cl0, np.array(tv), np.array(tc), None)
print("done", ds)
