"""Local search for compatible inputs under the declarations L1 and L2 (RQ1, Figure 3).

Every move keeps the node set, attributes, partition, the number of units m, the size of every unit (hence the
size histogram and the volume), and the unweighted clique (the set of node pairs covered by >= 1 unit). No move
creates a unit that already exists in the current hypergraph, so the number of repeated units and the largest
multiplicity can only go down. L2 moves also keep every node degree.

Moves
  SWAP(e, u, w)       e -> e - u + w, w not in e, w adjacent in the clique to every member of e - u, and every pair
                      {u, x}, x in e - u, covered by >= 2 units before the move.                         (L1)
  RELOCATE(e, f, S)   e -> S, S a k-subset of a unit f with |f| > k = |e|, allowed when every pair of e is covered by
                      >= 2 units before the move (all pairs of S lie in f, hence in the clique).          (L1)
  EXCHANGE(e1,u,e2,w) SWAP(e1, u, w) followed by SWAP(e2, w, u) on the updated counts, u in e1 - e2, w in e2 - e1,
                      each half valid on the counts it sees; u and w each leave one unit and join one.    (L2)
The objective is the volume-weighted sum of Y_e/|e|; with the volume fixed it is vol * HGP. Search values are
floats; the exact HGP of a stored witness comes from declared.verify_witness.
"""
from __future__ import annotations

import time

import numpy as np

from . import declared as K


class State:
    def __init__(self, units, n, keys, cnt, A, gcode, ccode, l2):
        self.units = [e.copy() for e in units]
        self.n = n; self.keys = keys; self.cnt = cnt.copy(); self.A = A
        self.g = gcode; self.c = ccode
        self.GN = int(gcode.max()) + 1; self.CN = int(ccode.max()) + 1  # codes for a group / cluster new to a unit
        self.pidx = [K.unit_pair_index(e, keys, n) for e in self.units]
        self.ukeys = {}
        for e in self.units:
            t = tuple(e.tolist()); self.ukeys[t] = self.ukeys.get(t, 0) + 1
        self.Y = np.array([K.Y_rows(gcode[e][None, :], ccode[e][None, :])[0] / e.size for e in self.units])
        self.total = float(self.Y.sum())
        self.vol = int(sum(e.size for e in self.units))
        self.l2 = l2
        if l2:
            self.nu = [set() for _ in range(n)]
            for i, e in enumerate(self.units):
                for v in e.tolist():
                    self.nu[v].add(i)
        self.changed = np.zeros(len(self.units), dtype=bool)
        self.moves = {"swap": 0, "relocate": 0, "exchange": 0}

    # ---------- primitives
    def adj(self, a, b):
        """vectorized: are pairs (a_i, b_i) in the clique."""
        lo = np.minimum(a, b); hi = np.maximum(a, b)
        pk = lo * self.n + hi
        idx = np.searchsorted(self.keys, pk)
        idx = np.minimum(idx, self.keys.size - 1)
        return self.keys[idx] == pk

    def removable(self, i):
        e = self.units[i]; k = e.size
        iu, iv = K.triu(k)
        cv = self.cnt[self.pidx[i]]
        M = np.full((k, k), 1 << 40, dtype=np.int64)
        M[iu, iv] = cv; M[iv, iu] = cv
        return M.min(axis=1) >= 2, bool((cv >= 2).all())

    def set_unit(self, i, new):
        """Replace unit i by the sorted member array `new` (validity checked by the caller)."""
        old = self.units[i]
        np.subtract.at(self.cnt, self.pidx[i], 1)
        pi = K.unit_pair_index(new, self.keys, self.n)
        np.add.at(self.cnt, pi, 1)
        to = tuple(old.tolist()); tn = tuple(new.tolist())
        self.ukeys[to] -= 1
        if self.ukeys[to] == 0:
            del self.ukeys[to]
        self.ukeys[tn] = self.ukeys.get(tn, 0) + 1
        if self.l2:
            for v in np.setdiff1d(old, new).tolist():
                self.nu[v].discard(i)
            for v in np.setdiff1d(new, old).tolist():
                self.nu[v].add(i)
        self.units[i] = new; self.pidx[i] = pi
        y = K.Y_rows(self.g[new][None, :], self.c[new][None, :])[0] / new.size
        self.total += y - self.Y[i]; self.Y[i] = y
        self.changed[i] = True

    def canon(self, ge, ce, w):
        gw = self.g[w]; cw = self.c[w]
        gw = np.where(np.isin(gw, ge), gw, self.GN); cw = np.where(np.isin(cw, ce), cw, self.CN)
        return gw, cw

    # ---------- swap candidates of one unit: arrays (pos i, w, gain)
    def swap_candidates(self, i, rem):
        e = self.units[i]; k = e.size
        A = self.A
        nb = np.concatenate([A.indices[A.indptr[x]:A.indptr[x + 1]] for x in e.tolist()])
        if nb.size == 0:
            return None
        cand, cc = np.unique(nb, return_counts=True)
        sel = cc >= k - 1
        cand, cc = cand[sel], cc[sel]
        notin = ~np.isin(cand, e)
        cand, cc = cand[notin], cc[notin]
        if cand.size == 0:
            return None
        ri = np.where(rem)[0]
        full = cand[cc == k]
        part = cand[cc == k - 1]
        P = []; W = []
        if full.size and ri.size:
            P.append(np.repeat(ri, full.size)); W.append(np.tile(full, ri.size))
        if part.size:
            ad = self.adj(np.repeat(part, k), np.tile(e, part.size)).reshape(part.size, k)
            miss = np.argmin(ad, axis=1)  # exactly one False per row (count k-1)
            ok = rem[miss]
            if ok.any():
                P.append(miss[ok]); W.append(part[ok])
        if not P:
            return None
        P = np.concatenate(P); W = np.concatenate(W)
        ge = self.g[e]; ce = self.c[e]
        gw, cw = self.canon(ge, ce, W)
        # combos: (label of u at P, canonical label of w)
        lu = ge[P] * (self.CN + 1) + ce[P]
        lw = gw * (self.CN + 1) + cw
        key = lu * ((self.GN + 1) * (self.CN + 1)) + lw
        uk, rep, inv = np.unique(key, return_index=True, return_inverse=True)
        M = uk.size
        Gr = np.tile(ge, (M, 1)); Cr = np.tile(ce, (M, 1))
        rr = np.arange(M)
        Gr[rr, P[rep]] = gw[rep]; Cr[rr, P[rep]] = cw[rep]
        ynew = K.Y_rows(Gr, Cr) / k
        gain = ynew[inv.ravel()] - self.Y[i]
        return P, W, gain

    def try_apply_swap(self, i, P, W, gain, sign, rng, eps):
        """Apply the best valid swap (by sign*gain); return gain applied or 0."""
        sg = sign * gain
        perm = rng.permutation(sg.size)
        order = perm[np.argsort(-sg[perm], kind="stable")]
        e = self.units[i]
        for j in order[:64]:
            if sg[j] <= eps:
                break
            # random tie-break among equal gains handled by shuffling within the slice
            new = np.sort(np.append(e[np.arange(e.size) != P[j]], W[j]))
            if tuple(new.tolist()) in self.ukeys:
                continue
            self.set_unit(i, new); self.moves["swap"] += 1
            return gain[j]
        return 0.0

    # ---------- relocation: best k-subset of a sampled larger unit by local search
    def best_subset(self, f, k, sign, rng, iters):
        S = np.sort(rng.choice(f, size=k, replace=False))
        y = K.Y_rows(self.g[S][None, :], self.c[S][None, :])[0] / k
        for _ in range(iters):
            O = np.setdiff1d(f, S)
            ge = self.g[S]; ce = self.c[S]
            gw, cw = self.canon(ge, ce, O)
            P = np.repeat(np.arange(k), O.size); Wi = np.tile(np.arange(O.size), k)
            lu = ge[P] * (self.CN + 1) + ce[P]
            lw = gw[Wi] * (self.CN + 1) + cw[Wi]
            key = lu * ((self.GN + 1) * (self.CN + 1)) + lw
            uk, rep, inv = np.unique(key, return_index=True, return_inverse=True)
            M = uk.size
            Gr = np.tile(ge, (M, 1)); Cr = np.tile(ce, (M, 1)); rr = np.arange(M)
            Gr[rr, P[rep]] = gw[Wi[rep]]; Cr[rr, P[rep]] = cw[Wi[rep]]
            yn = K.Y_rows(Gr, Cr) / k
            j = int(np.argmax(sign * yn))
            if sign * (yn[j] - y) <= 1e-12:
                break
            S = np.sort(np.append(S[np.arange(k) != P[rep[j]]], O[Wi[rep[j]]]))
            y = yn[j]
        return S, y

    def try_relocate(self, i, sign, rng, eps, big_by_size, R, iters):
        e = self.units[i]; k = e.size
        pool = big_by_size.get(k)
        if pool is None or pool.size == 0:
            return 0.0
        best = None
        for f_i in rng.choice(pool, size=min(R, pool.size), replace=False).tolist():
            if f_i == i:
                continue
            f = self.units[f_i]
            if f.size <= k:
                continue
            S, y = self.best_subset(f, k, sign, rng, iters)
            g = y - self.Y[i]
            if sign * g > eps and tuple(S.tolist()) not in self.ukeys and (best is None or sign * g > sign * best[1]):
                best = (S, g)
        if best is None:
            return 0.0
        self.set_unit(i, best[0]); self.moves["relocate"] += 1
        return best[1]

    # ---------- exchange (L2)
    def try_exchange(self, i, P, W, gain, sign, rng, eps, topc, partners):
        e1 = self.units[i]
        sg = sign * gain
        perm = rng.permutation(sg.size)
        order = perm[np.argsort(-sg[perm], kind="stable")][:topc]
        best = None
        for j in order.tolist():
            u = int(e1[P[j]]); w = int(W[j])
            new1 = np.sort(np.append(e1[np.arange(e1.size) != P[j]], w))
            t1 = tuple(new1.tolist())
            if t1 in self.ukeys:
                continue
            cand2 = [x for x in self.nu[w] if x != i]
            if not cand2:
                continue
            if len(cand2) > partners:
                cand2 = rng.choice(np.array(cand2), size=partners, replace=False).tolist()
            # tentative counts after half 1
            pi_old = self.pidx[i]; pi_new = K.unit_pair_index(new1, self.keys, self.n)
            np.subtract.at(self.cnt, pi_old, 1); np.add.at(self.cnt, pi_new, 1)
            for i2 in cand2:
                e2 = self.units[i2]
                if (e2 == u).any():
                    continue
                pos = int(np.where(e2 == w)[0][0])
                rest = e2[np.arange(e2.size) != pos]
                if not self.adj(np.full(rest.size, u), rest).all():
                    continue
                wk = np.minimum(rest, w) * self.n + np.maximum(rest, w)
                wi = np.searchsorted(self.keys, wk)
                if (self.cnt[wi] < 2).any():
                    continue
                new2 = np.sort(np.append(rest, u))
                t2 = tuple(new2.tolist())
                if t2 in self.ukeys or t2 == t1:
                    continue
                y2 = K.Y_rows(self.g[new2][None, :], self.c[new2][None, :])[0] / new2.size
                tot = gain[j] + (y2 - self.Y[i2])
                if sign * tot > eps and (best is None or sign * tot > sign * best[0]):
                    best = (tot, new1, i2, new2)
            np.subtract.at(self.cnt, pi_new, 1); np.add.at(self.cnt, pi_old, 1)
        if best is None:
            return 0.0
        _, new1, i2, new2 = best
        self.set_unit(i, new1); self.set_unit(i2, new2); self.moves["exchange"] += 1
        return best[0]


def run_search(units, n, keys, cnt0, A, gcode, ccode, rung, direction, seed, tcap, R=4, iters=8, topc=12,
               partners=24, eps=1e-9):
    """Hill climbing: passes over the units in random order, each unit takes its best improving move.
    Stops at the time cap or after a pass with no improving move. Returns the final State and a log dict."""
    l2 = rung == "L2"
    st = State(units, n, keys, cnt0, A, gcode, ccode, l2)
    sign = 1.0 if direction == "high" else -1.0
    rng = np.random.default_rng(seed)
    big_by_size = {}
    sizes = np.array([e.size for e in st.units])
    for k in np.unique(sizes).tolist():
        big_by_size[k] = np.where(sizes > k)[0]
    t0 = time.perf_counter()
    passes = 0; trace = [st.total / st.vol]
    done_by = "converged"
    while True:
        passes += 1; improved = 0
        for i in rng.permutation(len(st.units)).tolist():
            if time.perf_counter() - t0 > tcap:
                done_by = "time cap"
                break
            rem, full = st.removable(i)
            if not rem.any():
                continue
            res = st.swap_candidates(i, rem)
            if not l2:
                g = 0.0
                if res is not None:
                    g = st.try_apply_swap(i, *res, sign, rng, eps)
                if g == 0.0 and full:
                    g = st.try_relocate(i, sign, rng, eps, big_by_size, R, iters)
                improved += g != 0.0
            else:
                if res is not None:
                    g = st.try_exchange(i, *res, sign, rng, eps, topc, partners)
                    improved += g != 0.0
        trace.append(st.total / st.vol)
        if done_by == "time cap" or improved == 0:
            break
    return st, dict(passes=passes, stop=done_by, t=time.perf_counter() - t0, trace=trace,
                    moves=dict(st.moves), changed_units=int(st.changed.sum()))
