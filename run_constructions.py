"""The constructions of Sections 2 and 3 in exact rational arithmetic (about 1 minute).

Checks the running example, the CGP example and Remark 1, Lemma 1, Corollaries 1-4, Theorem 3, and the
additive audits that the 3-uniform weighted clique leaves undetermined. Every line ends in OK or FAILED.
Writes outputs/constructions.txt.
"""
from __future__ import annotations

import itertools
import random
from fractions import Fraction as Fr
from itertools import combinations, combinations_with_replacement
from math import ceil, floor

from ahc_audit.criterion import REL, Instance, criterion_type, psi_type, rel_wclique
from options import base_parser

A_, B_, C_, D_ = (0, 0), (0, 1), (1, 0), (1, 1)   # (group, cluster)
LAB4 = [A_, B_, C_, D_]
LINES = []
ALL_OK = [True]


def check(label, ok, detail=""):
    s = f"{label}: {detail} {'OK' if ok else 'FAILED'}".replace("  ", " ")
    print(s, flush=True)
    LINES.append(s)
    ALL_OK[0] &= bool(ok)


def chi(a, b):
    return 1 if (a[0] != b[0] and a[1] != b[1]) else 0


def hgp(units, labels):
    return sum(len(e) * psi_type(tuple(labels[v] for v in e)) for e in units) / sum(len(e) for e in units)


def pair_counts(units):
    pc = {}
    for e in units:
        for p in combinations(sorted(e), 2):
            pc[p] = pc.get(p, 0) + 1
    return pc


def weighted(units, omega):
    w = {}
    for e in units:
        for p in combinations(sorted(e), 2):
            w[p] = w.get(p, 0) + omega(len(e))
    return w


def P_release(units):
    P = {}
    for e in units:
        for u in e:
            for v in e:
                P[(u, v)] = P.get((u, v), 0) + Fr(1, len(e))
    return P


def degrees(units):
    d = {}
    for e in units:
        for v in e:
            d[v] = d.get(v, 0) + 1
    return d


def worst_gap(units, labels):
    best = Fr(0)
    for e in units:
        lab = [labels[v] for v in e]
        clusters = sorted({c for _, c in lab})
        for g in {g for g, _ in lab}:
            mem = [c for gg, c in lab if gg == g]
            tv = sum(abs(Fr(mem.count(c), len(mem)) - Fr(sum(1 for _, cc in lab if cc == c), len(lab)))
                     for c in clusters) / 2
            best = max(best, tv)
    return best


def u_p(units, labels):
    P = P_release(units)
    return sum(w for (u, v), w in P.items() if chi(labels[u], labels[v])) / sum(P.values())


def main():
    a = base_parser(__doc__.splitlines()[0]).parse_args()
    # ---------------- running example (Figure 1) and the dyadic surrogate
    lab = {0: (0, 0), 1: (0, 0), 2: (1, 1)}   # u, v in group a and cluster C1; w in group b and cluster C2
    H1 = [(0, 1, 2)]; H2 = [(0, 1), (0, 2), (1, 2)]
    check("Figure 1: HGP(H1) = 4/9, HGP(H2) = 1/3, same unweighted clique",
          hgp(H1, lab) == Fr(4, 9) and hgp(H2, lab) == Fr(1, 3) and set(pair_counts(H1)) == set(pair_counts(H2)),
          f"{hgp(H1, lab)} and {hgp(H2, lab)}")
    check("Figure 1: U_P equals HGP on H1 and on H2 (Prop. 2(a), units of at most three members)",
          u_p(H1, lab) == hgp(H1, lab) and u_p(H2, lab) == hgp(H2, lab))

    # ---------------- CGP example and Remark 1
    # a1=0, a2=1 in group a; b1=2, b2=3 in group b; units {a1,b1}, {a2,b2}; clusters {a1,b2} and {a2,b1}
    labx = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}
    U = [(0, 2), (1, 3)]
    cgp = lambda L: sum(Fr(sum(1 for v in L if L[v][1] == c), 4) * sum(abs(Fr(sum(1 for v in L if L[v] == (g, c)), sum(1 for v in L if L[v][1] == c)) - Fr(1, 2)) for g in (0, 1)) / 2 for c in (0, 1))
    swapped = dict(labx); swapped[0], swapped[1] = (0, 1), (0, 0)
    check("Section 2.2: CGP 0 and HGP 1/2; exchanging a1 and a2 keeps CGP 0 and gives HGP 0 (Remark 1)",
          cgp(labx) == 0 and hgp(U, labx) == Fr(1, 2) and cgp(swapped) == 0 and hgp(U, swapped) == 0)

    # ---------------- Lemma 1
    rng = random.Random(1)
    ok_le, ok_eq = True, True
    for _ in range(4000):
        k = rng.randint(2, 7)
        t = tuple(sorted((rng.randint(0, 2), rng.randint(0, 2)) for _ in range(k)))
        n = sum(chi(a, b) for a, b in combinations(t, 2))
        ok_le &= psi_type(t) <= Fr(2 * n, k * k)
        if k <= 3:
            ok_eq &= psi_type(t) == Fr(2 * n, k * k)
    check("Lemma 1: psi_e <= 2 N_e / |e|^2 on 4000 random units, with equality for |e| <= 3", ok_le and ok_eq)

    # ---------------- Corollary 1
    labs9 = [(g, c) for g in range(3) for c in range(3)]
    ok2 = all(Fr(len(t)) * psi_type(t) == chi(*t) for t in combinations_with_replacement(labs9, 2))
    ok3 = all(Fr(len(t)) * psi_type(t) == Fr(2, 3) * sum(chi(a, b) for a, b in combinations(t, 2))
              for t in combinations_with_replacement(labs9, 3))
    check("Corollary 1(i): eta = 2 chi / r solves Theorem 1(a) for r = 2, 3 (all label types, 3 groups x 3 clusters)", ok2 and ok3)
    lab7 = [(0, 0), (1, 1), (0, 1), (0, 1), (1, 0), (1, 0), (1, 1)]   # group a = {0, 2, 3}, cluster C1 = {0, 4, 5}
    pos = [(0, 2, 4), (0, 3, 5), (1, 2, 3), (1, 4, 5)]
    neg = [(0, 2, 3), (0, 4, 5), (1, 2, 4), (1, 3, 5)]
    for r in range(4, 9):
        Z = tuple(range(7, 7 + r - 4))
        labels = lab7 + [(1, 1)] * (r - 4)
        U1 = [tuple(sorted(e + (6,) + Z)) for e in pos]
        U2 = [tuple(sorted(e + (6,) + Z)) for e in neg]
        same = (pair_counts(U1) == pair_counts(U2) and P_release(U1) == P_release(U2) and degrees(U1) == degrees(U2)
                and weighted(U1, lambda k: Fr(1, k - 1)) == weighted(U2, lambda k: Fr(1, k - 1)))
        h1, h2 = hgp(U1, labels), hgp(U2, labels)
        ok = same and {h1, h2} == {Fr(r - 4, r * r), Fr(r - 2, r * r)} and abs(h2 - h1) == Fr(2, r * r)
        extra = ""
        if r == 4:
            wg = sorted([worst_gap(U1, labels), worst_gap(U2, labels)])
            ok &= wg == [0, Fr(1, 4)]
            extra = f", worst gap {wg[0]} vs {wg[1]}"
        check(f"Corollary 1(ii) r = {r}: same weighted cliques, P and degrees; HGP {min(h1, h2)} vs {max(h1, h2)}{extra}", ok)

    # ---------------- Corollary 2
    ok = True
    for _ in range(300):
        n = rng.randint(3, 7)
        labels = [rng.choice(LAB4) for _ in range(n)]
        units = [tuple(sorted(rng.sample(range(n), rng.randint(2, 3)))) for _ in range(rng.randint(1, 6))]
        ok &= u_p(units, labels) == hgp(units, labels)
    check("Corollary 2(i): HGP = U_P on 300 random multihypergraphs with units of two and three members", ok)
    labt = {0: (0, 0), 1: (0, 0), 2: (1, 1)}
    T3 = [(0, 1, 2)] * 3
    P2 = [(0, 1), (0, 2), (1, 2)] * 2
    PT, PP = P_release(T3), P_release(P2)
    off = all(PT.get((u, v), 0) == PP.get((u, v), 0) for u in range(3) for v in range(3) if u != v)
    volF = lambda U: sum(len(e) * psi_type(tuple(labt[v] for v in e)) for e in U)
    check("Corollary 2(i): three copies of a triple and two copies of each pair share the off-diagonal of P and vol*HGP; "
          "diagonal 1 vs 2, volumes 9 vs 12",
          off and volF(T3) == volF(P2) and PT[(0, 0)] == 1 and PP[(0, 0)] == 2 and sum(map(len, T3)) == 9 and sum(map(len, P2)) == 12)

    # ---------------- Corollary 3
    lab12 = [l for l in LAB4 for _ in range(3)]
    res = {}
    for rat in [Fr(1, 3), Fr(1, 2), Fr(2, 3), Fr(1), Fr(3, 2), Fr(2)]:
        rel = rel_wclique({2: 1, 3: rat})
        res[rat] = (criterion_type(lab12, (2, 3), rel, 'F')[0], criterion_type(lab12, (2, 3), rel, 'vol')[0])
    ok = all(res[r][0] == (r == Fr(2, 3)) and res[r][1] == (r == Fr(1, 2)) for r in res)
    check("Corollary 3(i): omega(3)/omega(2) determines vol*HGP exactly at 2/3 and vol exactly at 1/2",
          ok, str({str(k): v for k, v in res.items()}))
    lab3 = {0: (0, 0), 1: (0, 0), 2: (1, 1)}
    hT = hgp([(0, 1, 2)], lab3); hP = hgp([(0, 1), (0, 2), (1, 2)], lab3)
    check("Corollary 3(i): q copies of f vs p copies of its pairs give HGP 2N_f/9 and N_f/6 (4/9 and 1/3)",
          hT == Fr(4, 9) and hP == Fr(1, 3))
    l4 = {0: (0, 0), 1: (0, 0), 2: (0, 0), 3: (1, 1)}
    W1 = [(0, 3), (1, 3), (2, 3), (0, 1, 2)]; W2 = [(0, 1, 3), (0, 2, 3), (1, 2, 3)]
    nc = lambda U: weighted(U, lambda k: Fr(1, k - 1))
    check("Corollary 3(ii): {03,13,23,012} and {013,023,123} share the normalized clique; HGP 1/3 vs 4/9",
          nc(W1) == nc(W2) and hgp(W1, l4) == Fr(1, 3) and hgp(W2, l4) == Fr(4, 9))

    # ---------------- Corollary 4: decoding simple {2,3} releases with p >= 2
    bad = 0; trials = 0
    for (p, q) in [(2, 1), (2, 3), (3, 2), (5, 3)]:
        for _ in range(300):
            n = rng.randint(5, 9)
            labels = [rng.choice(LAB4) for _ in range(n)]
            pairs = [e for e in combinations(range(n), 2) if rng.random() < 0.3]
            tris = [e for e in combinations(range(n), 3) if rng.random() < 0.2]
            if not pairs and not tris:
                continue
            trials += 1
            Nv = {}
            for e in pairs:
                Nv[e] = Nv.get(e, 0) + q
            for e in tris:
                for pr in combinations(e, 2):
                    Nv[pr] = Nv.get(pr, 0) + p
            F = Fr(0); vol = 0
            for uv, N in Nv.items():
                xs = [x for x in (0, 1) if (N - q * x) >= 0 and (N - q * x) % p == 0]
                if len(xs) != 1:
                    bad += 1; break
                x = xs[0]; t = (N - q * x) // p
                F += x * chi(labels[uv[0]], labels[uv[1]]) + Fr(2, 3) * t * chi(labels[uv[0]], labels[uv[1]])
                vol += 2 * x + t
            trueF = sum(len(e) * psi_type(tuple(labels[v] for v in e)) for e in pairs + tris)
            bad += (F != trueF or vol != sum(len(e) for e in pairs + tris))
    check("Corollary 4: release with p >= 2 decodes units and vol*HGP on random simple {2,3} hypergraphs",
          bad == 0, f"{trials} inputs, {bad} failures")

    # ---------------- Theorem 3
    ok = True
    for r in range(3, 33):
        s, t = ceil(r / 2), floor(r / 2)
        labels = {v: ((0, 0) if v < s else (1, 1)) for v in range(r)}
        gap = hgp([tuple(range(r))], labels) - hgp(list(combinations(range(r), 2)), labels)
        ok &= gap == Fr(s * t * (r - 2), r * r * (r - 1))
    g = lambda r: Fr(ceil(r / 2) * floor(r / 2) * (r - 2), r * r * (r - 1))
    check("Theorem 3: gap gamma(r) for r = 3..32, gamma(3) = 1/9, gamma(4) = 1/6, gamma(32) ~ 0.242",
          ok and g(3) == Fr(1, 9) and g(4) == Fr(1, 6) and abs(float(g(32)) - 0.242) < 5e-4, f"gamma(32) = {float(g(32)):.4f}")

    # ---------------- other additive audits at uniform size 3
    def mi_parts(t):
        cnt = {}
        for l in t:
            cnt[l] = cnt.get(l, 0) + 1
        k = len(t); ng, ncl = {}, {}
        for (gg, c), v in cnt.items():
            ng[gg] = ng.get(gg, 0) + v; ncl[c] = ncl.get(c, 0) + v
        out = {}

        def fac(m, sgn, w):
            d = 2
            while m > 1:
                while m % d == 0:
                    out[d] = out.get(d, 0) + sgn * w; m //= d
                d += 1
        for (gg, c), v in cnt.items():
            fac(k * v, 1, v); fac(ng[gg] * ncl[c], -1, v)
        return {pp: x for pp, x in out.items() if x}

    fail = {"F": 0, "L2": 0, "MI": 0}
    for cnt in itertools.product(range(4), repeat=4):
        labels = [l for l, c in zip(LAB4, cnt) for _ in range(c)]
        if len(labels) < 3:
            continue
        fail["F"] += not criterion_type(labels, (3,), REL['w1'], 'F')[0]
        fail["L2"] += not criterion_type(labels, (3,), REL['w1'], 'L2')[0]
        fail["MI"] += not all(criterion_type(labels, (3,), REL['w1'], (lambda t, pp=pp: Fr(mi_parts(t).get(pp, 0))))[0]
                              for pp in (2, 3, 5, 7))
    check("Section 3.2: of the 256 choices, the 3-uniform weighted clique leaves the squared dependence and the "
          "mutual information undetermined in 129 and determines HGP in all",
          fail["F"] == 0 and fail["L2"] == 129 and fail["MI"] == 129, str(fail))

    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "constructions.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print("ALL CONSTRUCTIONS OK" if ALL_OK[0] else "SOME CONSTRUCTION FAILED")
    raise SystemExit(0 if ALL_OK[0] else 1)


if __name__ == "__main__":
    main()
