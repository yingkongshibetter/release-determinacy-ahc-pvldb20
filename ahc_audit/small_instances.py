"""Exhaustive enumeration of compatible inputs on small node sets (RQ2: the linear relaxation of Prop. 4).

For a labeled node set and admissible unit sizes, every multihypergraph with multiplicities bounded by b and
release mass bounded by mass_bound is enumerated; inputs with the same release form one complete set of
compatible inputs (a fiber), whose extreme audit values h_min and h_max are exact.
"""
import sys

from .criterion import *  # noqa: F401,F403
from .criterion import Fr, lcm

def scale_all(vals):
    L = 1
    for v in vals:
        L = lcm(L, Fr(v).denominator)
    return L, [int(Fr(v) * L) for v in vals]

def random_type_audit(types, rng, lo=-6, hi=6, den=5):
    d = {}
    for t in types:
        d[t] = Fr(rng.randint(lo, hi), rng.randint(1, den))
    return d

def rowspace_type_audit(labels, K, rel, rng):
    """f(tau) = row_tau . y for a random type-level y: criterion holds by construction."""
    M, _, cn, ty = type_system(labels, K, rel, 'vol')
    y = [Fr(rng.randint(-5, 5), rng.randint(1, 4)) for _ in cn]
    return {t: sum(a * b for a, b in zip(row, y)) for t, row in zip(ty, M)}

def add_to_span(span, d):
    # span: dict pivot -> sparse vector (dict index -> Fraction), pivot = smallest index of the vector
    v = {i: Fr(a) for i, a in d.items()}
    while v:
        p = min(v)
        if p in span:
            w = span[p]
            f = v[p] / w[p]
            for i, a in w.items():
                nv = v.get(i, 0) - f * a
                if nv == 0:
                    v.pop(i, None)
                else:
                    v[i] = nv
        else:
            span[p] = v
            return True
    return False

def orth_to_span(span, phi):
    return all(sum(phi[i] * a for i, a in w.items()) == 0 for w in span.values())

def enum_fibers(inst, audits, b=1, mass_bound=None, max_configs=6_000_000, span=None, span_cap=None):
    """Enumerate all multisets x of candidate units with multiplicity <= b and release mass <= mass_bound
    (mass = sum of release entries, which the release determines, so every fiber found is complete).
    audits: dict name -> list of Fractions (phi per unit). Returns dict key -> fiber record."""
    nE = len(inst.units)
    allv = [v for col in inst.cols for v in col.values()]
    Lr, _ = scale_all(allv)
    cols = [[(c, int(v * Lr)) for c, v in col.items()] for col in inst.cols]
    mass = [sum(v for _, v in col) for col in cols]
    order = sorted(range(nE), key=lambda i: mass[i])
    names = list(audits)
    scl = {}
    phiint = {}
    for nm in names:
        L, ints = scale_all(audits[nm])
        scl[nm] = L
        phiint[nm] = ints
    volint = [len(e) for e in inst.units]
    Mb = None if mass_bound is None else int(Fr(mass_bound) * Lr)
    state = [0] * inst.ncoord
    acc = [0] * len(names)
    vol = [0]
    mult = [0] * nE
    fib = {}
    count = [0]
    phis_by_unit = [[phiint[nm][i] for nm in names] for i in range(nE)]
    if span_cap is None:
        span_cap = [10 ** 9]

    def record():
        key = tuple(state)
        rec = fib.get(key)
        F = tuple(acc)
        if rec is None:
            fib[key] = {'n': 1, 'min': list(F), 'max': list(F), 'vol0': vol[0], 'F0': F,
                        'ratio_diff': [False] * len(names), 'x_min': [tuple(mult)] * len(names),
                        'x_max': [tuple(mult)] * len(names), 'rmax': [(F[i], vol[0]) for i in range(len(names))],
                        'x_rmax': [tuple(mult)] * len(names), 'x0': tuple(mult), 'allF': [F[0]]}
        else:
            rec['n'] += 1
            rec['allF'].append(F[0])  # audit value of every member (first audit)
            if span is not None and len(span) < span_cap[0]:
                x0 = rec['x0']
                d = {i: mult[i] - x0[i] for i in range(nE) if mult[i] != x0[i]}
                add_to_span(span, d)
            for i in range(len(names)):
                if F[i] < rec['min'][i]:
                    rec['min'][i] = F[i]; rec['x_min'][i] = tuple(mult)
                if F[i] > rec['max'][i]:
                    rec['max'][i] = F[i]; rec['x_max'][i] = tuple(mult)
                if F[i] * rec['vol0'] != rec['F0'][i] * vol[0]:
                    rec['ratio_diff'][i] = True
                fm, vm = rec['rmax'][i]
                if F[i] * vm > fm * vol[0]:
                    rec['rmax'][i] = (F[i], vol[0]); rec['x_rmax'][i] = tuple(mult)
        count[0] += 1
        if count[0] > max_configs:
            raise RuntimeError('too many configurations')

    sys.setrecursionlimit(10000)

    def dfs(start, rem):
        for pos in range(start, nE):
            i = order[pos]
            if Mb is not None and mass[i] > rem:
                break
            if mult[i] >= b:
                continue
            mult[i] += 1
            for c, v in cols[i]:
                state[c] += v
            pu = phis_by_unit[i]
            for j in range(len(names)):
                acc[j] += pu[j]
            vol[0] += volint[i]
            record()
            dfs(pos if mult[i] < b else pos + 1, None if rem is None else rem - mass[i])
            mult[i] -= 1
            for c, v in cols[i]:
                state[c] -= v
            for j in range(len(names)):
                acc[j] -= pu[j]
            vol[0] -= volint[i]

    dfs(0, Mb)
    return fib, names, count[0], scl

def summarize(fib, names):
    """Per audit: is F constant on every fiber? is F/vol constant on every fiber? one witness each."""
    res = {}
    for j, nm in enumerate(names):
        nonc = [k for k, r in fib.items() if r['min'][j] != r['max'][j]]
        rnonc = [k for k, r in fib.items() if r['ratio_diff'][j]]
        res[nm] = {'F_det': len(nonc) == 0, 'ratio_det': len(rnonc) == 0,
                   'F_wit': (fib[nonc[0]]['x_min'][j], fib[nonc[0]]['x_max'][j]) if nonc else None,
                   'n_nonconst': len(nonc), 'n_ratio_nonconst': len(rnonc)}
    return res


# ---------------------------------------------------------------- the linear relaxation on complete sets
import random as _random
import time as _time

A_, B_, C_, D_ = (0, 0), (0, 1), (1, 0), (1, 1)
CONFIGS = {
    # name: node labels (group, cluster), admissible sizes K, release, multiplicity bound, mass bound
    # (sum of release entries over unordered pairs), number of sets with a nonzero compatible range to evaluate
    'L1': dict(labels=[A_, D_, B_, B_, C_, C_, D_], K=(4,), rel='P', b=1, mb=Fr(25, 2), sample=120),
    'L2': dict(labels=[A_, A_, B_, C_, D_], K=(2, 3, 4), rel='P', b=1, mb=Fr(19, 2), sample=120),
    'L3': dict(labels=[A_, B_, C_, D_, A_, D_], K=(3, 4), rel='P', b=1, mb=Fr(10), sample=120),
    'L4': dict(labels=[A_, A_, B_, C_, D_], K=(2, 3, 4), rel='norm', b=1, mb=Fr(6), sample=80),
    'L5': dict(labels=[A_, B_, C_, D_, (2, 0), (2, 1), (0, 2)], K=(4,), rel='P', b=1, mb=Fr(25, 2), sample=80),
}


def _type_lp_data(inst, R, audit_f):
    labels, K, rel = inst.labels, inst.K, inst.rel
    M, f, cn, types = type_system(labels, K, rel, audit_f)
    rhs = []
    npairs = len(inst.pairs)
    for (kind, j, key) in cn:
        s = Fr(0)
        if kind == 'g':
            for i, (u, v) in enumerate(inst.pairs):
                a, b = labels[u], labels[v]
                if ((a, b) if a <= b else (b, a)) == key:
                    s += R[j * npairs + i]
        else:
            for v in range(inst.n):
                if labels[v] == key:
                    s += R[len(rel['c']) * npairs + j * inst.n + v]
        rhs.append(s)
    A = [[M[t][c] for t in range(len(types))] for c in range(len(cn))]
    return A, rhs, f


def _u_p(inst, R, vol):
    s = Fr(0)
    for i, (u, v) in enumerate(inst.pairs):
        a, b = inst.labels[u], inst.labels[v]
        if a[0] != b[0] and a[1] != b[1]:
            s += 2 * R[i]
    return s / vol


def _u_q_cap(inst, R, rbar):
    tot = sum(R[:len(inst.pairs)])
    s = Fr(0)
    for i, (u, v) in enumerate(inst.pairs):
        a, b = inst.labels[u], inst.labels[v]
        if a[0] != b[0] and a[1] != b[1]:
            s += R[i]
    return Fr(rbar - 1, rbar) * s / tot


def lp_check(name, log=print):
    """Evaluate the linear relaxation (U_LP) against h_max and the certificate (U_P, or U_Q,4 for the
    normalized clique) on complete sets of compatible inputs of one configuration. Returns (stats, rows)."""
    cfg = CONFIGS[name]
    rel = REL[cfg['rel']]
    inst = Instance(cfg['labels'], cfg['K'], rel)
    phiF = inst.phi('F')
    t_en = _time.time()
    fib, names, ncfg, scl = enum_fibers(inst, {'F': phiF}, b=cfg['b'], mass_bound=cfg['mb'])
    t_en = _time.time() - t_en
    rng = _random.Random(11)
    keys = list(fib)
    wide = [k for k in keys if fib[k]['min'][0] != fib[k]['max'][0]]
    rest = [k for k in keys if fib[k]['min'][0] == fib[k]['max'][0]]
    pick = wide[:cfg['sample']] + rng.sample(rest, min(len(rest), cfg['sample'] // 2))
    Arows = inst.A_rows()
    stats = dict(config=name, release=cfg['rel'], sets=0, chain_ok=0, dual_ok=0, lp_eq_hmax=0, lp_lt_cert=0,
                 cert_eq_hmax=0, wide=0, t_enum_s=t_en)
    rows = []
    for key in pick:
        rec = fib[key]
        x = {i: m for i, m in enumerate(rec['x_max'][0]) if m}
        R = inst.release(x)
        vol = sum(len(inst.units[i]) * m for i, m in x.items())
        hmax = Fr(rec['max'][0], scl['F']) / vol
        hmin = Fr(rec['min'][0], scl['F']) / vol
        t_lp = _time.perf_counter()
        val, xopt, y = lp_max(Arows, R, phiF)
        t_lp = _time.perf_counter() - t_lp
        dual_ok = check_dual(Arows, R, phiF, y, val)
        ULP = val / vol
        At, bt, ft = _type_lp_data(inst, R, 'F')
        valt, yt, dt_ = lp_max(At, bt, ft)
        dual_ok_t = check_dual(At, bt, ft, dt_, valt)
        Utype = valt / vol
        t_u = _time.perf_counter()
        U = _u_p(inst, R, vol) if cfg['rel'] == 'P' else _u_q_cap(inst, R, max(cfg['K']))
        t_u = _time.perf_counter() - t_u
        stats['sets'] += 1
        stats['chain_ok'] += (hmin <= hmax <= ULP <= Utype <= U)
        stats['dual_ok'] += (dual_ok and dual_ok_t)
        stats['lp_eq_hmax'] += (ULP == hmax)
        stats['lp_lt_cert'] += (ULP < U)
        stats['cert_eq_hmax'] += (U == hmax)
        stats['wide'] += (hmin < hmax)
        rows.append(dict(config=name, release=cfg['rel'], members=rec['n'], h_min=str(hmin), h_max=str(hmax),
                         U_LP=str(ULP), U_type=str(Utype), U_cert=str(U), t_lp_s=t_lp, t_cert_s=t_u))
    log(f"  {name}: {stats}")
    return stats, rows
