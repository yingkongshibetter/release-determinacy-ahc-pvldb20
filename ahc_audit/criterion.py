"""Exact rational tools for the determinacy criterion (Theorems 1 and 2) and the certificates as dual
solutions (Proposition 4).

Labels are (group, cluster) pairs and units are sorted tuples of node ids. A linear pairwise release is given
by pair coefficients c_j(|e|) and node coefficients d_j(|e|); an additive audit by phi(label type). The type
system of Theorem 1(c), exact row reduction, and an exact simplex method (lp_max) with dual checks are in
Fraction arithmetic throughout.
"""
from fractions import Fraction as Fr
from itertools import combinations, combinations_with_replacement
from math import comb, lcm
import random

# ---------------------------------------------------------------- audits
def counts_of(labs):
    cnt = {}
    for l in labs:
        cnt[l] = cnt.get(l, 0) + 1
    return cnt

def psi_type(labs):
    """psi_e of Eq. (hgp_dep) from the multiset of member labels."""
    cnt = counts_of(labs)
    k = len(labs)
    ng, nc = {}, {}
    for (g, c), v in cnt.items():
        ng[g] = ng.get(g, 0) + v
        nc[c] = nc.get(c, 0) + v
    tot = 0
    for g in ng:
        for c in nc:
            tot += abs(k * cnt.get((g, c), 0) - ng[g] * nc[c])
    return Fr(tot, 2 * k * k)

def ncross_type(labs):
    n = 0
    for (a, b) in combinations(labs, 2):
        if a[0] != b[0] and a[1] != b[1]:
            n += 1
    return n

def l2dep_type(labs):
    cnt = counts_of(labs)
    k = len(labs)
    ng, nc = {}, {}
    for (g, c), v in cnt.items():
        ng[g] = ng.get(g, 0) + v
        nc[c] = nc.get(c, 0) + v
    s = Fr(0)
    for g in ng:
        for c in nc:
            r = Fr(cnt.get((g, c), 0), k) - Fr(ng[g] * nc[c], k * k)
            s += r * r
    return k * s

AUDITS = {
    'F':     lambda L: len(L) * psi_type(L),          # vol * HGP numerator, phi(e) = |e| psi_e
    'vol':   lambda L: Fr(len(L)),
    'N':     lambda L: Fr(ncross_type(L)),             # crossing pairs
    'N/k':   lambda L: Fr(ncross_type(L), len(L)),
    'split': lambda L: Fr(1 if len({c for (_, c) in L}) >= 2 else 0),   # unit split by partition
    'mixsplit': lambda L: Fr(1 if (len({c for (_, c) in L}) >= 2 and len({g for (g, _) in L}) >= 2) else 0),
    'L2':    lambda L: l2dep_type(L),                  # |e| * sum of squared residuals
}

# ---------------------------------------------------------------- releases
# A release is given by pair-coefficient functions c_j(k) and node-coefficient functions d_j(k):
#   R_(uv,j) = sum_{e >= uv} c_j(|e|),   R_(v,j) = sum_{e ni v} d_j(|e|).
def rel_wclique(omega):
    return {'name': 'wclique' + str({k: str(v) for k, v in omega.items()}), 'c': [lambda k, o=omega: Fr(o[k])], 'd': []}

REL = {
    'norm':  {'name': 'normalized clique', 'c': [lambda k: Fr(1, k - 1)], 'd': []},
    'P':     {'name': 'random-walk P (pairs 1/k, diagonal 1/k)', 'c': [lambda k: Fr(1, k)], 'd': [lambda k: Fr(1, k)]},
    'Poff':  {'name': 'P without diagonal', 'c': [lambda k: Fr(1, k)], 'd': []},
    'w1':    {'name': 'weighted clique omega=1', 'c': [lambda k: Fr(1)], 'd': []},
}
def rel_sizesep(K):
    cs = [(lambda k, j=j: Fr(1 if k == j else 0)) for j in sorted(K)]
    return {'name': 'size-separated pair counts + degrees', 'c': cs, 'd': [lambda k: Fr(1)]}

class Instance:
    def __init__(self, labels, K, rel):
        self.labels = list(labels)
        self.n = len(labels)
        self.K = sorted(K)
        self.rel = rel
        self.units = [e for k in self.K for e in combinations(range(self.n), k)]
        self.pairs = list(combinations(range(self.n), 2))
        self.pidx = {p: i for i, p in enumerate(self.pairs)}
        nc, nd = len(rel['c']), len(rel['d'])
        self.ncoord = nc * len(self.pairs) + nd * self.n
        self.cols = []   # column of unit e: dict coord -> Fraction
        for e in self.units:
            k = len(e)
            col = {}
            for j, cf in enumerate(rel['c']):
                v = cf(k)
                if v != 0:
                    for p in combinations(e, 2):
                        col[j * len(self.pairs) + self.pidx[p]] = v
            for j, df in enumerate(rel['d']):
                v = df(k)
                if v != 0:
                    for x in e:
                        col[nc * len(self.pairs) + j * self.n + x] = v
            self.cols.append(col)

    def ltype(self, e):
        return tuple(sorted(self.labels[v] for v in e))

    def phi(self, audit):
        f = AUDITS[audit] if isinstance(audit, str) else audit
        cache = {}
        out = []
        for e in self.units:
            t = self.ltype(e)
            if t not in cache:
                cache[t] = Fr(f(t))
            out.append(cache[t])
        return out

    def release(self, x):
        """x: dict unit_index -> multiplicity. Returns dense list of Fractions."""
        r = [Fr(0)] * self.ncoord
        for i, m in x.items():
            for c, v in self.cols[i].items():
                r[c] += m * v
        return r

    def A_rows(self):
        rows = [[Fr(0)] * len(self.units) for _ in range(self.ncoord)]
        for j, col in enumerate(self.cols):
            for c, v in col.items():
                rows[c][j] = v
        return rows

# ---------------------------------------------------------------- exact linear algebra
def rref(M):
    """Row-reduce a copy of M (list of lists of Fractions). Returns (R, pivots)."""
    R = [row[:] for row in M]
    piv = []
    r = 0
    ncols = len(R[0]) if R else 0
    for c in range(ncols):
        p = None
        for i in range(r, len(R)):
            if R[i][c] != 0:
                p = i
                break
        if p is None:
            continue
        R[r], R[p] = R[p], R[r]
        inv = 1 / R[r][c]
        R[r] = [a * inv for a in R[r]]
        for i in range(len(R)):
            if i != r and R[i][c] != 0:
                f = R[i][c]
                Ri, Rr = R[i], R[r]
                R[i] = [a - f * b for a, b in zip(Ri, Rr)]
        piv.append(c)
        r += 1
        if r == len(R):
            break
    return R, piv

def rank(M):
    if not M:
        return 0
    return len(rref(M)[1])

def nullspace(M, ncols):
    """Basis of {z : M z = 0} (M rows of length ncols)."""
    if not M:
        return [[Fr(1) if i == j else Fr(0) for i in range(ncols)] for j in range(ncols)]
    R, piv = rref(M)
    free = [c for c in range(ncols) if c not in piv]
    basis = []
    for f in free:
        z = [Fr(0)] * ncols
        z[f] = Fr(1)
        for i, pc in enumerate(piv):
            z[pc] = -R[i][f]
        basis.append(z)
    return basis

def solve_consistent(M, rhs):
    """Is M y = rhs solvable? Returns (bool, one solution or None)."""
    ncols = len(M[0]) if M else 0
    aug = [row[:] + [b] for row, b in zip(M, rhs)]
    R, piv = rref(aug)
    if ncols in piv:
        return False, None
    y = [Fr(0)] * ncols
    for i, pc in enumerate(piv):
        y[pc] = R[i][ncols]
    return True, y

def criterion_full(inst, phi):
    """Exists g with phi(e) = <rho(e), g> for all candidate units? (phi in row space of A)."""
    AT = [[Fr(0)] * inst.ncoord for _ in inst.units]
    for j, col in enumerate(inst.cols):
        for c, v in col.items():
            AT[j][c] = v
    ok, g = solve_consistent(AT, phi)
    return ok, g

def to_int_vec(z):
    L = 1
    for a in z:
        L = lcm(L, a.denominator)
    return [int(a * L) for a in z]

# ---------------------------------------------------------------- label types
def label_counts(labels):
    return counts_of(labels)

def realizable_types(labels, K):
    cnt = counts_of(labels)
    labs = sorted(cnt)
    out = []
    for k in sorted(K):
        for t in combinations_with_replacement(labs, k):
            c = counts_of(t)
            if all(c[l] <= cnt[l] for l in c):
                out.append(t)
    return out

def pair_types(labels):
    cnt = counts_of(labels)
    labs = sorted(cnt)
    pts = []
    for i, a in enumerate(labs):
        for b in labs[i:]:
            if a == b and cnt[a] < 2:
                continue
            pts.append((a, b))
    return pts

def type_system(labels, K, rel, audit):
    """Rows M (one per realizable type), rhs f, column names."""
    pts = pair_types(labels)
    labs = sorted(counts_of(labels))
    colnames = [('g', j, pt) for j in range(len(rel['c'])) for pt in pts] + \
               [('d', j, l) for j in range(len(rel['d'])) for l in labs]
    cidx = {c: i for i, c in enumerate(colnames)}
    f = AUDITS[audit] if isinstance(audit, str) else audit
    M, rhs, types = [], [], []
    for t in realizable_types(labels, K):
        k = len(t)
        row = [Fr(0)] * len(colnames)
        c = counts_of(t)
        for j, cf in enumerate(rel['c']):
            ck = cf(k)
            if ck == 0:
                continue
            for (a, b) in combinations(t, 2):
                pt = (a, b) if a <= b else (b, a)
                row[cidx[('g', j, pt)]] += ck
        for j, df in enumerate(rel['d']):
            dk = df(k)
            if dk == 0:
                continue
            for l, v in c.items():
                row[cidx[('d', j, l)]] += dk * v
        M.append(row)
        rhs.append(Fr(f(t)))
        types.append(t)
    return M, rhs, colnames, types

def criterion_type(labels, K, rel, audit):
    M, rhs, cn, ty = type_system(labels, K, rel, audit)
    ok, y = solve_consistent(M, rhs)
    return ok, y, cn, ty

# ---------------------------------------------------------------- exact simplex (max c^T x, A x = b, x >= 0)
def lp_max(A, b, c):
    m, n = len(A), len(c)
    T, sign = [], []
    for i in range(m):
        row = list(A[i])
        bi = Fr(b[i])
        s = 1
        if bi < 0:
            row = [-a for a in row]
            bi = -bi
            s = -1
        sign.append(s)
        art = [Fr(0)] * m
        art[i] = Fr(1)
        T.append(row + art + [bi])
    N = n + m
    basis = [n + i for i in range(m)]
    alive = list(range(m))

    def pivot(r, col):
        inv = 1 / T[r][col]
        T[r] = [a * inv for a in T[r]]
        Tr = T[r]
        for i in range(len(T)):
            if i != r and T[i][col] != 0:
                f = T[i][col]
                T[i] = [a - f * bb for a, bb in zip(T[i], Tr)]
        basis[r] = col

    def run(cost, allowed):
        while True:
            cb = [cost[basis[i]] for i in range(len(T))]
            enter = None
            for j in allowed:
                if j in basis:
                    continue
                rc = cost[j] - sum(cb[i] * T[i][j] for i in range(len(T)) if T[i][j] != 0)
                if rc > 0:
                    enter = j
                    break
            if enter is None:
                return
            best, br = None, None
            for i in range(len(T)):
                if T[i][enter] > 0:
                    ratio = T[i][N] / T[i][enter]
                    if best is None or ratio < best or (ratio == best and basis[i] < basis[br]):
                        best, br = ratio, i
            if br is None:
                raise ValueError('unbounded')
            pivot(br, enter)

    cost1 = [Fr(0)] * n + [Fr(-1)] * m
    run(cost1, list(range(N)))
    if any(T[i][N] != 0 for i in range(len(T)) if basis[i] >= n):
        raise ValueError('infeasible')
    # drive artificials out of the basis
    keep = []
    for i in range(len(T)):
        if basis[i] >= n:
            jj = next((j for j in range(n) if T[i][j] != 0), None)
            if jj is not None:
                pivot(i, jj)
                keep.append(i)
            else:
                pass  # redundant row, artificial stays at zero and never re-enters
        else:
            keep.append(i)
    cost2 = list(map(Fr, c)) + [Fr(0)] * m
    run(cost2, list(range(n)))
    x = [Fr(0)] * n
    for i in range(len(T)):
        if basis[i] < n:
            x[basis[i]] = T[i][N]
    val = sum(ci * xi for ci, xi in zip(cost2[:n], x))
    # dual y = c_B B^{-1}: B^{-1} sits in the artificial columns
    cb = [cost2[basis[i]] if basis[i] < n else Fr(0) for i in range(len(T))]
    y = [Fr(0)] * m
    for k in range(m):
        y[k] = sum(cb[i] * T[i][n + k] for i in range(len(T)) if T[i][n + k] != 0) * sign[k]
    return val, x, y

def check_dual(A, b, c, y, val):
    """Exact check: A^T y >= c and b^T y == val."""
    m, n = len(A), len(c)
    for j in range(n):
        if sum(A[i][j] * y[i] for i in range(m) if A[i][j] != 0) < c[j]:
            return False
    return sum(bi * yi for bi, yi in zip(b, y)) == val
