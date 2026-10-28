"""Budgets (Section 5.1, Table 4).

Shared B         95th percentile of HGP over 200 partitions that keep the cluster sizes of the AHRC partition
                 and reassign nodes at random within strata, which are the class labels (labels.csv).
Self-calibrated  95th percentile of the auditor's own statistic over the same 200 partitions.
Partition-cal.   as B, with the cluster sizes of the audited partition.
Scaled           shared or self-calibrated budget times 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0.
Dissimilarity    tau in 0.1..0.7 on Dbar (and on D_CGP), read at 0.3 and 0.6.
Fixed            0.1 on HGP and CGP.

The shared budgets come from ONE generator, numpy default_rng(2026), consumed over the datasets and
attributes in the order of data.DATASETS / data.ATTRIBUTES. For each attribute the stream holds
  200 within-label reassignments (the reference partitions),
  two bootstrap calls of 200 resamples each (one for a CGP budget that the paper does not use, then the
  95% interval of B), and
  60 further within-label reassignments followed by 120 permutations of all nodes (a null-model comparison
  that the paper does not use).
Replaying this order reproduces the budgets of the paper to about 1e-13 (floating-point summation order).
Within one reassignment the strata are visited in the lexicographic order of the string "<label code>::0",
and the nodes of a stratum are permuted with rng.permutation; strata of one node are skipped.
"""
from __future__ import annotations

import numpy as np

from .data import ATTRIBUTES, DATASETS, METHODS, SCREENED
from .hgp import hgp_float

R = 200
Q = 0.95
FACTORS = ["0.6", "0.7", "0.8", "0.9", "1.0", "1.1", "1.25", "1.5", "2.0"]
TAUS = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"]
FIXED = "0.1"
SEED_SHARED = 2026
SEED_BOOT = 2027
SEED_PARTITION = 2026


def strata(labels: np.ndarray) -> list[np.ndarray]:
    """Node index arrays of the class-label strata with at least two nodes, in stream order."""
    keys = sorted({f"{int(c)}::0" for c in np.unique(labels)})
    out = []
    for k in keys:
        idx = np.flatnonzero(labels == int(k.split("::")[0]))
        if idx.size > 1:
            out.append(idx)
    return out


def reassign(clusters: np.ndarray, groups: list[np.ndarray], rng) -> np.ndarray:
    out = clusters.copy()
    for idx in groups:
        out[idx] = rng.permutation(out[idx])
    return out


def quantile_with_ci(samples, q=Q, B=200, rng=None):
    arr = np.asarray(samples, dtype=np.float64)
    point = float(np.quantile(arr, q))
    if rng is None:
        rng = np.random.default_rng(0)
    boots = np.empty(B, dtype=np.float64)
    n = len(arr)
    for b in range(B):
        boots[b] = np.quantile(arr[rng.integers(0, n, size=n)], q)
    return point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def boot_ci(x, spawn_key):
    rng = np.random.default_rng(np.random.SeedSequence(SEED_BOOT, spawn_key=spawn_key))
    n = len(x)
    b = np.array([np.quantile(x[rng.integers(0, n, size=n)], Q) for _ in range(200)])
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def shared_budget_stream(load, stats_fn, datasets=None, include_screened=False, log=print):
    """Replay the shared-budget stream.

    load(ds) -> Dataset; stats_fn(ds_obj, attribute) -> function(partition) -> dict of float statistics
    (must contain 'HGP'). Yields one dict per (dataset, attribute) with B, its 95% bootstrap interval and
    the 95th percentile of every other statistic (self-calibrated budgets) with their intervals.
    `datasets` restricts the output, but every earlier dataset in the order is still replayed (cheaply,
    without statistics) so that the stream position is right."""
    want = set(datasets or DATASETS)
    last = max(DATASETS.index(d) for d in want)
    rng = np.random.default_rng(SEED_SHARED)
    for di, ds in enumerate(DATASETS[:last + 1]):
        D = load(ds)
        base = D.partition("AHRC")
        groups = strata(D.labels)
        attrs = ATTRIBUTES[ds]
        need = [a for a in attrs if ds in want and (include_screened or (ds, a) not in SCREENED)]
        last_attr = max((attrs.index(a) for a in need), default=-1) if ds in want else len(attrs) - 1
        for ai, attr in enumerate(attrs):
            if ai > last_attr:
                break
            fn = stats_fn(D, attr) if attr in need else None
            vals = {}
            for r in range(R):
                p = reassign(base, groups, rng)
                if fn is not None:
                    for k, v in fn(p).items():
                        vals.setdefault(k, np.empty(R))[r] = v
            quantile_with_ci(np.zeros(R), Q, rng=rng)                 # CGP budget of the original run (unused)
            h = vals.get("HGP", np.zeros(R))
            B, lo, hi = quantile_with_ci(h, Q, rng=rng)
            for _ in range(60):
                reassign(base, groups, rng)
            for _ in range(120):
                rng.permutation(base)
            if fn is None:
                continue
            row = dict(dataset=ds, attribute=attr, B=B, B_CI_lo=lo, B_CI_hi=hi)
            for j, k in enumerate([k for k in vals if k != "HGP"], start=1):
                x = vals[k]
                sl, sh = boot_ci(x, (di, ai, j))
                row[f"B_{k}"] = float(np.quantile(x, Q)); row[f"B_{k}_CI_lo"] = sl; row[f"B_{k}_CI_hi"] = sh
            log(f"  {D.display:12s} {attr:16s} B {B:.6f} [{lo:.6f}, {hi:.6f}]")
            yield row


def partition_calibrated(D, method, attributes, log=print):
    """B_pm for every attribute of one (dataset, method): 200 within-label reassignments that keep the cluster
    sizes of the audited partition, seeds SeedSequence(2026, (dataset index, method index))."""
    di, mi = DATASETS.index(D.name), METHODS.index(method)
    cl = D.partition(method)
    groups = strata(D.labels)
    codes = {a: D.group(a) for a in attributes}
    rng = np.random.default_rng(np.random.SeedSequence(SEED_PARTITION, spawn_key=(di, mi)))
    null = {a: np.empty(R) for a in attributes}
    for t in range(R):
        shuf = reassign(cl, groups, rng)
        for a in attributes:
            null[a][t] = hgp_float(D.H, shuf, codes[a])
    out = {}
    for ai, a in enumerate(attributes):
        b, lo, hi = quantile_with_ci(null[a], Q, B=200,
                                     rng=np.random.default_rng(np.random.SeedSequence(SEED_BOOT, spawn_key=(di, mi, ai))))
        out[a] = dict(B_pm=b, B_pm_CI_lo=lo, B_pm_CI_hi=hi)
    return out
