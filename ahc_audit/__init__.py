"""Release determinacy for fairness certification in attributed hypergraph clustering.

Modules
    data          datasets, attributes, class labels and partition files
    hgp           exact HGP, CGP and the interaction index from incidence (exact rationals)
    sql           the exact HGP query on SQLite and DuckDB (per-size counts and sums)
    releases      unweighted clique, size-only weighted cliques, random-walk release P and T
    certificates  U_P, U_Q, U_Q,rbar and their dissimilarity-scale versions (Props. 2 and 3)
    inexact       certificates from rounded and pruned releases (Prop. 5)
    auditors      dyadic surrogate, normalized-clique surrogate, maximal-clique reconstruction
    budgets       shared, self-calibrated, partition-calibrated, scaled, tau and fixed budgets
    cells         all statistics of one cell (method, dataset, attribute)
    verdicts      misses, false alarms and coverage of every auditor under every budget
    declared      compatible inputs under declared unit sizes (exact extremes and witnesses)
    declared_search  local search for compatible inputs with declared number and sizes of units
    exchange      same-group exchanges that keep every cluster-level audit (Remark 1)
    exchange_kernels  numba kernels of the exchange search (optional)
    policy        cluster-level invitation policies and their decision gaps (Prop. 1)
    criterion     the determinacy criterion, exact rational linear algebra and linear programming
    small_instances  enumeration of compatible inputs on five to seven nodes (Prop. 4)
"""

__version__ = "1.0.0"
