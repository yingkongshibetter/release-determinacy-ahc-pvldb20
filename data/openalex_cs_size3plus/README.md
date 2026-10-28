# openalex_cs_size3plus

Derived from `openalex_cs` by keeping hyperedges of size >= 3.

- nodes: 24286
- hyperedges: 8219
- volume: 44525
- feature dim: 2225


<!-- diagnostics-summary -->
# openalex_cs_size3plus

- nodes: **24286**, hyperedges: **8219**, volume: **44525**, feature_dim: **2225**
- hyperedge size  min/median/p90/p99/max = 3 / 4 / 9 / 20 / 25
- node degree     min/median/p90/p99/max = 1 / 1 / 3 / 8 / 31

## Sensitive attributes
- **country** | weighted-TV 0.761 | NMI(label) 0.062 | missing 0.00%
    - feature predictability skipped (skipped: 72 classes > 32 (high cardinality))
- **institution_type** | weighted-TV 0.398 | NMI(label) 0.041 | missing 0.00%
    - feature → institution_type predictability acc=0.721 ± 0.004, macro-F1=0.385 (n=20000, K=9)
- **career_stage** | weighted-TV 0.382 | NMI(label) 0.012 | missing 0.00%
    - feature → career_stage predictability acc=0.779 ± 0.004, macro-F1=0.690 (n=20000, K=3)

## Warnings
- institution_type: feature predictability accuracy 0.72 > 0.7
- career_stage: feature predictability accuracy 0.78 > 0.7
