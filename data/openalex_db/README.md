# openalex_db

Hyperedges = one OpenAlex work; members = its authors.
Filter = primary_location.source.id ∈ {SIGMOD, VLDB, ICDE, PODS} (IDs supplied via --source-ids).
Features = per-author OpenAlex topic-id histogram, L1-normalised.
Sensitive = country, institution_type, career_stage. Label = primary topic field.


<!-- diagnostics-summary -->
# openalex_db

- nodes: **3027**, hyperedges: **2823**, volume: **11819**, feature_dim: **386**
- hyperedge size  min/median/p90/p99/max = 3 / 4 / 6 / 10 / 24
- node degree     min/median/p90/p99/max = 1 / 2 / 7 / 28 / 64

## Sensitive attributes
- **country** | weighted-TV 0.714 | NMI(label) 0.026 | missing 0.00%
    - feature → country predictability acc=0.453 ± 0.015, macro-F1=0.224 (n=2997, K=29)
- **institution_type** | weighted-TV 0.337 | NMI(label) 0.018 | missing 0.00%
    - feature → institution_type predictability acc=0.717 ± 0.016, macro-F1=0.282 (n=3026, K=8)
- **career_stage** | weighted-TV 0.398 | NMI(label) 0.011 | missing 0.00%
    - feature → career_stage predictability acc=0.522 ± 0.020, macro-F1=0.496 (n=3027, K=3)

## Warnings
- institution_type: feature predictability accuracy 0.72 > 0.7
