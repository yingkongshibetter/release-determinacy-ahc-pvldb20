# openalex_db

Hyperedges = one OpenAlex work; members = its authors.
Filter = concepts.id:C95444343 (Biology) (IDs supplied via --source-ids).
Features = per-author OpenAlex topic-id histogram, L1-normalised.
Sensitive = country, institution_type, career_stage. Label = primary topic field.


<!-- diagnostics-summary -->
# openalex_bio

- nodes: **3764**, hyperedges: **1799**, volume: **9279**, feature_dim: **612**
- hyperedge size  min/median/p90/p99/max = 3 / 4 / 9 / 16 / 25
- node degree     min/median/p90/p99/max = 1 / 2 / 4 / 9 / 26

## Sensitive attributes
- **country** | weighted-TV 0.538 | NMI(label) 0.034 | missing 0.00%
    - feature → country predictability acc=0.726 ± 0.021, macro-F1=0.432 (n=3738, K=25)
- **institution_type** | weighted-TV 0.525 | NMI(label) 0.015 | missing 0.00%
    - feature → institution_type predictability acc=0.531 ± 0.011, macro-F1=0.373 (n=3764, K=8)
- **career_stage** | weighted-TV 0.271 | NMI(label) 0.010 | missing 0.00%
    - feature → career_stage predictability acc=0.912 ± 0.013, macro-F1=0.838 (n=3764, K=3)

## Warnings
- country: feature predictability accuracy 0.73 > 0.7
- career_stage: feature predictability accuracy 0.91 > 0.7
