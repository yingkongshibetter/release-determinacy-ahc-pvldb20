# ACS-PUMS Household hypergraph

States: ma (2023 1-Year ACS PUMS).
n nodes (persons) = 59452
m hyperedges (households) = 20539
feature dimension = 142
label field = `education_bucket`  (9 unique)

## Sensitive attributes (audit only)
- `sex`: 2 unique values
- `age_bucket`: 5 unique values
- `race`: 7 unique values
- `disability`: 2 unique values

## Feature columns used
`ESR`, `COW`, `OCCP_super`, `RELSHIPP`, `PINCP_decile`

## Filters
- household type: housing unit (TYPEHUGQ=1)
- household size: [2, 10]
- household-level sampling cap: max 80000 persons (seed=2026)
- features exclude sex/age/race/disability and the label field

## License
Public domain (U.S. federal government work).
