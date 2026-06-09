# sum_sum_sum_3c3d7e533ce5

## Classification: BAD_ORACLE

## Current Result

- Family: `distilbert_embedding_scatter_reduce`
- Oracle path: `repros/canonical/sum_sum_sum_3c3d7e533ce5/oracle_distilbert_embedding_scatter_reduce.py`
- Correctness: PASS
- Oracle: `294.62 us`
- `torch.compile coordinate_descent_tuning=True`: `224.99 us`
- Ratio: 0.764 (oracle slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle's DistilBERT embedding scatter-reduce kernel is 31% slower than Inductor's generated code. Inductor already handles this pattern more efficiently.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
