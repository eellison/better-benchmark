# sum_sum_sum_cad02fe18a69

## Current Result

- Family: `structured_pool_upsample_backward_reduce`
- Classification: `SCATTER_REDUCE`
- Oracle path: `repros/canonical/sum_sum_sum_cad02fe18a69/oracle_bert_embedding_scatter_reduce.py`
- Correctness: PASS
- Oracle: `96.288 us`
- `torch.compile coordinate_descent_tuning`: `155.552 us`
- `torch.compile combo_looped_cd`: `144.352 us`
- Best guarded compile/oracle gap: `1.50x`
- Valid floor: yes

## Diagnosis

The oracle covers the full BERT embedding/layernorm-backward repro scope: it
forms the summed activation, computes the two per-token layernorm reductions,
accumulates the two per-hidden reductions, and returns the same `[512,768]`,
`[2,768]`, and `[30522,768]` indexed accumulation outputs as `Repro.forward`.
It differs from Inductor by sharing the rowwise layernorm producer across all
five returned outputs and issuing the indexed accumulation epilogues directly
from the source rows. Inductor cannot do this today because its scheduler lowers
the rowwise reductions, sibling reductions, and three `index_put(accumulate=True)`
consumers as generic tensor work rather than as one structured scatter-reduce
template with multiple accumulator destinations. The actionable fix is
`SCATTER_REDUCE`: add an embedding/backward scatter-reduce lowering that fuses
the rowwise layernorm-backward producer with per-hidden reductions and indexed
row accumulation epilogues.

## Commands

```bash
python -m py_compile repros/canonical/sum_sum_sum_cad02fe18a69/oracle_bert_embedding_scatter_reduce.py
python repros/canonical/sum_sum_sum_cad02fe18a69/oracle_bert_embedding_scatter_reduce.py --check
python repros/canonical/sum_sum_sum_cad02fe18a69/oracle_bert_embedding_scatter_reduce.py --bench --warmup 10 --rep 50
```
