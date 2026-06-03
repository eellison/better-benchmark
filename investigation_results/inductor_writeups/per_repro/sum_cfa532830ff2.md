# sum_cfa532830ff2

## Current Result

- Family: `structured_pool_upsample_backward_reduce`
- Classification: `SCATTER_REDUCE`
- Oracle path: `repros/canonical/sum_cfa532830ff2/oracle_structured_scatter_reduce.py`
- Correctness: PASS
- Oracle: `173.122 us`
- `torch.compile coordinate_descent_tuning`: `222.179 us`
- `torch.compile combo_looped_cd`: `234.627 us`
- Best compile/oracle gap: `1.28x`

## Diagnosis

The oracle writes the full `[64, 512, 1493]` zero-padded `slice_scatter`
side output and accumulates the `[512]` masked channel reduction from the same
source-space pass over `getitem_21`. It differs from Inductor by treating the
padding write and the unpadded `where(arg33_1, full_1, getitem_21).sum([0, 2])`
as one structured template with partial channel accumulators, instead of
materializing/scheduling the side output and reduction as separate generic
tensor work. The actionable fix is `SCATTER_REDUCE`: add a structured
`slice_scatter` lowering that can emit required padded side-output stores while
accumulating sibling reductions directly from the source tile.

## Commands

```bash
python -m py_compile repros/canonical/sum_cfa532830ff2/oracle_structured_scatter_reduce.py
python repros/canonical/sum_cfa532830ff2/oracle_structured_scatter_reduce.py --check
python repros/canonical/sum_cfa532830ff2/oracle_structured_scatter_reduce.py --bench --warmup 10 --rep 50
```
