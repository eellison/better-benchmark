# sum_7d7c71d09ef5

## Current Result

- Family: `structured_pool_upsample_backward_reduce`
- Classification: `SCATTER_REDUCE`
- Oracle path: `repros/canonical/sum_7d7c71d09ef5/oracle_structured_pool_upsample_reduce.py`
- Correctness: PASS
- Oracle: `308.324 us`
- `torch.compile coordinate_descent_tuning=True`: `429.606 us`
- `torch.compile combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `435.974 us`
- Best compile/oracle gap: `1.39x`
- Valid floor: yes

## Diagnosis

The oracle covers the full repro scope by materializing the required
`[64, 256, 5979]` zero-padded `slice_scatter` output and accumulating the
returned `[256]` masked channel sum from the same source-space pass over
`getitem_15`. It differs from Inductor by treating the central slice copy,
padding stores, and `where(arg32_1, full_1, getitem_15).sum([0, 2])` as one
structured scatter-reduce template with partial channel accumulators instead of
separate generic tensor work that rereads the source for the reduction after
building the padded side output. Inductor cannot do this today because the
scheduler/codegen does not represent a materialized zero-padded `slice_scatter`
producer plus an unpadded sibling reduction epilogue as a single structured
operation. The actionable fix is `SCATTER_REDUCE`: add a structured
`slice_scatter` lowering that emits required padded side-output stores and
accumulates sibling reductions directly from source tiles.

## Commands

```bash
python -m py_compile repros/canonical/sum_7d7c71d09ef5/oracle_structured_pool_upsample_reduce.py
python repros/canonical/sum_7d7c71d09ef5/oracle_structured_pool_upsample_reduce.py --check
python repros/canonical/sum_7d7c71d09ef5/oracle_structured_pool_upsample_reduce.py --bench --warmup 10 --rep 50
```
