# sum_sum_sum_4bb798eb542e

## Current Result

- Family: `structured_pool_upsample_backward_reduce`
- Classification: `SCATTER_REDUCE`
- Oracle path: `repros/canonical/sum_sum_sum_4bb798eb542e/oracle_structured_scatter_reduce.py`
- Correctness: PASS
- Oracle: `124.194 us`
- `torch.compile coordinate_descent_tuning`: `141.506 us`
- `torch.compile combo_looped_cd`: `139.042 us`
- Best compile/oracle gap: `1.12x`

## Diagnosis

The oracle covers the full Swin repro scope: it reduces `fma` over batch and
scatter-adds by `arg173_1`, computes the full `fma_default` tensor from
`bmm_5.view(128, 32, 49, 49)` and `arg419_1`, reduces that tensor over batch
and scatter-adds by `arg166_1`, and returns the required materialized
`[4096, 49, 49]` view. It differs from Inductor by using source-space Triton
tiles that partially reduce over batch before issuing duplicate-index scatter
adds, while the `fma_default` tile also writes the side output. Inductor cannot
do this today because duplicate-index `index_put(accumulate=True)` and the
materialized side-output view are scheduled as generic tensor operations rather
than one structured producer with reduction epilogues. The actionable fix is
`SCATTER_REDUCE`: add an indexed scatter-reduce lowering that can accumulate
duplicate spatial indices while preserving required side-output stores.

## Commands

```bash
python -m py_compile repros/canonical/sum_sum_sum_4bb798eb542e/oracle_structured_scatter_reduce.py
python repros/canonical/sum_sum_sum_4bb798eb542e/oracle_structured_scatter_reduce.py --check
python repros/canonical/sum_sum_sum_4bb798eb542e/oracle_structured_scatter_reduce.py --bench --warmup 10 --rep 50
```
