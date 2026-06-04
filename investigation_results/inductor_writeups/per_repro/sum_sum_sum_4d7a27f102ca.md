# sum_sum_sum_4d7a27f102ca

## Current Result

- Family: `structured_pool_upsample_backward_reduce`
- Classification: `SCATTER_REDUCE`
- Oracle path: `repros/canonical/sum_sum_sum_4d7a27f102ca/oracle_structured_scatter_reduce.py`
- Correctness: PASS
- Oracle, `coordinate_descent_tuning=True`: `430.11 us`
- `torch.compile coordinate_descent_tuning=True`: `999.52 us`
- Oracle, combo looped config: `419.10 us`
- `torch.compile combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `685.28 us`
- Best compile/oracle gap: `1.64x`
- Historical best compile: `757.824 us`; local combo-looped compile improved this to `685.28 us`.

## Diagnosis

The oracle covers the full Swin repro scope. It reduces `fma_22` over the
`8192` batch dimension and scatter-adds the `[49, 49, 4]` relative-position
values into duplicate `[169, 4]` buckets by `primals_26`. It also computes the
complete softmax-backward producer from `bmm_141.view(8192, 4, 49, 49)` and the
non-contiguous `div` input, writes the required contiguous `[32768, 49, 49]`
side output, reduces the same producer over batch, and scatter-adds into the
second `[169, 4]` bucket table by `primals_11`.

Inductor currently lowers the two `sum(dim=0) -> permute/reshape ->
index_put(accumulate=True)` branches and the softmax-backward side-output
branch as separate generic reductions, layout operations, scatter kernels, and
pointwise work over materialized intermediates. The oracle instead uses one
source-space Triton producer that keeps each row tile live, emits the full side
output, and accumulates duplicate relative-position buckets from the same tile.

The actionable fix is `SCATTER_REDUCE`: add a structured relative-position
scatter-reduce lowering that recognizes duplicate-index
`index_put(accumulate=True)` consumers and can fuse them with compatible
batch-reduction producers plus required materialized side-output stores.

## Commands

```bash
python -m py_compile repros/canonical/sum_sum_sum_4d7a27f102ca/oracle_structured_scatter_reduce.py
python repros/canonical/sum_sum_sum_4d7a27f102ca/oracle_structured_scatter_reduce.py --check
python repros/canonical/sum_sum_sum_4d7a27f102ca/oracle_structured_scatter_reduce.py --bench --warmup 10 --rep 50
TORCHINDUCTOR_CACHE_DIR=/tmp/bb_sum_sum_sum_4d7a27f102ca_combo_cache python - <<'PY'
import runpy
import sys
import torch._inductor.config as cfg

cfg.combo_kernels = True
cfg.combo_kernel_per_subkernel_blocks = True
cfg.coordinate_descent_tuning = True
cfg.benchmark_combo_kernel = True
cfg.triton.multi_kernel = 3

sys.argv = [
    'oracle_structured_scatter_reduce.py',
    '--bench',
    '--warmup', '10',
    '--rep', '50',
]
runpy.run_path(
    'repros/canonical/sum_sum_sum_4d7a27f102ca/oracle_structured_scatter_reduce.py',
    run_name='__main__',
)
PY
```
