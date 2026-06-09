# var_mean_c28c8bc99940

## Current Result

- Family: `var_mean`
- Classification: `SCHEDULER_FUSION`
- Oracle path: `repros/canonical/var_mean_c28c8bc99940/oracle_densenet_bn_train_cat.py`
- Correctness: PASS
- Oracle: `9.98 us`
- `torch.compile`: `14.05 us`
- Ratio: 1.407x
- `torch.compile triton.multi_kernel=3`: `11.10 us` (1.43x speedup)
- Parent status: `fixable_config`

## Diagnosis

The oracle computes the full DenseNet concat-BatchNorm-ReLU training scope directly
from three channel-concat source tensors (avg_pool2d [64,512,7,7] + two convolutions
[64,32,7,7]) including per-channel var_mean, invstd and mean side outputs, in-place
running mean/variance copy_ returns, and the affine ReLU activation, all without
materializing the logical cat, in a single-pass Triton kernel with BLOCK_K=4096
(>3136 elements per channel).

Inductor generates 1 fused kernel but uses a **two-pass** approach: the first loop
computes Welford reduction statistics over the 3136 elements per channel, then a
second loop re-reads all input data to compute the normalized/affine/ReLU output.
This doubles memory traffic for the large input tensor (~7MB).

### Root Cause

Inductor's reduction code generation splits the var_mean reduction from the
normalization epilogue when the reduction dimension is too large for a single-pass
persistent approach. The reduction loop accumulates statistics (Welford), then
a separate epilogue loop re-reads data to apply normalize+affine+ReLU. With 3136
elements per channel, the data can fit in a BLOCK_K=4096 register tile, enabling
a single-pass approach where values are loaded once, used for statistics, then
immediately reused for normalization. Inductor does not currently attempt this
"register reuse" optimization for reductions with fused epilogues.

### Inductor Kernel Count

1 kernel (fully fused but 2-pass)

### Configs Tried

- `persistent_reductions=True` (already default): no change, epilogue still re-reads.
  The persistent reduction threshold for INNER hints is 1024, but r0_numel=3136 exceeds it.
- `max_autotune=True`: no structural change to kernel strategy
- **`triton.multi_kernel=3`: 1.43x faster (11.10us vs 15.84us baseline)**. This config
  generates both a loop-based and a persistent reduction variant, benchmarks both at
  compile time, and selects the faster one. The persistent variant uses RBLOCK=4096
  (next power of 2 above 3136) and avoids the second data read entirely.

### Fix Assessment

FIXABLE via `triton.multi_kernel=3`. This config raises the persistent reduction
threshold by 16x (1024*16=16384 > 3136), causing Inductor to also generate a
persistent variant. The persistent kernel holds all 3136 elements in registers and
computes both the Welford reduction and the normalize+affine+ReLU epilogue in a
single pass, matching the oracle's strategy.

Alternatively, the persistent reduction threshold heuristic in
`torch/_inductor/choices.py:should_use_persistent_reduction` could be adjusted
to allow persistent mode for reductions up to 4096 elements with INNER hints when
the x dimension (576 channels here) provides sufficient parallelism.

## Commands

```bash
python repros/canonical/var_mean_c28c8bc99940/oracle_densenet_bn_train_cat.py --check
python repros/canonical/var_mean_c28c8bc99940/oracle_densenet_bn_train_cat.py --bench
python -m py_compile repros/canonical/var_mean_c28c8bc99940/oracle_densenet_bn_train_cat.py
python scripts/validate_corpus_invariants.py
```
