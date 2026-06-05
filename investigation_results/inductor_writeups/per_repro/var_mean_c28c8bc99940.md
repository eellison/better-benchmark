# var_mean_c28c8bc99940

## Current Result

- Family: `var_mean`
- Classification: `SCHEDULER_FUSION`
- Oracle path: `repros/canonical/var_mean_c28c8bc99940/oracle_densenet_bn_train_cat.py`
- Correctness: PASS
- Oracle: `9.98 us`
- `torch.compile`: `14.05 us`
- Ratio: 1.407x
- Parent status: `not_fixable_config`

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

- `persistent_reductions=True` (already default): no change, epilogue still re-reads
- `max_autotune=True`: no structural change to kernel strategy

### Fix Assessment

Not fixable via config. Requires Inductor scheduler/codegen changes to recognize
when a reduction epilogue can reuse values already loaded for the reduction (i.e.,
when reduction dimension fits in a single block). This is a known limitation of
Inductor's reduction template.

## Commands

```bash
python repros/canonical/var_mean_c28c8bc99940/oracle_densenet_bn_train_cat.py --check
python repros/canonical/var_mean_c28c8bc99940/oracle_densenet_bn_train_cat.py --bench
python -m py_compile repros/canonical/var_mean_c28c8bc99940/oracle_densenet_bn_train_cat.py
python scripts/validate_corpus_invariants.py
```
