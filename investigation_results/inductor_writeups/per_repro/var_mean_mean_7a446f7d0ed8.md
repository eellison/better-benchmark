# var_mean_mean_7a446f7d0ed8

## Current Result

- Family: `var_mean_mean`
- Classification: `SCHEDULER_FUSION`
- Oracle path: `repros/canonical/var_mean_mean_7a446f7d0ed8/oracle_bn_relu_spatial_mean.py`
- Correctness: PASS
- Oracle: `52.10 us`
- `torch.compile`: `61.44 us`
- Ratio: 1.179x
- `torch.compile coordinate_descent_tuning=True`: `64.28 us` (measured with CUDAGraph)
- Parent status: `not_fixable_config`

## Diagnosis

The oracle computes the MobileNetV3 training-BatchNorm (var_mean over [0,2,3] dims
on a [256,120,28,28] tensor), running-stat copy_ side effects, affine ReLU, and
final [256,120,1,1] spatial mean (mean over [-1,-2]) using a split-statistics
approach: first kernel computes partial sums across batch blocks, second kernel
finalizes stats, updates running mean/var, then applies normalize/ReLU/spatial-pool
in a single fused pass per batch block.

Inductor generates 2 kernels:
1. Reduction kernel for var_mean (xnumel=120, r0_numel=200704=256*28*28): computes
   BN statistics + running stat updates
2. Per-element kernel (xnumel=30720=256*120, r0_numel=784=28*28): reads the full
   input again, applies normalize, ReLU, and spatial mean reduction

### Root Cause

Inductor separates the BN statistical reduction (over batch+spatial dims) from the
downstream normalize+ReLU+spatial_mean computation. The second kernel must re-read
all 256*120*28*28 = 24M elements. The oracle's split-statistics design avoids this
by computing partial statistics first (small intermediate), then doing
normalize+ReLU+pool in a single pass that only reads each element once for the
second phase.

The fundamental issue: Inductor's norm-template scheduler does not fuse a downstream
spatial reduction (mean over H,W) into the normalization epilogue. It materializes
the full normalized+ReLU tensor, then launches a separate reduction for the spatial
mean.

### Inductor Kernel Count

2 kernels

### Configs Tried

- Default config: 2 kernels, ~62us (with CUDAGraph)
- `coordinate_descent_tuning=True`: ~64us (marginal, no structural change)
- `combo_kernels=True`: no improvement over cd alone
- The issue is structural scheduler fusion, not autotuning

### Fix Assessment

Not fixable via config. `coordinate_descent_tuning` improves block sizes slightly
but cannot fuse the spatial mean into the normalization epilogue. Requires extending
Inductor's BN-training template to recognize when a downstream spatial reduction
consumer (mean.dim) can be fused into the normalization epilogue, avoiding
materialization of the full normalized tensor (256*120*28*28 = ~96MB fp32).

## Commands

```bash
python repros/canonical/var_mean_mean_7a446f7d0ed8/oracle_bn_relu_spatial_mean.py --check
python repros/canonical/var_mean_mean_7a446f7d0ed8/oracle_bn_relu_spatial_mean.py --bench
python -m py_compile repros/canonical/var_mean_mean_7a446f7d0ed8/oracle_bn_relu_spatial_mean.py
python scripts/validate_corpus_invariants.py
```
