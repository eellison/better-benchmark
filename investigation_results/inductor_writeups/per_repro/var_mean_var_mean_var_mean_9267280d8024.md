# var_mean_var_mean_var_mean_9267280d8024

## Current Result

- Family: `var_mean_var_mean_var_mean`
- Classification: `SCHEDULER_FUSION`
- Oracle path: `repros/canonical/var_mean_var_mean_var_mean_9267280d8024/oracle_inception_bn_relu_pool.py`
- Correctness: PASS
- Oracle: `82.78 us`
- `torch.compile`: `93.95 us`
- Ratio: 1.135x
- Parent status: `not_fixable_config`

## Diagnosis

The oracle computes 6 Inception-v3 branches of training BatchNorm + affine ReLU +
spatial mean (global average pool over 8x8), with in-place running-stat updates,
outputting a concatenated [128,2048] result. It uses 3 phases:
1. Partial statistics: per-branch partial sum/sumsq computation (shared across branches)
2. Finalize statistics: aggregates partials, updates running mean/var
3. BN+ReLU+spatial mean: normalizes, applies ReLU, and computes spatial mean in one pass

Inductor generates **10 kernels**:
- 3 var_mean reduction kernels (for channel dims 320, 384, 192)
- 3 finalize kernels (running stat updates per branch group)
- 3 pointwise kernels (normalize + ReLU, one handles cat)
- 1 spatial mean (per) kernel at the end

### Root Cause

Two related scheduler issues:
1. **Spatial mean not fused into normalization**: Inductor materializes the full
   normalized [128, C, 8, 8] tensors then launches a separate mean kernel. The
   oracle fuses spatial mean directly into the normalize+ReLU pass, avoiding the
   full materialization.
2. **Excessive kernel launches**: 10 kernels for what is structurally 6 independent
   BN+ReLU+pool branches. The oracle uses 18 launches but each is smaller/faster.
   The key savings come from not materializing intermediate normalized tensors.

Total input data: 6 branches x 128 x C x 8 x 8 floats, with C in {320,384,192}.
The extra materialization and re-read for spatial mean adds ~50MB of unnecessary
memory traffic.

### Inductor Kernel Count

10 kernels

### Configs Tried

- Default: 10 kernels, 93.95us
- Issue is structural: norm-template does not fuse downstream spatial reduction

### Fix Assessment

Not fixable via config. Same fundamental issue as var_mean_mean_7a446f7d0ed8:
Inductor's scheduler does not fuse spatial mean reductions into the normalization
epilogue. Additionally, the multi-branch structure means many kernel launches for
what could be more parallel work within fewer kernels.

## Commands

```bash
python repros/canonical/var_mean_var_mean_var_mean_9267280d8024/oracle_inception_bn_relu_pool.py --check
python repros/canonical/var_mean_var_mean_var_mean_9267280d8024/oracle_inception_bn_relu_pool.py --bench
python -m py_compile repros/canonical/var_mean_var_mean_var_mean_9267280d8024/oracle_inception_bn_relu_pool.py
python scripts/validate_corpus_invariants.py
```
