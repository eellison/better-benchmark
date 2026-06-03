# Inductor Writeup: BN/LN/RMSNorm Templates

## Status

- Queue id: `norm_templates_bn_ln_rms`
- Priority: P1
- Implementation status: **bandwidth_bound** -- no config-level improvement found (Planck)
- Owner: Planck
- Oracle coverage: `repros/canonical/var_mean_765fb8f2c85e/oracle_bn_training_forward.py`
- Implementation investigation: `investigation_results/inductor_writeups/norm_templates_impl.md`

## Target Repros

- `var_mean_765fb8f2c85e`
- `var_mean_598830735cf6`
- `var_mean_var_mean_var_mean_0407b3e7c77f`
- `mean_mean_1b98d81214e6`

## Objective

Recover semantic norm templates after decomposition and tune under coordinate descent. Do not rely on default heuristic recognition only.

## Oracle Gap Rationale

Generic `var_mean`/`mean` lowering computes stats and epilogues through generic reductions/pointwise chains. The oracle scaffold represents a semantic BN training forward floor with stats, affine, ReLU, and running-stat outputs.

## Stash Relevance

- scalar accumulators
- tiling scores
- `num_warps=2`
- persistent/reduction threshold changes

## Likely Inductor Hooks

- post-grad graph patterns for decomposed BN/LN/RMSNorm motifs
- lowering around `aten.var_mean`, `aten.mean`, `rsqrt`, affine epilogues
- TritonTemplate choices for `_inductor_bn_training_forward`, `_inductor_rms_norm_forward`, `_inductor_layer_norm_forward`

## Implementation Plan

1. Start with BN training forward motif for `var_mean_765fb8f2c85e`.
2. Add a semantic composite pattern before generic `var_mean` lowering.
3. Add a tuned template candidate that includes affine+ReLU epilogue and stats outputs.
4. Extend to multi-branch BN and RMSNorm only after first BN target improves.
5. Validate under forced coordinate-descent configs.

## Validation Commands

```bash
python scripts/bench_compare.py repros/canonical/var_mean_765fb8f2c85e/repro.py \
  --config "coordinate_descent_tuning=True" --label baseline_cd \
  --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=2" --label combo_persistent_cd \
  --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3" --label combo_looped_cd \
  --output /tmp/norm_template_compare.json

python repros/canonical/var_mean_765fb8f2c85e/oracle_bn_training_forward.py --check --no-append
```

## Success Metric

Best tuned runtime moves toward oracle scaffold and improves over final 3-config best.
