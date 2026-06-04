# var_mean_var_mean_6d7a29cb97f1

- Family: `norm_template_canonicalization`
- Classification: `RECOMPUTE_FUSION`
- Status: full-scope oracle is a true floor on the required gates
- Oracle path: `repros/canonical/var_mean_var_mean_6d7a29cb97f1/oracle_norm_template.py`

## Queue State

- Work queue rank: `553`
- Current shared CSV status: `active_subagent`
- Current shared CSV canonical_oracle_path: empty
- Current shared CSV notes: `owner=Codex-template-norm-6d7a; claimed by Codex using template workflow; target_path=repros/canonical/var_mean_var_mean_6d7a29cb97f1/oracle_norm_template.py; no heuristic status filter`
- Parent integration recommendation: set status to `oracle_measured`, set canonical_oracle_path to the oracle path above, and record `measured_oracle_us=210.464`.

## Scope

The oracle covers the exact `Repro()(*make_inputs())` scope. It consumes the same
nine f32 CUDA inputs, mutates the four running-stat inputs in-place like the
`aten.copy_` nodes, and returns all eight outputs in repro order with matching
shapes, dtypes, and strides:

1. int8 low-memory max-pool offsets `[64, 64, 56, 56]`
2. second BN inverse std `[64]`
3. second BN affine/ReLU output `[64, 64, 56, 56]`
4. centered pooled tensor `[64, 64, 56, 56]`
5. first running mean update `[64]`
6. first running var update `[64]`
7. second running mean update `[64]`
8. second running var update `[64]`

## Gap Diagnosis

The oracle uses Triton for both channel `var_mean` reductions, the low-memory
max-pool offset contract, the second normalization epilogue, and the running-stat
updates. The key difference from Inductor is that the first BN affine/ReLU
producer is recomputed inside the 3x3 stride-2 max-pool stencil instead of being
materialized as a full `[64, 64, 112, 112]` tensor and reread by pooling. That
trades overlapping pointwise recomputation for less global-memory traffic and
establishes a lower full-scope floor.

The actionable compiler class is `RECOMPUTE_FUSION`: add a profitability-gated
path that sinks cheap pointwise producers, including BN affine/ReLU when their
per-channel stats are already materialized, into overlapping pooling stencils.

## Measurements

Correctness command:
`python repros/canonical/var_mean_var_mean_6d7a29cb97f1/oracle_norm_template.py --check`

Result: PASS for all eight outputs; output 0 matched exactly with zero int8
offset mismatches, and every output matched shape, dtype, and stride.

Benchmark command:
`python repros/canonical/var_mean_var_mean_6d7a29cb97f1/oracle_norm_template.py --bench --warmup 10 --rep 50`

- Oracle: `210.464 us`
- `coordinate_descent_tuning=True`: `364.672 us`
- `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `360.704 us`
- Historical best compile: `442.4000084400177 us`
- True floor: yes
