# sum_sum_sum_ed6c74b5ef96

## Status

- Family: `multi_output_reduction_templates`
- Owner: `Codex-template-multi-ed6c`
- Oracle artifact: `repros/canonical/sum_sum_sum_ed6c74b5ef96/oracle_multi_output_reduction.py`
- Classification: `SCATTER_REDUCE`
- True floor: no, diagnosis-only. Parent should keep `canonical_oracle_path` blank.

## Scope

The oracle covers the full `Repro()(*make_inputs())` scope and returns the same
eight outputs with matching shapes, dtypes, and strides. It inverts the
max-pool-backward scatter-add in Triton for each `[N, C, H, W]` element, adds the
two dense residual inputs, then runs one all-channel two-accumulator reduction
for the four BN/ReLU-backward branches. A finalizer writes the four vector
outputs, and a Triton epilogue recomputes the fused producer to write the four
tensor outputs. The timed oracle uses no eager/torch computation beyond tensor
allocation.

## Results

- `python -m py_compile repros/canonical/sum_sum_sum_ed6c74b5ef96/oracle_multi_output_reduction.py`: PASS
- `python repros/canonical/sum_sum_sum_ed6c74b5ef96/oracle_multi_output_reduction.py --check --no-skip-stochastic`: PASS; all eight outputs matched value tolerance, shapes, dtypes, and strides.
- `python repros/canonical/sum_sum_sum_ed6c74b5ef96/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50`
  - oracle full-scope Triton: `869.184 us`
  - `coordinate_descent_tuning=True`: `772.928 us`
  - `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `692.544 us`
  - historical `best_compile_us`: `598.0799794197083 us`

## Diagnosis

The missing optimization is not a valid faster floor on this run, but the fused
shape is clear: Inductor materializes the structured `scatter_add` result before
the four BN/ReLU backward branches consume it through sibling reductions and
dependent tensor epilogues. A useful compiler fix would be `SCATTER_REDUCE`
support that recognizes this max-pool-backward scatter pattern and lets
downstream channel reductions and epilogues consume the scatter producer without
forcing the full `[128, 288, 35, 35]` intermediate.
