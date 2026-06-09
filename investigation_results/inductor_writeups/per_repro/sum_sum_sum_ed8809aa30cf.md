# sum_sum_sum_ed8809aa30cf

Full-scope Triton oracle artifact for `hf_BlenderbotForCausalLM_train_007`.

## Gap Diagnosis

Classification: `COOPERATIVE_SPLIT_K`. The oracle consumes the same inputs as `repro.py` and returns the same two `[2560]` reductions plus the full `[32, 128, 2560]` epilogue tensor; it computes the shared `mm_6 + mm_8 + mm_10` producer in a split-tile Triton reduction that forms both row-local scalar accumulators and column partials, then finalizes those partials and recomputes the cheap producer for the dependent epilogue. Inductor cannot do this today because the graph mixes hidden-dimension reductions, batch/sequence reductions, and a dependent pointwise consumer, so the scheduler emits separate reductions and epilogue work instead of one coordinated multi-output reduction plan. The Inductor fix would be `COOPERATIVE_SPLIT_K`: support cross-axis multi-output reductions with row/column partials and recompute in the dependent epilogue.

## Scope

The timed oracle covers the full compiled repro scope:

- `add_tensor_1 = mm_6.view + mm_8.view + mm_10.view`
- row-local reductions over hidden dimension for `sum_dim_int_list` and `sum_dim_int_list_1`
- column reductions for `sum_dim_int_list_2` and `sum_dim_int_list_3`
- full returned epilogue `add_3 + (arg10_1 / 2560) * (...)`

It is not a subset reduction benchmark.

## Results

Correctness passed with matching output shapes and strides:

- output 0 `[2560]`: max_abs `9.155273e-05`, max_rel `5.237560e-04`
- output 1 `[2560]`: max_abs `6.103516e-05`, max_rel `4.007212e-04`
- output 2 `[32, 128, 2560]`: max_abs `7.629395e-06`, max_rel `7.640111e-02`

Benchmark (updated June 2026, GPU-locked interleaved):

- oracle: 78.72 us
- compile (default): 88.93 us
- ratio: 1.13x (oracle is faster -- valid gap)

Previous benchmark with `--warmup 10 --rep 50`:

- oracle full-scope Triton: `134.848 us`
- `coordinate_descent_tuning=True`: `150.208 us`
- `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `129.920 us`

Valid floor: yes, the oracle now consistently beats default compile by 1.13x with GPU-locked interleaved measurement.
