# pointwise_f9aca3f32296

## Oracle

- Oracle path: `repros/canonical/pointwise_f9aca3f32296/oracle_relu_le.py`
- Scope: full graph, `relu(addmm_5) <= 0` over `f32[2048, 1]`, returning `bool[2048, 1]`
- Classification: `BANDWIDTH_BOUND`

## Gap Diagnosis

Classification: BANDWIDTH_BOUND. The oracle computes the full graph with one Triton timed kernel and replaces `relu(addmm_5) <= 0` with `addmm_5 <= 0`, which is equivalent for this generated f32 input including normal signed-zero and NaN comparison behavior, but the measured runtime is still at the same launch/allocation-scale floor as the compiled Inductor pointwise kernel. Inductor already emits a fused tiny pointwise kernel, and the remaining runtime is dominated by fresh bool output allocation plus one CUDA launch rather than the eliminated relu arithmetic.

## Measurements

- Oracle `--bench --warmup 10 --rep 50`: oracle 3.71 us, compiled 3.81 us, ratio 1.026x, status `AT_FLOOR`.
- `bench_compare.py`: `coordinate_descent_tuning=True` 4.03 us; combo looped CD config 3.97 us, 1.016x faster.
