# sum_a58bd57ea950

## Status

- Family: `multi_output_reduction_templates`
- Claimed owner: `Codex-bottom-multi-a58b`
- Classification: `BANDWIDTH_BOUND`
- Oracle artifact: `repros/canonical/sum_a58bd57ea950/oracle_multi_output_reduction.py`
- Oracle status: diagnosis-only, not a true floor
- Parent integration note: leave the main queue `oracle_path` blank for true-floor tracking because the artifact loses the historical-best gate.

## Full-Scope Contract

The oracle consumes the same inputs as `repro.py`:

- `mm_66`: contiguous `float32[131072, 288]`
- `arg225_1`: contiguous `float32[131072, 288]`
- shape parameters for the `[512, 256, 288]`, `[131072, 288]`, and `[288]` views

It computes the complete captured expression:

```text
value[row, col] = mm_66[row, col]
                * sigmoid(arg225_1[row, col])
                * (arg225_1[row, col] * (1 - sigmoid(arg225_1[row, col])) + 1)
```

and returns both captured outputs:

- `float32[288, 131072]` transpose view with stride `(1, 288)`
- `float32[288]` column sum with stride `(1,)`

The timed path is Triton-only after allocation: one tiny zeroing kernel and one
full-scope tiled producer that stores the transpose side output and atomically
accumulates the column sums. It is not a reduction-subset microbenchmark.

## Measurements

Measured with:

```bash
python repros/canonical/sum_a58bd57ea950/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50
```

- Triton full-scope oracle: `142.816 us`
- `torch.compile` with `coordinate_descent_tuning=True`: `178.944 us`
- `torch.compile` with `combo_kernels=True,combo_kernel_per_subkernel_blocks=True,coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3`: `178.528 us`
- Historical queue `best_compile_us`: `87.80799806118011 us`

The oracle is faster than both required local compile configs on this run, but
it is slower than the historical queue best. It is therefore diagnosis-only and
should not be used as a true performance floor.

## Gap Diagnosis

The oracle differs from Inductor by forcing a hand-written full-scope
multi-output schedule: the same Triton tile computes the SiLU-gradient producer,
writes the required dense transposed side output through its physical
row-major storage offset, and accumulates the per-column sum. This tests the
idea that Inductor is leaving performance on the table by not explicitly
combining a materialized transpose side-output with the same-producer column
reduction.

Inductor cannot express that exact schedule today because the generic scheduler
does not model a required dense side-output plus a sibling reduction as one
pre-zeroed atomic multi-output reduction template. The scheduler-fusion change
that would address the hypothesis is to let multi-output reductions emit a
fused producer that writes dense side outputs while accumulating compatible
reduction outputs from the same computed value.

The historical-best gate rejects this as a true floor: a prior compile result
is `87.80799806118011 us`, substantially faster than the `142.816 us` oracle.
The practical classification for queue integration is `BANDWIDTH_BOUND` /
diagnosis-only rather than an actionable new floor.

## Validation

- `python -m py_compile repros/canonical/sum_a58bd57ea950/oracle_multi_output_reduction.py`: passed
- `python repros/canonical/sum_a58bd57ea950/oracle_multi_output_reduction.py --check`: passed; output 0 max abs `9.536743e-07`, output 1 max abs `4.882812e-04`, shapes/dtypes/strides matched
- `python repros/canonical/sum_a58bd57ea950/oracle_multi_output_reduction.py --bench --warmup 10 --rep 50`: completed; diagnosis-only due to historical-best gate
