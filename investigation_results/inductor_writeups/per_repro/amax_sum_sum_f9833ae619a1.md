# amax_sum_sum_f9833ae619a1

## Classification: `NEW_PATTERN`

## Pattern

Biased ignore-index cross-entropy mean:

- Labels: `i64[16, 128]` flattened to 2048 rows
- Logits: `f32[2048, 8008]`
- Bias: `f32[1, 8008]`
- Output: scalar `f32` loss mean

The repro computes `mm + arg665_1`, row log-softmax via `amax/sub/exp/sum/log`,
gathers the target log-probability with `-100` mapped to class 0, masks ignored
targets to zero, sums the numerator, counts valid targets, and divides.

## Measurements

Measured with `--warmup 10 --rep 50` on the full default shape, with deterministic
`-100` labels injected to exercise the ignore-index path.

| Metric | Value |
|--------|-------|
| Oracle biased online xent mean | `38.944 us` |
| `torch.compile coordinate_descent_tuning=True` | `77.184 us` |
| `torch.compile` combo-looped-CD config | `80.576 us` |

Correctness check:

- `ref=9.85848618`
- `oracle=9.85848522`
- `max_abs=9.536743e-07`
- `max_rel=9.673639e-08`
- `allclose=True`

## Diagnosis

The oracle folds the bias add into a single-pass online logsumexp row kernel,
loads the target logit directly, writes only per-row loss and valid-count
scalars, then performs a small scalar reduction for the final mean. Inductor
still treats the graph as decomposed log-softmax plus gather/mask/count/div work,
so it materializes and rereads the row-sized log-softmax intermediate instead of
computing the gathered cross-entropy value from the row accumulators. Closing
the gap needs a `NEW_PATTERN` lowering for biased
`log_softmax + gather + ignore-index masked mean` cross entropy.
