# amax_sum_sum_b9ac8700504c

## Classification: `NEW_PATTERN`

## Pattern

Biased ignore-index cross-entropy mean:

- Logits: `f32[2048, 8008]`
- Bias: `f32[1, 8008]`
- Labels: `i64[16, 128]` flattened to 2048 rows
- Output: scalar `f32` loss mean

The repro computes `mm + arg2_1`, row log-softmax via
`amax/sub/exp/sum/log`, gathers the target log-probability with `-100` mapped
to class 0, masks ignored targets to zero, counts valid targets, and divides
the numerator by that count.

## Measurements

Measured with `--warmup 10 --rep 50` on the full default shape, with
deterministic `-100` labels injected to exercise the ignore-index path.

| Metric | Value |
|--------|-------|
| Oracle biased online xent mean | `40.960 us` |
| `torch.compile coordinate_descent_tuning=True` | `101.248 us` |
| `torch.compile` combo-looped-CD config | `107.200 us` |

Correctness check:

- `ref=9.89640808`
- `oracle=9.89640713`
- `max_abs=9.536743e-07`
- `max_rel=9.636570e-08`
- `allclose=True`

## Diagnosis

The oracle folds the bias add into a single-pass online logsumexp row kernel,
loads the target logit directly, writes only per-row loss and valid-count
scalars, then performs a small scalar reduction for the final mean. Inductor
still treats the graph as decomposed log-softmax plus gather/mask/count/div
work, so it materializes and rereads the row-sized log-softmax intermediate
instead of computing the gathered cross-entropy value from the row accumulators.
Closing the gap needs a `NEW_PATTERN` lowering for biased
`log_softmax + gather + ignore-index masked mean` cross entropy.
