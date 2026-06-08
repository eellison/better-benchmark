# sum_sum_a25a5817ed69

## Status

- Family: `unclassified_priority_gap`
- Closure status: `at_floor`
- Artifact: `repros/canonical/sum_sum_a25a5817ed69/oracle_gated_product_channel_sum.py`
- Classification: `BANDWIDTH_BOUND`

## Full-Scope Contract

The oracle computes the complete GhostNet gated product-reduction scope,
multiplying the two `[512, 120, 28, 28]` inputs, reducing each `(N,C)` row over
spatial dimensions in f32, applying the captured `arg280_1 > -3.0 & arg280_1 < 3.0`
gate with the exact `0.16666666666666666` scale/full fallback, then finalizing
the `[120]` channel sum.

- Inputs: `T([512, 120, 28, 28], f32)` x2, `T([512, 120, 1, 1], f32)`, scalar
- Models: timm_ghostnet_100_train (3 variants)
- Correctness: PASS, max_diff=5.34e-05

## Timings

- Oracle: 116.22 us
- torch.compile (combo+CDT): 119.55 us
- Ratio: 1.029x (effectively at floor)

## Gap Diagnosis

Inductor already lowers this as the same bandwidth-heavy fused product plus
dependent reductions. The oracle mainly exposes a row-boundary partial schedule
without changing the math. Both paths must stream the two dense inputs and
produce one small reduction result. Classification: BANDWIDTH_BOUND -- no
action needed.

## Validation

```bash
python repros/canonical/sum_sum_a25a5817ed69/oracle_gated_product_channel_sum.py --check
python repros/canonical/sum_sum_a25a5817ed69/oracle_gated_product_channel_sum.py --bench --warmup 10 --rep 50
```
