# any_amax_sum_727a4ae37fb6

## Classification: `BANDWIDTH_BOUND`

## Pattern

DistilBERT inference masked attention softmax:

- Scores input: `bmm_10`, `f32[3072, 128, 128]`, contiguous.
- Mask input: `expand_2`, `bool[256, 1, 128, 128]`, stride `(0, 128, 1, 0)`.
- Repro scope: view scores to `[256, 12, 128, 128]`, turn the broadcast bool
  mask into `0/-inf`, add it to scores, compute the `any(eq(-inf))` all-masked
  row guard, stable softmax over the last dimension, zero all-masked rows,
  expand to the same logical shape, and view back to `[3072, 128, 128]`.

The mask stride means each query row is either a normal score-row softmax or an
all-zero output row. The benchmark input has `64/128` query mask entries true.

## Measurements

Measured with `python repros/canonical/any_amax_sum_727a4ae37fb6/oracle_online_softmax.py --bench --warmup 10 --rep 50`.

| Metric | Value |
|--------|-------|
| Full-scope Triton diagnostic oracle | `91.360 us` |
| `torch.compile coordinate_descent_tuning=True` | `118.112 us` |
| `torch.compile` combo-looped-CD config | `118.336 us` |
| Historical queue `best_compile_us` | `73.37599992752075 us` |

Correctness check:

- Output shape: `[3072, 128, 128]`
- Output dtype: `torch.float32`
- Output stride: `(16384, 128, 1)`
- Default-input `max_abs=5.960464e-08`
- Forced masked/all-inf/partial-inf probe `max_abs=5.960464e-08`
- `allclose=True`

Valid floor: no. The oracle beats both required local compile configs in this
run, but it is slower than the historical queue best compile time. The main
queue `oracle_path` should remain blank; the oracle file is retained as a
diagnosis artifact only.

## Diagnosis

The oracle computes the full `Repro.forward` result in one Triton row-softmax
kernel and uses the original stride-zero mask directly, including masked score
loads for rows that become all-zero output. This avoids materializing the
decomposed mask/add/any/amax/exp/sum/div/where tensors, but it does not set a
true lower floor after applying the historical-best gate.

Inductor already recognizes this graph as an online-softmax persistent
reduction (`prepare_softmax_online`) and emits one full-scope fused kernel for
the repro. The remaining gap to memcpy SOL is the expected cost of the row
softmax math and f32 read/write traffic, not evidence of a missing profitable
fusion. No actionable Inductor change is recommended beyond preserving the
existing online-softmax/persistent-reduction lowering and tuning.
