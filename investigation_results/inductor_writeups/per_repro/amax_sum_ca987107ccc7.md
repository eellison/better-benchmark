# amax_sum_ca987107ccc7

## Classification: `NEW_PATTERN`

## Pattern

GPT-Neo attention masked softmax materialization:

- `unsqueeze`: `i64[1, 128]`
- `cumsum`: `i64[32, 128]`
- `bmm`: `f32[512, 128, 128]`, viewed as `f32[32, 16, 128, 128]`
- `arg8_1`: `bool[1, 1, 2048, 2048]`, sliced to `[:128, :128]`
- Output: `f32[512, 128, 128]`

The repro builds a causal/segment mask from `unsqueeze` and indexed `cumsum`,
applies the large attention mask slice, adds both masks to the score tensor, and
returns the row softmax over the last dimension.

## Measurements

Measured with `--warmup 10 --rep 50` on the full default shape.

| Metric | Value |
|--------|-------|
| Oracle full-scope masked softmax | `67.200 us` |
| `torch.compile coordinate_descent_tuning=True` | `86.784 us` |
| `torch.compile` combo-looped-CD config | `78.144 us` |

Correctness check:

- Output shape: `[512, 128, 128]`
- Max abs diff: `2.384186e-07`
- Max rel diff: `4.507227e-07`
- Stride match: `True`
- `allclose=True`

## Diagnosis

The oracle uses one Triton program per `[128]` row, recomputes the
causal/segment predicate and the large attention-mask slice in registers, applies
the same `where`/add semantics to the score row, and normalizes that row without
materializing either mask tensor or the masked-score intermediate. Inductor
currently treats the graph as separate advanced-index mask construction,
broadcasted mask application, score add, and softmax work over a materialized
`[32, 16, 128, 128]` tensor. Closing the gap needs a `NEW_PATTERN` masked
softmax lowering that accepts structured attention predicates and recomputes
cheap boolean mask logic inside the row template.
