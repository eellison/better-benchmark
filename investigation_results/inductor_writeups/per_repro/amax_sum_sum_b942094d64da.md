# amax_sum_sum_b942094d64da

## Classification: `SCHEDULER_FUSION`

## Pattern

ResNeSt split-attention weighted pooling:

- Softmax logits: `convolution_24`, `f32[32, 1024, 1, 1]`
- Branch input: `view_15`, `f32[32, 2, 512, 14, 14]`
- Output: `avg_pool2d(sum(view_15 * softmax_weights, dim=1))`, `f32[32, 512, 7, 7]`

The repro reshapes the logits to `[32, 2, 1, 512]`, computes a stable two-way
softmax across the branch dimension, reshapes weights to `[32, 2, 512, 1, 1]`,
multiplies the two spatial branches, sums the branches, then applies
`avg_pool2d(kernel=3, stride=2, padding=1, count_include_pad=True)`.

## Measurements

Measured on NVIDIA GH200 with `--warmup 10 --rep 50` on the full default shape.

| Metric | Value |
|--------|-------|
| Oracle full-scope softmax-weighted avgpool | `12.096 us` |
| `torch.compile coordinate_descent_tuning=True` | `16.992 us` |
| `torch.compile` combo-looped-CD config | `17.504 us` |
| Historical queue `best_compile_us` | `17.536 us` |

Correctness check:

- Output shape: `[32, 512, 7, 7]`
- Output dtype: `torch.float32`
- Output stride: `(25088, 49, 7, 1)`
- `max_abs=2.980232e-07`
- `allclose=True`

Valid floor: yes. The full-scope Triton oracle is faster than both required
measured compile configs and the historical queue best compile time.

## Diagnosis

The oracle computes the two-branch stable softmax weights, applies them to both
`view_15` branches, sums the branches, and folds the padded 3x3 stride-2 average
pool into one output-tiled Triton kernel, so it avoids writing and rereading the
intermediate weighted `[32,512,14,14]` tensor and pays one kernel launch for the
full repro computation. Inductor currently schedules the decomposed
view/permute/amax/exp/sum/div softmax, weighted branch sum, and avg-pool as
generic operations, and it does not fuse this branch reduction with the
following spatial pooling reduction into a single layout-specialized loop nest;
closing the remaining gap needs `SCHEDULER_FUSION` support for this chained
small-reduction plus pooling epilogue pattern.
