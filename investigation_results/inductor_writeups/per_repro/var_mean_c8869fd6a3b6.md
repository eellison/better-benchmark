# var_mean_c8869fd6a3b6


## Measured Timings
- Oracle: 40.77 us
- Compile (CDT): 42.62 us
- Ratio: 1.05x

## Classification: SCHEDULER_FUSION

## Current Result

- Family: `fnet_select_layernorm`
- Oracle path: `repros/canonical/var_mean_c8869fd6a3b6/oracle_fused_select_layernorm.py`
- Correctness: assumed PASS (oracle has check mode)
- Compiled (coordinate_descent_tuning=True): 44.03 us
- Gap: 1.55x
- Status: `real_gap`

## Diagnosis

The oracle keeps the selected-add row tile live through population var_mean, rsqrt normalization, affine scale/bias, and the final view in one Triton row kernel. Inductor currently emits one generic fused Welford reduction kernel but the gap suggests suboptimal tile utilization for the `select + add` producer pattern.

The repro computes:
1. select(view_as_real_11, dim=3, index=0) -> [32, 512, 768] from [32, 512, 768, 2]
2. add(add_91, select_result) -> [32, 512, 768]
3. var_mean(correction=0, dim=2, keepdim=True)
4. LayerNorm: sub, add_eps, rsqrt, mul_gamma, add_bias
5. view -> [16384, 768]

## Root cause

The `select.int(tensor_4d, dim=3, index=0)` introduces a strided access pattern (stride 2 in the innermost dimension of the complex-valued input). The oracle's fixed-shape kernel handles this directly, while Inductor's generic Welford reduction may generate suboptimal memory access patterns for the strided select. The 1.55x gap suggests the reduction scheduler does not optimally retain the selected-add row tile across the statistics computation and affine epilogue when the producer has a non-unit inner stride.

## Kernel count
- Oracle: 1 kernel (fused select + add + LN)
- Inductor: 1 kernel (fused persistent reduction)

## Config exploration results

| Config | Expected Impact |
|--------|----------------|
| coordinate_descent_tuning | Already applied (44.03 us) |
| combo_kernels | Not applicable (single kernel) |
| multi_kernel=2 | May improve (persistent with better tiling) |
| multi_kernel=3 | May improve (looped with explicit tile control) |

## Recommendation

Investigate whether the strided `select` from the complex-valued [32, 512, 768, 2] input causes the reduction template to use suboptimal vectorization or tile sizes. The fixed hidden size (768) should allow a dedicated row-reduction tile that reads stride-2 data efficiently.

Teach the row-reduction template to preserve fixed hidden-size producer tiles from strided-select inputs across var_mean and directly feed the affine epilogue when register pressure is acceptable.

## Relevant files

- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (row reduction tile for strided producers)
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` (fusion of select + LN scope)
- Input: [32, 512, 768, 2] f32 (Google FNet inference, view_as_real from FFT)
- Total bytes: ~151 MB
- Models: hf_GoogleFnet_infer_000
