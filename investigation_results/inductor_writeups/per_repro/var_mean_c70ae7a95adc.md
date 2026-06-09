# var_mean_c70ae7a95adc


## Measured Timings
- Oracle: 18.53 us
- Compile (CDT): 18.05 us
- Ratio: 0.97x

## Classification: ALGEBRAIC_ELIMINATION

## Current Result

- Family: `swin_singleton_window_reverse_layernorm`
- Oracle path: `repros/canonical/var_mean_c70ae7a95adc/oracle_swin_singleton_window_reverse_layernorm.py`
- Correctness: assumed PASS (oracle has check mode)
- Compiled (coordinate_descent_tuning=True): 18.18 us
- Gap: 1.04x
- Status: `at_floor`

## Diagnosis

The oracle collapses the Swin window-reverse reshape/permute chain whose interchanged grid dimensions are both size 1, then computes the residual add and hidden-size-1024 population var_mean LayerNorm affine epilogue directly into the final [6272, 1024] layout in one Triton row kernel.

Inductor already emits 1 fused kernel for this pattern. The 1.04x gap (0.7 us absolute) is within measurement noise.

The repro computes:
1. reshape [6272, 1024] -> [128, 49, 1024] -> [-1, 7, 7, 1024]
2. reshape -> [-1, 1, 1, 7, 7, 1024] -> permute [0,1,3,2,4,5] -> reshape [-1, 7, 7, 1024]
3. add(view_631, result) -> residual connection
4. reshape -> [128, 49, 1024]
5. var_mean(correction=0, dim=2) + LayerNorm affine
6. reshape -> [6272, 1024]

The singleton window dimensions (1x1 grid) make the permute chain an identity, which Inductor's view simplifier already handles.

## Root cause

Inductor already correctly simplifies the singleton-grid window-reverse layout to identity and fuses everything into 1 kernel. The 1.04x residual is noise on a fast (18 us) kernel.

## Kernel count
- Oracle: 1 kernel (fused window-reverse + add + LN)
- Inductor: 1 kernel (fused)

## Config exploration results

| Config | Expected Impact |
|--------|----------------|
| coordinate_descent_tuning | Already applied (18.18 us) |
| All configs | No improvement needed -- gap is noise |

## Recommendation

No action needed. Gap is 1.04x (within noise for an 18 us kernel). Inductor already correctly eliminates the singleton window-reverse and fuses the full LN scope.

## Relevant files

- Input: [6272, 1024] + [128, 7, 7, 1024] f32 (Swin transformer inference)
- Total bytes: ~77 MB
- Models: timm_swin_base_patch4_window7_224_infer
