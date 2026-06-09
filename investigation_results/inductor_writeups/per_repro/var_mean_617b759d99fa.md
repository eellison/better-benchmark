# var_mean_617b759d99fa

## Classification: NCHW_CHANNEL_LAYERNORM_TILING

## Current Result

- Family: `channel_layernorm`
- Oracle path: `repros/canonical/var_mean_617b759d99fa/oracle_channel_layernorm.py`
- Correctness: PASS
- Oracle: `32.45 us`
- `torch.compile coordinate_descent_tuning=True`: `34.3 us`
- Ratio: 1.057
- Best config: `default (cd=True)`: `34.3 us`
- Status: `real_gap` (marginal)

## Diagnosis

Same family as `var_mean_4df459ab207d` -- ConvNeXtV2 channel-LayerNorm on shape [128, 320, 14, 14]. This variant has a single output (no side rsqrt output), but the same tiling gap exists.

The oracle tiles spatial rows together for coalesced channel access in NCHW layout. Inductor uses a generic row reduction that does not optimize for the non-inner channel reduction dimension.

## Config exploration results

| Config | Time (us) |
|--------|-----------|
| Default (cd=True) | 34.30 |
| combo+mk=2 | 35.91 |
| combo+mk=3 | 35.92 |
| Oracle | 32.45 |

No config closes the gap. Multi-kernel configs make it slightly worse.

## Root cause

Same as `var_mean_4df459ab207d`: the 5.7% gap comes from the oracle's spatial-row grouped tiling for NCHW channel-LayerNorm. The gap is marginal and at the boundary of measurement noise.

## Kernel count
- Oracle: 1 kernel
- Inductor: 1 kernel

## Recommendation

Same as `var_mean_4df459ab207d`. Gap is marginal (5.7%). A dedicated NCHW channel-LayerNorm tiling rule would close it.

File references: `torch/_inductor/codegen/triton.py` (tiling heuristics).
