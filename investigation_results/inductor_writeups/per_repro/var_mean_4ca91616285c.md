# var_mean_4ca91616285c


## Measured Timings
- Oracle: 238.40 us
- Compile (CDT): not available
- Ratio: N/A

## Classification: NORMALIZATION_SIDE_OUTPUT

## Current Result

- Oracle path: `repros/canonical/var_mean_4ca91616285c/oracle_affine_residual_layernorm_side.py`
- Default ratio: 1.388x
- Best config: `mk2` -> 1.372x
- Status: `real_gap`

## Diagnosis

The oracle computes a DINOv2 affine-residual LayerNorm scope (gamma multiply before residual add, var_mean, rsqrt, affine, side output) in one kernel. The oracle claims BANDWIDTH_BOUND but actually shows a 1.39x gap. mk3 and fast_math OOM on this large shape. mk2 improves marginally to 1.372. The gap appears to be due to Inductor materializing the residual producer before the normalization rather than inlining the affine-residual into the row reduction.

## Config exploration results

| Config | Ratio |
|--------|-------|
| Default (cd=True) | 1.388x |
| combo+mk=2 | 1.372x |
| combo+mk=3 | OOM |
| fast_math | OOM |
| Oracle | 1.000x |

## Root cause

No standard config fully closes this gap. The oracle computes a DINOv2 affine-residual LayerNorm scope (gamma multiply before residual add, var_mean, rsqrt, affine, side output) in one kernel. The oracle claims BANDWIDTH_BOUND but actually shows a 1.39x gap. mk3 and fast_math OOM on this large shape. mk2 improves marginally to 1.372. The gap appears to be due to Inductor materializing the residual producer before the normalization rather than inlining the affine-residual into the row reduction.

## Recommendation

Requires scheduler/codegen enhancement. See classification above.
