# var_mean_2ec780efd8cf

## Classification: ALGEBRAIC_ELIMINATION

## Current Result

- Oracle path: `repros/canonical/var_mean_2ec780efd8cf/oracle_selected_layernorm_side.py`
- Default ratio: 1.366x
- Best config: `fast_math` -> 1.343x
- Status: `real_gap`

## Diagnosis

The oracle computes the complete residual-add hidden-size-768 LayerNorm scope, including the live rsqrt/768 side output, and the two non-contiguous selected token views sharing one backing buffer, while Inductor materializes the full [128,198,768] affine tensor before returning only token 0 and token 1. No config closes the 1.37x gap. The fix requires normalization epilogue liveness to push final select users into the row-reduction store plan while preserving alias-view metadata.

## Config exploration results

| Config | Ratio |
|--------|-------|
| Default (cd=True) | 1.366x |
| combo+mk=2 | 1.605x |
| combo+mk=3 | 1.349x |
| fast_math | 1.343x |
| Oracle | 1.000x |

## Root cause

No standard config fully closes this gap. The oracle computes the complete residual-add hidden-size-768 LayerNorm scope, including the live rsqrt/768 side output, and the two non-contiguous selected token views sharing one backing buffer, while Inductor materializes the full [128,198,768] affine tensor before returning only token 0 and token 1. No config closes the 1.37x gap. The fix requires normalization epilogue liveness to push final select users into the row-reduction store plan while preserving alias-view metadata.

## Recommendation

Requires scheduler/codegen enhancement. See classification above.
