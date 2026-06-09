# var_mean_1bab7e80cec1


## Measured Timings
- Oracle: 6.69 us
- Compile (CDT): 7.17 us
- Ratio: 1.07x

## Classification: AT_FLOOR

## Current Result

- Oracle path: `repros/canonical/var_mean_1bab7e80cec1/oracle_selected_token_pair_layernorm.py`
- Oracle: measured
- Ratio: 1.023x
- Status: `at_floor`

## Diagnosis

Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete DeiT residual-add, fp32 population var_mean(correction=0), `libdevice.rsqrt(var + 1e-6)`, affine LayerNorm, and two token-select outputs by reducing only token indices 0 and 1 for each batch while returning views from a full-stride `[128,198,768]` storage, whereas Inductor lowers the same row-independent LayerNorm producer over all 25,344 token rows before returning two `select(..., 1, {0,1})` views; Inducto

## Config exploration results

- Inductor already matches or beats the oracle (ratio <= 1.05x). No config exploration needed.
- Status: Small gap (1.023x) within noise threshold
