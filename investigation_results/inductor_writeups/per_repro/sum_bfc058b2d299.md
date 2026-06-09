# sum_bfc058b2d299 - oracle_dropout_row_sum_fma

## Status: AT_FLOOR (compile matches oracle)

- Oracle: 13.09 us
- Compile: 12.99 us
- Ratio: 0.993x

## Classification: NO_GAP

Oracle path: `repros/canonical/sum_bfc058b2d299/oracle_dropout_row_sum_fma.py`

The compiled Inductor output matches the oracle. No Inductor improvement needed.

## Details
- Models: hf MT5, torchbench hf_T5_base, hf_T5 training (4 instances)
- Pattern: MT5/T5 dropout-masked probabilities-backward row update (bool mask conversion, f32 dropout scale, row product sum, fma epilogue)
- Shape: [192, 128, 128] f32 output
- Classification in oracle: BANDWIDTH_BOUND -- Inductor already emits one fused persistent-reduction kernel for this scope
- Correctness: PASS
