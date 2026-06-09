# sum_a9e8fa1651a2 - oracle_broadcast_row_sum

## Status: AT_FLOOR (compile matches oracle)

- Oracle: 102.02 us
- Compile: 99.10 us
- Ratio: 0.971x

## Classification: NO_GAP

Oracle path: `repros/canonical/sum_a9e8fa1651a2/oracle_broadcast_row_sum.py`

The compiled Inductor output slightly outperforms the oracle. No Inductor improvement needed.

## Details
- Models: hf DebertaV2ForMaskedLM training (2 instances)
- Pattern: Dropout-scaled attention row-sum backward epilogue with broadcast-mask where
- Shape: [192, 512, 512] f32 output
- Classification in oracle: BANDWIDTH_BOUND -- Inductor already emits the full-scope persistent reduction without materializing intermediates
- Correctness: PASS
