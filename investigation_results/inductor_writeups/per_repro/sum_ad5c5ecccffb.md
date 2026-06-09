# sum_ad5c5ecccffb - oracle_select_scatter_sum

## Status: BAD_ORACLE (compile beats oracle)

- Oracle: 33.50 us
- Compile: 29.82 us
- Ratio: 0.890x

## Classification: NO_GAP

Oracle path: `repros/canonical/sum_ad5c5ecccffb/oracle_select_scatter_sum.py`

The compiled Inductor output outperforms the oracle by 11%. No Inductor improvement needed.

## Details
- Model: torchbench BERT_pytorch training
- Pattern: BERT dropout-mask product, select_scatter bias, returned transposed view, and sibling hidden-dimension sum
- Shapes: [768, 16384] f32 (layout) + [768] f32 (sum)
- Classification in oracle: BANDWIDTH_BOUND -- Inductor already fuses the materialized side output with first-stage reduction partials
- Correctness: PASS
