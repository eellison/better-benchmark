# amax_sum_any_5686b4ede363

## Classification: AT_FLOOR

## Oracle: oracle_full_bert_attention_softmax_dropout.py

## Measurements

- Compiled: 352.0 us
- Oracle: 345.5 us
- Ratio: 1.019x
- Oracle correctness: PASS (stochastic output skipped)

## Diagnosis

No meaningful performance gap (1.9%). The oracle computes BERT attention softmax with dropout in a fused persistent template, but Inductor's generic decomposed lowering matches the oracle within noise. Stochastic output prevents exact comparison (not_true_floor).

## Config exploration

Not needed -- already at floor (< 1.05x).

## Status: closed_no_gap (AT_FLOOR)
