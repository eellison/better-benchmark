# amax_sum_557afc9e24ea

## Classification: AT_FLOOR

## Oracle: oracle_full_t5_attention_softmax_dropout.py

## Measurements

- Compiled: 266.3 us
- Oracle: 255.1 us
- Ratio: 1.044x
- Oracle correctness: PASS (stochastic output skipped)

## Diagnosis

No meaningful performance gap (4.4%). The oracle computes T5 attention softmax with dropout in a fused persistent template, but Inductor's generic decomposed lowering is within noise of the oracle performance. Stochastic output prevents exact comparison.

## Config exploration

Not needed -- already at floor (< 1.05x).

## Status: closed_no_gap (AT_FLOOR)
