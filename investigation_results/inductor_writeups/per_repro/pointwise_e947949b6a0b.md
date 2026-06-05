# pointwise_e947949b6a0b

## Classification: AT_FLOOR

## Oracle: oracle_reformer_dropout_relu_layout.py

## Measurements

- Compiled: 61.3 us
- Oracle: 63.4 us
- Ratio: 0.967x (oracle 3% slower)
- Oracle correctness: PASS (stochastic output skipped)

## Diagnosis

The oracle (reformer dropout + ReLU layout) is marginally slower than torch.compile output. Inductor already achieves optimal performance for this stochastic pointwise pattern with shape [32768, 512]. No gap to investigate.

## Status: closed_no_gap (AT_FLOOR, stochastic)
