# sum_5f7b790ecd66

## Status: AT_FLOOR (compile matches oracle)

- Oracle: 190.08 us
- Compile: 192.22 us
- Ratio: 1.011x

## Classification: NO_GAP

Oracle path: `repros/canonical/sum_5f7b790ecd66/oracle_dropout_softmax_backward.py`

The compiled Inductor output already matches the oracle performance (within 1.1%). No Inductor improvement needed.

## Details
- Pattern: dropout softmax backward
- Shape: [384, 512, 512]
- Correctness: PASS
