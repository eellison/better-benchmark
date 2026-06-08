# var_mean_b1a4a4630c41

## Status: BAD_ORACLE (compile already faster)

- Oracle: 21.79 us
- Compile: 20.58 us
- Ratio: 0.944x (oracle is 1.06x slower than compile)

## Classification: NO_GAP

Oracle path: `repros/canonical/var_mean_b1a4a4630c41/oracle_swin_window_layernorm.py`

The compiled Inductor output already slightly outperforms this oracle. No Inductor improvement needed.

## Details
- Pattern: Swin window layernorm
- Shape: [25088, 512]
- Correctness: PASS
