# var_mean_aabf4b70dc64

## Status: AT_FLOOR (compile matches oracle)

- Oracle: 21.73 us
- Compile: 21.38 us
- Ratio: 0.984x

## Classification: NO_GAP

Oracle path: `repros/canonical/var_mean_aabf4b70dc64/oracle_residual_layernorm_aliases.py`

The compiled Inductor output already matches (or slightly exceeds) the oracle performance. No Inductor improvement needed.

## Details
- Pattern: residual layernorm with aliases (24 outputs)
- Shape: [8192, 1024]
- Correctness: PASS
