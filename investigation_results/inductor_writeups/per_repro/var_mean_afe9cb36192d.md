# var_mean_afe9cb36192d

## Status: AT_FLOOR (compile matches oracle)

- Oracle: 33.73 us
- Compile: 33.6 us
- Ratio: 0.996x

## Classification: NO_GAP

Oracle path: `repros/canonical/var_mean_afe9cb36192d/oracle_layoutlm_embedding_layernorm.py`

The compiled Inductor output already matches the oracle performance. No Inductor improvement needed.

## Details
- Pattern: LayoutLM embedding layernorm (3 outputs)
- Shape: [16384, 768]
- Correctness: PASS
