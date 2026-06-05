# var_mean_1f499ddad801

## Classification: BAD_ORACLE

## Current Result

- Oracle path: `repros/canonical/var_mean_1f499ddad801/oracle_whisper_gelu_embedding_layernorm_aliases.py`
- Correctness: PASS
- Oracle: `28.58 us`
- `torch.compile coordinate_descent_tuning=True`: `25.47 us`
- Ratio: 0.891 (oracle 12% slower than compile)
- Status: `bad_oracle`

## Diagnosis

The oracle's Whisper GELU + embedding + layernorm + aliases kernel for [12000,384] fp16 is 12% slower than Inductor on this hardware. Inductor's norm template with fp16 handling outperforms the oracle's fused approach.

## Config exploration results
- No configs needed -- oracle is slower than baseline compile.
