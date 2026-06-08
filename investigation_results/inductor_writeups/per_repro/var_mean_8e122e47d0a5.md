# var_mean_8e122e47d0a5 - oracle_residual_layernorm

## Status: AT_FLOOR (compile matches oracle)

- Oracle: 38.69 us
- Compile: 37.57 us
- Ratio: 0.971x

## Classification: NO_GAP

Oracle path: `repros/canonical/var_mean_8e122e47d0a5/oracle_residual_layernorm.py`

The compiled Inductor output slightly outperforms the oracle. No Inductor improvement needed.

## Details
- Models: hf Albert, Bert, DebertaV2, DistilBert, DistillGPT2, Electra, GPT2, GPTNeo, GoogleFnet, LayoutLM inference (11 instances)
- Pattern: Residual add + hidden-size population var_mean LayerNorm + affine + contiguous view
- Shape: [4096, 4096] f32 output
- Classification in oracle: SCHEDULER_FUSION -- Inductor's generic normalization schedule already handles view/add producer fusion efficiently at this shape
- Correctness: PASS
