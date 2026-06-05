# pointwise_b7a851a9ac93 — Constant Bool Mask (DeBERTa)

## Summary
- **Model**: hf_DebertaV2ForMaskedLM_infer_000
- **Classification**: BAD_ORACLE
- **Ratio**: 0.463x (oracle 12.22us vs compile 5.66us)
- **Status**: Oracle is much slower than compile; no gap to investigate

## Analysis

The repro generates 24 identical `bool[8, 1, 512, 512]` attention masks from a constant all-ones [8, 512] tensor. The computation is `full(1) -> unsqueeze -> mul -> convert_to_bool` producing 24 copies of the same all-True mask.

Inductor is 2.16x faster than the oracle. Inductor likely recognizes that:
1. The result is entirely constant (all True)
2. The 24 outputs can share storage or be computed with a single memset

The oracle's Triton kernel performs unnecessary computation on data that Inductor can constant-fold.

## No Action Required

Oracle is much slower than compile. Inductor already handles this pattern optimally through constant folding or output aliasing.
