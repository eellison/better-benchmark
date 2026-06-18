# Canonical headline numbers (genai/microbenchmarks EXCLUDED)

**Always cite the genai-EXCLUDED cut.** Microbenchmarks (genai/static, 8 models)
distort the model-level mean/median and must not be in the headline aggregate.

## Deliverable 1 — pytorch perf-branch model improvement (A/B, re-validated 2026-06-18)
- **median +2.18%, mean +4.33%** — 119 improved / 31 regressed / 8 flat, over 158 real models.
- A/A noise floor ±0.82% → the win is real, above noise.
- Re-validated on the dynamo-reset-fixed harness; reproduces prior (+2.23% median) within noise.
- Arms: baseline 244fdb379d11 vs branch tmp_work daa79cd25ca; fresh cache per arm; cache isolation + swap integrity confirmed.
- (genai-INCL cut, for reference only, NOT the headline: median +2.30% / mean +5.31% / 166 models.)

## Deliverable 2 — distance to agent (oracle) ceiling (v4 clean floors)
- genai-excluded by construction (upside_v2_v3.py drops genai model names).
- median remaining-headroom 0.4% → most models AT the agent ceiling.
- Top: Longformer 37.4% (one real, fixable kernel), swin 9.2%, vgg16 6.9%, BERT-train cluster ~4-5%.
- Contamination-free: >=5x points 265 -> 0 after the dynamo-reset fix + 69-dir blind-spot correction.
