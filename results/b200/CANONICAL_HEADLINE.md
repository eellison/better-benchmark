# Canonical headline numbers (genai/microbenchmarks EXCLUDED)

**Always cite the genai-EXCLUDED cut.** Microbenchmarks (genai/static, 8 models)
distort the model-level mean/median and must not be in the headline aggregate.

## Deliverable 1 — pytorch perf-branch model improvement (A/B, re-baselined 2026-06-19)
- **median +2.18%, mean +4.05%, geomean +4.51%** — 120 improved / 28 regressed / 10 flat, over 158 real models.
- A/A noise floor ±0.82% → the win is real, above noise.
- **Arms: baseline `5e2ab3055de` (the perf branch's TRUE base) vs branch `tmp_work daa79cd25ca`**; fresh cache per arm; cache isolation + swap integrity confirmed. `compute_ab.py` default reproduces these.
- (genai-INCL cut, for reference only, NOT the headline: median +2.46% / mean +5.04% / geomean +6.02% / 166 models.)
- ‡ **Earlier-cited contaminated number** used baseline `244fdb379d11` (median +2.18% / mean +4.33% / geomean +4.84%). That baseline is an older ancestor with **56 upstream pytorch PRs** between it and `5e2ab`; re-baselining leaves the **median UNCHANGED** and trims mean/geomean ~0.3pp. The whole delta = one upstream cross-entropy sum-reduction kernel (`sum_81b4fd73f8d1`) mis-credited to 2 train models (TrOCR, Pegasus). See `results/perf_ab/rebaseline_5e2ab_2026-06-19/`.
- **compile_us variance:** individual per-model A/B ratios (and oracle-vs-compile ratios) move run-to-run; the STABLE deliverables are the oracle FLOORS + the D2 ranking. Cite floors as durable, individual compile ratios as point-in-time.

## Deliverable 2 — distance to agent (oracle) ceiling (v4 clean floors)
- genai-excluded by construction (upside_v2_v3.py drops genai model names).
- median remaining-headroom 0.4% → most models AT the agent ceiling.
- Top: Longformer 37.4% (one real, fixable kernel), swin 9.2%, vgg16 6.9%, BERT-train cluster ~4-5%.
- Contamination-free: >=5x points 265 -> 0 after the dynamo-reset fix + 69-dir blind-spot correction.
