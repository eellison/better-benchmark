# v2 Model-Level Floor Roll-Up — Validation (2026-06-16)

The migration's premise — per-pattern oracle floors compose to model-level
floors via correct new model->repro mappings — is VALIDATED end-to-end on v2.

## How it was unblocked
1. Restored repros_v2/models/ (891 files, 163 models) — an agent had slimmed
   them off the branch in 807aac037; recovered from 00c5097c8.
2. Root-caused a false "0/17 unmatched": model_graph_accounting.py hardcoded
   the OLD corpus path. v2 model->repro linkage is actually 100% (3208/3208
   patterns across 157 manifests resolve bidirectionally). Repointed the tool
   at repros_v2.
3. Added `--oracles` mode to scripts/bench_parallel.py (reuses native
   parallel orchestration) and ran a full sweep -> results/all_oracle_timings
   _b200_v2.json (1381 dirs with real oracle_us; ~189 check-gate-blocked
   bench-hack/stochastic oracles correctly carry no floor).

## Results (8 fully-covered models, --timings roll-up)
| model | matched | model_floor_us | compile_us | speedup |
|---|---|---|---|---|
| alexnet | 14/14 | 318.0 | 411.0 | 1.29x |
| vgg16 | 14/14 | 547.8 | 743.1 | 1.36x |
| mnasnet1_0 | 14/14 | 559.7 | 725.3 | 1.30x |
| squeezenet1_1 | 18/18 | 589.7 | 623.2 | 1.06x |
| BertForMaskedLM | 63/63 | 6433.4 | 10142.2 | 1.58x |
| DebertaV2ForMaskedLM | 100/100 | 13590.6 | 14111.1 | 1.04x |
| GPTNeoForCausalLM | 30/30 | 7598.0 | 7679.1 | 1.01x |
| beit_base_patch16_224 | 19/19 | 7446.1 | 5425.2 | FLAG (floor>compile) |

All models: 0 unmatched partitions. fusible_oracle_us_total = SUM(oracle_us x
occurrences), arithmetically verified (alexnet 318.0 == 318.03 reported).

## Follow-ups (non-blocking)
- beit_base_patch16_224: floor (7446us) EXCEEDS compile (5425us) — a floor
  should not exceed compile. Investigate: untimed patterns dropping
  compile-side work, or an oracle slower-than-compile on some shapes.
- ~189 oracles have no floor (check-gate-blocked bench-hacks/stochastic);
  expected, excluded from roll-up.
