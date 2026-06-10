# online_softmax_cross_entropy family — June 2026 investigation

Branch: pytorch-work pr-184905. Hardware: B200. Fresh cache per bench.

## Triage (fresh re-bench 2026-06-10, oracle*.py --bench)

| repro_id | oracle_us | compile_us | abs gap (us) | ratio | verdict | pattern |
|----------|-----------|------------|--------------|-------|---------|---------|
| amax_sum_sum_6fd07d12d98a | 520.3 | 738.3 | 218.0 | 1.42 | REAL | DistillGPT2 infer: shifted-label ignore-index CE mean, f32[16384, 50257] |
| amax_sum_sum_86d05d6810f4 | 521.0 | 739.0 | 218.0 | 1.42 | REAL | DistillGPT2 train: same pattern |
| amax_sum_sum_e2f518f0a274 | 91.0 | 149.2 | 58.2 | 1.64 | REAL | GPT2 shifted causal-LM CE mean, [2048, 50257] |
| amax_sum_d112f48ea917 | 75.4 | 101.0 | 25.6 | 1.34 | REAL | Reformer stable softmax (where(inf) guard) |
| amax_sum_f5253e4f250e | 337.5 | 358.0 | 20.6 | 1.06 | near floor | |
| amax_sum_sum_0e37ca9164b3 | 33.7 | 43.9 | 10.2 | 1.30 | REAL | Blenderbot ignore-index CE mean |

Skipped per task: amax_sum_9940b361e5b4 (Longformer band assembly — separate blocker).

Stale-jsonl agreement: jsonl ratios match fresh measurements within noise; family genuinely open.

## Status: WIP — blocker analysis in progress
