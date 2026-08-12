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

## E2E reconciliation on v2 (2026-06-16)

Re-confirmed the roll-up reconciles with REAL full-graph runs on v2 (prior
~5% result was old-corpus). Method: model_graph_accounting --timings
(fusible_oracle_us_total vs fusible_compile_us_total, same partitions) +
bench_parallel --full-graphs for true e2e.

| model | oracle Σus | compile Σus | oracle/compile | e2e us | compile/e2e |
|---|---|---|---|---|---|
| LayerNormForward (control, float in) | 351.5 | 343.8 | 1.022 | 342.9 | 1.003 |
| DistilBertForMaskedLM | 873.6 | 1999.2 | 0.437 | 5525.6 | 0.362 |
| GPTNeoForCausalLM | 2116.3 | 4877.3 | 0.434 | 13986.8 | 0.349 |
| BlenderbotForCausalLM | 352.6 | 429.1 | 0.822 | 33230.9 | 0.013 |

- LayerNorm control: real e2e == compile-Σ to 0.24%, oracle roll-up +2.5% — inside 5% target. Arithmetic + timings JSON wired correctly.
- compile-Σ < e2e for transformers BY DESIGN: oracle roll-up is fusible-only; the rest is non-fusible conv/mm/sdpa (out of scope). Fusible fraction ~35% (Bert-family), 1.3% (Blenderbot, attention-dominated).
- Modeled speedup (deliverable): oracle/compile ~0.44 transformer pointwise/norm (~2.3x over compile fusion), ~1.0 LayerNorm.
- 0 unmatched partitions on all 8 models. beit-style floor>compile only on hf/train/AlbertForMaskedLM (1.107x, within ~10%, training backward).
- Caveat: HF-infer full graphs need --allow-unsafe-full-graphs (int token inputs) for clean true-e2e; LayerNorm (float) is the clean load-bearing check.

## Rigorous HF-infer e2e (2026-06-16) — caveat closed

Ran 4 HF-infer full graphs e2e with VALID integer inputs (no
--allow-unsafe-full-graphs; v2 meta sidecars already carry correct
gen:['index',low,high] ranges for token/position/type ids). CUDAGraph + CD +
bench-lock + fresh cache, min-of-5, numerics-checked vs eager (no NaN).

| model | real_e2e_us | fus_compile_Σ | fus_oracle_Σ | modeled fusible speedup | nonfusible GEMM_us (%e2e) |
|---|---|---|---|---|---|
| DistilBertForMaskedLM | 5076 | 1999 | 874 | 2.29x | 3452 (68%) |
| BertForMaskedLM | 6016 | 3485 | 1316 | 2.65x | 3998 (66%) |
| ElectraForCausalLM | 3889 | 3669 | 1593 | 2.30x | 2490 (64%) |
| AlbertForMaskedLM | 15802 | 3244 | 1282 | 2.53x | 13691 (87%) |

- Non-fusible GEMMs (addmm/bmm/mm, measured directly with exact arg shapes x occ)
  explain 64-87% of e2e — a true lower bound.
- nonfusible + fusible_compile = 107-158% of e2e (never under-counts).
- The fusible ROLL-UP over-counts the in-graph fusible remainder by 1.2-2.6x:
  it sums 72-138 independent partition launches but the full graph fuses into
  56-104 kernels. Excess = partition-boundary launch-floor (5.4us/kernel B200)
  + re-materialized partition I/O elided when fused. Largest for small/numerous
  partitions (Electra hidden=256, 158%), smallest for GEMM-dominated (DistilBert/
  Albert, 107%). This is the documented Σparts-vs-e2e launch-floor correction,
  not a roll-up error.
- VERDICT: accounting reconciles with real HF-infer e2e once non-fusible ops +
  launch-floor correction are applied. Per-pattern oracle floors sound; naive
  Σ over-counts by a known/correctable launch factor. Modeled fusible speedup
  2.3-2.65x (~1100-2200us/model). LayerNorm-only caveat closed.

## CORRECTION (2026-06-16): launch-floor correction resolves the 107-158% overshoot

The HF e2e section above reported `nonfusible + fusible_compile = 107-158% of
e2e` and wrongly let it stand as "expected." That was the RAW overshoot with
the launch-floor correction NOT applied. Reproducing the dress rehearsal with
the canonical tool scripts/model_attribution.py (e2e ~= Σfusible×occ +
Σextern×occ − G×(n_occ−n_graphs), G measured = 4.7us on B200) shows the
accounting reconciles to ~5-9%, matching the original 0.99-1.08 result:

timm (conv family): raw 111-131% -> corrected **100-109%** (0 unmatched)
| model | real_e2e | raw %e2e | corrected %e2e |
|---|---|---|---|
| mobilenetv2_100 | 1872 | 131% | 104% |
| mobilenetv3_large_100 | 5394 | 114% | 102% |
| visformer_small | 4196 | 119% | 104% |
| repvgg_a2 | 2332 | 120% | 107% |
| tf_efficientnet_b0 | 3233 | 123% | 100% |
| nfnet_l0 | 47355 | 111% | 109% |

HF transformers re-run through model_attribution.py: raw 116-135% ->
corrected **107-109%** (within ~7-9%), same regime as timm, 0 unmatched.

ROOT CAUSE: (a) the correction was simply not applied — NOT a double-counting
bug. Extern ops partition-disjoint, occurrence joins exact (0 unmatched on all
14 models), overshoot scales with launch count as the launch-floor model
predicts.

TRUSTWORTHINESS CAVEATS:
1. The static results/all_oracle_timings_b200_v2.json covers only 27-67% of
   fusible occurrences per model — reconcile via model_attribution.py's fresh
   per-point bench, NOT the static JSON. (The --oracles sweep timings are
   honest per the bench audit, but the file is incomplete for full models.)
2. ViT-family (beit/deit/vit) UNDERshoots (corrected 0.69-0.82): in-graph
   layernorm/attention runs slower than standalone — a known real perf finding
   (see deit_incontext_layernorm_slowdown.md), not an accounting error.

bench_parallel --oracles audit (separate): timings are honest — times the
oracle (verified 38x-distinct case), enforces check-gate, holds GPU lock,
correct per-shape dispatch. So data is sound; only the correction needed applying.
