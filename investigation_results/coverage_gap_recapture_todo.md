# Coverage-gap recapture TODO (filed 2026-06-16)

Tracked follow-up for families present in the **old** `repros/` corpus that
have **no v2 successor by family name**. Filed per user decision at flip time:
*"lets just file these as TODOs."* These do NOT block the atomic flip.

## Why this is cheap to defer

- The old `repros/canonical` carries **0 hand-written oracles** (all 1727
  oracles live in `repros_v2/canonical`). Losing these families costs only
  **regenerable full-graph reference trees**, not irreplaceable oracle work.
- The v2 capture pipeline (wave-1) reproduces full-graph trees from scratch;
  a future recapture wave can backfill any of these without redoing oracles.

## The raw "428 lost models" was a naming artifact

A raw model-dir name diff reported 428 losses. Normalizing to **family names**
collapses that to **55**, of which:

- **17 REDUNDANT** — covered by the canonical `hf/` and `timm/` suites, which
  are supersets. (torchbench carried its own `hf_bert`/`hf_gpt2`/`hf_t5`/… and
  `efficientnet`/`nfnet`/`vision_transformer` copies; these models are captured
  cleanly in the dedicated suites.)
- **6 DOCUMENTED blockers** (wave1_capture_report.md):
  - `dlrm`, `moco` — distributed models (skipped, §5)
  - `soft_actor_critic` — `torch.geqrf` requires LAPACK (§4d)
  - `opacus_cifar10` — wave-2 deferred, capture-hook index-bounds failure (§4d/§7)
  - `llama` — gated repo, Meta approval pending (§1)
  - `swin_base_patch4_window7_224` — **infer captured**; train hit 91-min
    compile timeout (§3). Only the train tree is missing.
- **9 GENUINE architecturally-distinct gaps** — the actual TODO below.

## TODO: recapture these 9 families (priority order)

Higher value (modern LLM / multimodal):

- [ ] `nanogpt` (infer + train)
- [ ] `modded_nanogpt` (infer + train)
- [ ] `llava` (infer)
- [ ] `moondream` (infer)

Lower value (older CV, largely represented by timm conv families already):

- [ ] `regnet` (infer + train)
- [ ] `resnest` (infer + train)
- [ ] `vovnet` (infer + train)
- [ ] `doctr_det_predictor` (infer)
- [ ] `doctr_reco_predictor` (infer)

Also worth a cheap retry under the TODO umbrella:

- [ ] `swin_base_patch4_window7_224` **train** — only blocker was a 91-min
  compile timeout; an overnight run likely captures it. (Infer already in v2.)

## Validation note

This gap is a **coverage** decision, fully orthogonal to capture **fidelity**.
The model-accounting gate (0 unmatched partitions / 0 bench failures across 21
models, all suites) confirms every captured pattern maps exactly to canonical —
see `model_accounting_v2_validation.md` / `v2_model_rollup_validation.md`.
