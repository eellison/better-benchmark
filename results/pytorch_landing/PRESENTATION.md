# Presentation package — a faster inner loop for `torch.compile` performance work

A corpus of 1,727 deduplicated kernel microbenchmarks (~5,000 shape points)
partitioned from 158 OSS models' post-grad FX graphs, each with an
agent-written Triton reference kernel as an optimization target, plus
per-model accounting that rolls kernel wins up to end-to-end impact. Used to
drive a WIP Inductor branch to +4.5% e2e geomean across the 158 models, then
attribute that win commit-by-commit into a prioritized landing queue.

## The story in 3 figures

1. **[fig2 — 158 models, before/after](post_figures/fig2_per_model.html)** —
   sorted per-model e2e change with the ±0.82% noise band: the win is broad
   (median +2.2%), not one lucky model.
2. **[fig1 — per-commit attribution](post_figures/fig1_per_commit.html)** —
   each commit's marginal delta in kernel-geomean vs honest model-e2e: one
   rsqrt canonicalization carries ~half the win, and several features measure
   net-negative.
3. **[Longformer perfetto trace](perfetto_demos/AllenaiLongformerBase_oracle_overlay.perfetto.json)**
   — drag into https://ui.perfetto.dev: twelve giant `amax_sum` bars, each
   2.6x over the oracle-ceiling track, ARE the entire 6,159us gap (fusible
   ratio 2.07x); everything else is at floor.

(All five figures: [post_figures/index.html](post_figures/index.html).)

## The full narrative

**[INNER_LOOP_POST.md](INNER_LOOP_POST.md)** — the post. Final numbers are in;
4 `[TODO]`s remain for the author (citation, FlashInfer confirm, RFC name,
NVIDIA collab details).

## One-slide version

**[PSC_BLURB.md](PSC_BLURB.md)** — the five-bullet summary.

## Deep dives

- **[LANDING_PRIORITY.md](LANDING_PRIORITY.md)** — the per-commit landing
  queue (tiers 0-3, gates required, what to bundle vs headline).
- **[A2_verdict_table.md](A2_verdict_table.md)** — full ranked per-commit
  verdicts with rationale and caveats.
- **[perfetto_demos/README.md](perfetto_demos/README.md)** — all traces
  (headroom / at-ceiling / honest all-miss), ratios, view + regen instructions.

## Headline numbers (all verified)

| Metric | Value |
|---|---|
| Model e2e win (158 models, whole branch) | **+2.18% median / +4.51% geomean** |
| Corpus datapoint vs nightly dashboard | **14-36 min** vs ~10 h wall (~47.5 GPU-h) |
| Median distance to oracle ceiling | **~0.6%** (headroom concentrated: Longformer ~37%) |
| Largest single commit | rsqrt canonicalization = **~half** the model-level win |
| Oracle corpus | **4,976** shape points benched; **~30%** gate-rejected; **6** tolerance-gaming cheats caught by source audit |
| Family coverage | all **1,727** families swept; **1,514** usable floors; **213** gate-rejected (oracle-rewrite residue, incl. mobilenet-infer BN-fold) |
| Noise floors (measured) | ±0.82% per-model / ~0.2% kernel-geomean |
