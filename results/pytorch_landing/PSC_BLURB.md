# PSC blurb — better-benchmark / Inductor hill-climbing

**Built a fast inner loop for `torch.compile` performance work** — a corpus of
1,727 deduplicated kernel microbenchmarks (~5,000 shape points) partitioned
from 158 OSS models' post-grad FX graphs, each with an agent-written (Codex
5.5) Triton reference kernel as an optimization target, plus per-model
accounting that rolls kernel wins up to end-to-end impact.

- **A faster cycle with better signal:** ~14 min to a full corpus datapoint vs
  ~10 h for one `inductor-perf-nightly-h100` run (~40× faster, ~50× fewer
  GPU-hours) — and the signal is per-kernel with a concise standalone repro,
  not a single net number per model, so you see *which* kernel moved and by
  how much, and a 90% kernel win isn't drowned by run-to-run noise on the
  model's GEMMs.
- **Demonstrated the loop end-to-end:** drove a WIP Inductor branch to
  **+4.5% e2e geomean (+2.2% median) across 158 models**, incl. 45–67% on
  cross-entropy/softmax micros and 85–94% on BatchNorm `var_mean` kernels.
- **Per-commit attribution → landing queue:** benched the branch at ~37
  commits (≈ the GPU cost of 1–2 nightly runs); identified that one rsqrt
  canonicalization carries ~half the win, isolated the other landable units,
  and caught several net-negative/dead features before they shipped.
- **Ceiling analysis:** with all 1,727 kernel families swept (1,514 with usable
  floors; 213 gate-rejected), the median model sits ~0.6% from the
  reference-kernel ceiling; remaining headroom is concentrated and named
  (Longformer 37%).
- **Measurement rigor as product:** GPU-locked timing (CI-guarded), automatic
  clock pinning, measured noise floors (±0.82% model / ~0.2% kernel geomean),
  and a numerics gate + source audit that rejected ~30% of agent-written
  oracle points and caught 6 kernels gaming the tolerance.

Next: land the attributed winners upstream as individual PRs; extend to
internal models (internal-only corpus variant); expose more codegen decisions
to autotuning on top of this signal pipeline.
