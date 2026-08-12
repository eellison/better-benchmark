> **SUPERSEDED by INNER_LOOP_POST.md** — this is the old 2026-06-23 draft; some numbers (e.g. ~0.4% ceiling distance, pre-gapfill gate counts) are stale. Do not fix here; read INNER_LOOP_POST.md.

# Hill-climbing Inductor with a kernel benchmark corpus — what we built and what it found

*(Draft post — Elias. Numbers as of 2026-06-23, 4×B200. All measured through the
locked bench path; A/A noise floor ±0.82% per-model, ~0.2% kernel-geomean.)*

## The one-paragraph version

We built a corpus of **1,727 deduplicated kernel patterns** extracted from 158
real models (torchbench / HuggingFace / timm, train+infer), each with a
reference ("oracle") implementation as an optimization target, and a benchmark
harness fast enough to use as a hill-climbing signal: **~14 min / ~0.9
GPU-hours for a full all-shapes corpus sweep**, vs **~47.5 GPU-hours and ~10 h
wall-clock** for one `inductor-perf-nightly-h100` dashboard datapoint —
roughly **50× cheaper per datapoint**, which is what makes per-commit
attribution affordable at all. Using it, a WIP Inductor branch reaches
**+2.2% median / +4.5% geomean per-model end-to-end** across the 158 models —
and, more importantly, we then used the same infrastructure to attribute that
win **commit-by-commit**, which is what turns a WIP branch into a prioritized
landing queue. The branch isn't landable as-is; the attribution tells us exactly
which pieces are.

## Time to signal — vs. the nightly dashboard

The core loop is: change Inductor → sweep the corpus → per-model rollup.

- Full corpus, all shapes (~5K points): **~14 min** on 4×B200 (after sharding
  each repro's shapes across GPUs) ≈ **0.9 GPU-hours per datapoint**.
- Genai micro set (softmax/CE/norms fwd+bwd): minutes, stable to **0.03–0.5%**.
- Per-model rollup from two sweep JSONs: seconds, no GPU
  (`scripts/perf_ab_rollup.py`, on main — reproduces the headline from raw
  sweeps).

For scale, one `inductor-perf-nightly-h100` dashboard datapoint costs
**~47.5 H100 GPU-hours** (21 single-GPU shards over 3 suites, eager+compiled
end-to-end + accuracy) with **~10 h wall-clock** to signal (pinned by one 9.4 h
straggler shard; ~2.3 h if perfectly balanced). Per datapoint that makes the
corpus sweep roughly **50× cheaper and 40× faster to signal** — which changes
what you can afford to ask. A nightly budget buys the dashboard *one* HEAD
datapoint; the same GPU spend buys ~50 corpus datapoints, i.e. an entire
per-commit walk. That's exactly how the attribution below was affordable:
benching the branch at **43 individual commits** cost on the order of *one*
nightly run.

The honest trade: the dashboard measures true end-to-end wall-clock (eager
baseline, compile time, accuracy) on every model; the corpus measures kernels
and *estimates* model e2e via the occurrence-weighted rollup. The estimate is
validated — it reconciles against measured full-graph e2e within the
CUDAGraph-launch correction, and the A/B headline reproduces from raw sweeps —
but the two are complements, not substitutes: the corpus for fast, attributable
hill-climbing signal; the dashboard for the ground-truth checkpoint.

## Headline results

**1) The WIP branch (A/B vs its true base):**
- **+2.18% median / +4.51% geomean** per-model e2e, 158 models
  (119 improved / 31 regressed / 8 flat). A/A floor is ±0.82%, so this is
  well above noise.
- Individual kernels move a lot more: the BatchNorm `var_mean` family improves
  **85–94%** per-kernel; conv/BN-heavy models like repvgg_a2 gain **+26%** e2e.
- Genai micros: SoftmaxForward **+45%**, CrossEntropy fwd/bwd **+64–67%**
  end-to-end vs base — no genai micro regresses.

**2) Distance to the reference-kernel ceiling (how much is left):**
- Median remaining headroom vs the oracle ceiling: **0.4%** — most models'
  fusible kernels are effectively *at* the reference ceiling after the branch.
- The exceptions are concentrated and named: Longformer **37%** (one real,
  fixable scatter/gather kernel), Swin ~9%, vgg16 ~7%, a BERT-train cluster
  ~4–5%. This is the future worklist, ranked.

## The part I think is most useful: per-commit attribution

A 5% geomean on a 139-commit WIP branch is not a landing plan. So we benched
the branch **at every substantive commit** (43 states) and computed each
commit's marginal delta in two units — per-kernel geomean and honest
per-model e2e (extern-diluted; kernel numbers overstate model impact ~2–3×).
Findings:

| What | kernel Δ | model-e2e Δ | verdict |
|---|---:|---:|---|
| `reciprocal(sqrt(x))→rsqrt(x)` canonicalization | +5.6pp | **+2.1pp** | ~half the branch. 45-line pattern; verified mechanism (software sqrt+div → one MUFU rsqrt; kernels go compute→memory-bound) |
| CE loop-invariant hoist | +1.2pp | +0.5pp | land |
| online-softmax fast combine | +0.8pp | +0.3pp | land |
| scalar-accumulator reduction configs | +0.6pp | +0.2pp | land (this is the SoftmaxForward +45%) |
| layout-transform store-sinking | 1.2× on its kernels | small | land (structural fusion win) |
| MOR finalize-sum kernel | 1.14× on 52 reduction repros | — | land |
| ~7 other features | — | — | **net-negative or dead** on the corpus — drop or gate |

The last row is the point: the same infrastructure that found the wins also
found that several plausible-looking features are **net-negative** (e.g. an
inlining pass that wins 1.36× on Longformer but loses 25–40% on XLNet/XGLM
reductions; a BN-affine-fold pass at −0.3pp corpus-wide) or **dead** (a
1,929-line fusion pass that fires on 0/1727 corpus patterns). Without
per-commit + per-feature attribution, those would have shipped inside the
bundle. Three of the branch's regressions (pytorch_unet −20%, Longformer −9%)
are root-caused with validated gate patches ready.

## Measurement rigor (what it took to trust these numbers)

Worth a paragraph because two "findings" this month were measurement artifacts:
- A ±10–44% genai swing turned out to be a bespoke timer skipping the per-GPU
  bench lock — on the locked path the same micros are stable to 0.03–0.5%. We
  deleted the bespoke timers and added a CI guard test so an unlocked timing
  path can't land again.
- A naive all-points kernel diff reported "871 regressions"; auditing showed
  ~95% were a default-config-vs-autotuned metric mismatch plus per-point noise
  — every genuine regression was already in the known list. The per-model
  rollup (occurrence-weighted, shape-matched) is the number we headline.
- GPU clocks now lock automatically per sweep (`--lock-clocks`, on by default,
  non-blocking), A/A floors are measured and cited with every claim.

## What's next

1. Land the attributed winners upstream as individual PRs (rsqrt first —
   verified standalone, cherry-picks cleanly onto the base).
2. Gate-and-land the two features with validated regression patches.
3. Fill the oracle-timing coverage gap (346/1727 families have reference
   kernels but no timings yet — skewed toward conv/BN patterns) so the
   ceiling numbers cover the full corpus.
4. (Built, optional) Perfetto trace export: a model's rollup rendered as an
   execution-ordered timeline, compile vs oracle-ceiling per kernel — useful
   for eyeballing where a specific model's headroom lives.

---
*Repro: `github.com/eellison/better-benchmark` — README "Measuring a compiler
change" has the 3-command workflow; `perf_ab_rollup.py` reproduces the headline
from the sweep JSONs.*
