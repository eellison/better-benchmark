# Two-Floor Accounting (v4)

Two side-by-side floors for every model, built by pure JSON recomposition of the
v4 oracle-floor sweep — **no re-benching, no GPU**. Both views run through the
same coverage-gated accounting tool (`scripts/model_graph_accounting.py`, commit
`e12e25b49`), which traces each model's saved full graphs in fake-mode,
partitions them with the capture pipeline's own partitioner, and sums the
per-shape floor over every partition occurrence.

## The two floors

| View | Per-shape floor | Reads | Cite it as |
|------|-----------------|-------|------------|
| **ORACLE-floor** | oracle time on shapes where **oracle BEAT compile**; no floor where it lost | `all_oracle_timings_v4.json` | **reference-kernel ceiling / optimization target** — "how good are our agent/reference kernels; what is the ceiling a perfect kernel would hit" |
| **MIN-floor** | **min(oracle, compile)** per shape, everywhere it is defined | `all_oracle_timings_v4_minfloor.json` | **achievable-today floor** — "what a model would run at if every kernel hit the best of oracle-or-compile, using compile where our oracle kernel loses" |

The two differ because the oracle-floor only credits a kernel where our tuned
("oracle") implementation is at least as fast as `torch.compile`. The min-floor
additionally credits the **compile** time wherever compile is the faster of the
two — which is the policy already underpinning Deliverable 2's headroom
(`max(0, branch − oracle)` never goes below the compile-achievable point).

### Which floor answers which question
- **"How close are our reference kernels to the best we know how to do?"** →
  ORACLE-floor. A model is *complete* under this view only if every one of its
  kernels has an oracle that beat compile. Most models are **incomplete** here —
  that is the honest statement that we do not yet have a winning reference kernel
  for some of their kernels.
- **"If we shipped today, picking the best of oracle-or-compile per kernel, what
  would each model's fusible time floor be?"** → MIN-floor. This is the number to
  quote as the *achievable* per-model floor, and the one that makes ~155
  models stop looking spuriously *incomplete* just because our oracle lost on a
  few of their kernels (compile still gives those kernels a floor).

## How the MIN-floor file is built (`build_minfloor.py`)
Starting from `all_oracle_timings_v4.json`:
1. **Real priced dirs (1148):** each per-shape floor is lowered to
   `min(oracle_us, compile_us)`. In **560** of these dirs at least one shape's
   floor drops to compile — 1202 `AT_FLOOR` shapes where oracle was within
   tolerance but compile was genuinely faster, plus 732 per-shape `BAD_ORACLE`
   points sitting inside otherwise-priced dirs. Top-level `oracle_us`/`compile_us`
   are re-derived as the median of the (lowered) per-shape values.
2. **`all_bad_oracle` failure dirs (384):** oracle lost on *every* point, but
   every point has a `compile_us`. These were previously dropped as "unpriced",
   which wrongly made ~155 models look incomplete. We synthesize a schema-
   identical priced entry whose per-shape floor IS the compile time (stored in
   the `oracle_us` field so the roll-up prices it), `status: AT_COMPILE_FLOOR`,
   tagged `synthesized_from: all_bad_oracle`. 544 per-shape points; compile-floor
   aggregate across the 384 dirs: median **38.56 us**, mean **131.52 us**, range
   4.96 – 3016.38 us.
3. **`no_valid_point` failure dirs (195):** left OUT — they have **no floor under
   either view** (no usable compile floor either). The separate numerics-opt-out
   agent handles these.

The min-floor is **monotone**: it only ever adds or lowers floors. Confirmed by
**zero** reverse flips (no model goes complete → incomplete).

## Projection results (166 models)

| | ORACLE-floor | MIN-floor |
|---|---|---|
| COMPLETE | 11 | **38** |
| INCOMPLETE | 155 | 128 |

**27 models flip INCOMPLETE → COMPLETE** under the min-floor — folding the 384
`all_bad_oracle` dirs gives them a floor for kernels that had none before. The
128 still-incomplete models are correctly incomplete: their remaining unpriced
kernels are 378 `no_valid_point` dirs (no compile floor either — left out by
design) + 87 UNMATCHED patterns (no canonical repro at all — a corpus-coverage
gap, not a pricing gap).

### Top-15 by ORACLE-floor projected fusible time
("reference-kernel ceiling" — only 11 models are complete under this view, so
this is the full complete set; the heavyweights are absent precisely because our
oracle does not yet win on all their kernels)

| model | oracle-floor us | occ |
|-------|----------------:|----:|
| timm/infer/convnextv2_nano.fcmae_ft_in22k_in1k | 7703.5 | 43 |
| genai/static/SoftmaxForward | 1919.1 | 1 |
| torchbench/infer/demucs | 472.2 | 25 |
| genai/static/LayerNormForward | 351.9 | 1 |
| genai/static/RMSNormForward | 343.9 | 1 |
| torchbench/train/dcgan | 156.0 | 24 |
| torchbench/train/lennard_jones | 122.7 | 22 |
| torchbench/infer/vgg16 | 117.3 | 15 |
| torchbench/infer/nvidia_deeprecommender | 88.0 | 6 |
| torchbench/train/tts_angular | 12.0 | 2 |
| torchbench/infer/tts_angular | 5.6 | 1 |

### Top-15 by MIN-floor projected fusible time
("achievable-today floor" — the real heavy training graphs surface once their
all_bad_oracle kernels get a compile floor)

| model | min-floor us | occ |
|-------|-------------:|----:|
| hf/train/XLNetLMHeadModel | 26662.1 | 775 |
| hf/train/GPTNeoForCausalLM | 13653.3 | 750 |
| hf/train/GPTNeoForSequenceClassification | 12848.8 | 750 |
| hf/train/MegatronBertForCausalLM | 11833.0 | 707 |
| timm/infer/convnextv2_nano.fcmae_ft_in22k_in1k | 7586.8 | 43 |
| hf/train/GoogleFnet | 7429.9 | 80 |
| hf/infer/XLNetLMHeadModel | 5494.9 | 171 |
| torchbench/train/BERT_pytorch | 4577.6 | 421 |
| genai/static/SoftmaxBackward | 4421.2 | 2 |
| timm/infer/dm_nfnet_f0 | 3621.3 | 135 |
| timm/infer/nfnet_l0 | 2873.5 | 134 |
| torchbench/train/vgg16 | 2822.3 | 80 |
| hf/train/XGLMForCausalLM | 2725.6 | 46 |
| hf/infer/GoogleFnet | 2152.0 | 40 |
| genai/static/CrossEntropyBackward | 2151.9 | 2 |

The convnextv2 row shows the floors are close where both are defined
(7703.5 → 7586.8, a 116.8 us / 1.5% drop): the min-floor is rarely *much* lower
per shared kernel; its impact is **coverage** (turning unpriced kernels into
priced ones), not a large per-kernel discount. That is why the distinction
matters far more for the absolute projection than for the headroom % below.

## Deliverable 2 (headroom %) under the two floors

Re-ran `results/perf_ab/upside_v2_v3.py` against the min-floor file. The
remaining-headroom ("what to work on next") ranking is **essentially unchanged**:

- Top-15 overlap: **14/15** (only swap: `hf/train/TrOCRForCausalLM` ↔
  `torchbench/infer/lennard_jones`, both negligible absolute us).
- #1–#4 identical and in order: AllenaiLongformerBase 37.4%, tts_angular ~13%,
  swin ~9.5%, vgg16 ~7%.
- Median headroom-% delta **0.12pp** (mean 0.21pp); only **3** models move >1pp,
  only **1** >2pp; median rank shift 4 positions.

This is expected: headroom only credits oracle-**wins** (`max(0, branch −
floor)`), and the `all_bad_oracle` dirs contribute ~0 headroom either way (their
floor ≈ compile ≈ branch). The min-floor mainly **raises coverage**
(e.g. swin `orac_cov` 23/39 → 35/39, MobileBert 57/70 → 69/70), which is why a
couple of high-coverage models nudge up the list, but the percentages barely
move. **The two-floor distinction matters for the absolute PROJECTION, not for
the headroom ranking.**

## Files
- `all_oracle_timings_v4.json` — ORACLE-floor timings (unchanged).
- `all_oracle_timings_v4_minfloor.json` — MIN-floor timings (this work).
- `build_minfloor.py` — builder; `MINFLOOR_changelog.md` — fold-in changelog.
- `model_projections/projections_v4_coveragegated.json` — ORACLE-floor projection.
- `model_projections/projections_v4_minfloor.json` — MIN-floor projection.
- `model_projections/compare_two_floors.py` → `two_floor_comparison.json` — the
  flip count + top-15 tables above.
