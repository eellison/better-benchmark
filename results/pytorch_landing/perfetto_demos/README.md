# Perfetto demo traces

Execution-ordered Chrome-trace JSONs produced by `scripts/rollup_to_perfetto.py`.
**View: drag a file into https://ui.perfetto.dev** (or `chrome://tracing` -> Load).
All traces are valid Chrome Trace Event Format (`traceEvents` + `displayTimeUnit`).

Track layout (attribution mode, `--oracle-timings` overlay):
- tid 1 `fusible (current compile, real us)` — measured compile time per fused kernel, execution order
- tid 2 `extern (real us: conv/GEMM/SDPA)` — measured extern time (not oracle targets)
- tid 3 `oracle ceiling` — per-fusible oracle_us at the SAME ts; externs mirrored so they cancel between totals

The vertical per-kernel gap between the "fusible (current compile)" and
"oracle ceiling" rows is the headroom, in execution order.

Timings file: `results/all_oracle_timings_b200_v2.json` (gap-filled 2026-07-13:
1,514 priced families; `load_timings` skips `_`-prefixed keys so the
`_metadata` / `__failures__` entries load cleanly). Device: NVIDIA B200,
SM clocks locked to 1852 MHz during GPU runs.

## The showcase traces (2026-07-14)

### 1. `AllenaiLongformerBase_oracle_overlay.perfetto.json` — THE MONEY SHOT (headroom)

The #1-headroom model: twelve giant `amax_sum` (scatter-attention) kernels each
2.6x over the oracle ceiling dominate the timeline; everything else is at floor.

- Attribution run: e2e=14,452us, sum_parts=15,788us, corrected ratio=1.02 (identity closes)
- Oracle join: 118 hit / 13 miss (misses are tiny FAM_UNPRICED pointwise families)
- Fusible-only: compile=11,907.5us vs oracle=5,748.2us → **ratio 2.07, headroom=6,159us (43% of e2e)**
- The gap is almost entirely ONE family: `amax_sum_528a3c274a41`, 12 occurrences,
  compile=833.3us vs oracle=320.2us each → 6,157us of the 6,159us headroom.
  This is the Longformer scatter/amax_sum kernel that drove the D2 investigation.
- Attribution source: `attr_longformer.json` (in this dir).

### 2. `alexnet_oracle_overlay.perfetto.json` — the "we're done here" picture (at-ceiling)

A fully-priced conv model where every fusible kernel's oracle bar visually
equals its compile bar — compile is AT the reference-kernel ceiling.

- Attribution run: e2e=773us, sum_parts=1,322us (conv-dominated; 8 externs = 1,213us)
- Oracle join: 7 hit / 0 miss — 100% priced, zero unpriced markers
- Fusible-only: compile=108.7us vs oracle=110.1us → **ratio 0.99, headroom=-1.4us**
  (compile marginally beats the oracle; statuses GOOD/AT_FLOOR)
- Attribution source: `attr_alexnet.json` (in this dir).

### 3. `mobilenetv2_100_oracle_overlay.perfetto.json` — honest all-miss (gate-rejected families)

A finding, not a demo: even after the gap-fill (1,514 priced families),
mobilenetv2_100 inference remains 0-hit — all 52 fusible slices are
`[no-oracle:FAM_UNPRICED]` 0-dur markers.

- Root cause verified: its inference pointwise BN-fold families
  (`pointwise_9595e73ff43b`, `pointwise_818c3aa59952`, `pointwise_e6ddc8e897ec`, ...)
  EXIST as `repros/canonical/` dirs but were gate-rejected in the v2 timings file
  (`no_valid_point` in `__failures__`). The 10 v2 families carrying
  mobilenetv2_100 points are its train-side reduction families — mobilenet-infer
  pointwise is part of the 213 gate-rejected families (oracle-rewrite residue).
- Attribution source: `attr_mobilenetv2_collect_order.json` (in this dir; reused
  locked bench — overlay regen is CPU-only).

### Summary table

| trace | fusible ratio (compile/oracle) | oracle hits | what it shows |
|---|---|---|---|
| AllenaiLongformerBase | **2.07** (headroom 6,159us) | 118/131 | the whole gap = 12 amax_sum bars |
| alexnet | **0.99** (at ceiling) | 7/7 | fully priced, compile at the floor |
| mobilenetv2_100 | n/a (0 hits) | 0/52 | honest all-miss: gate-rejected BN-fold families |

## Older traces (kept)

### `mobilenetv2_true_e2e.perfetto.json`
TRUE-to-e2e trace for `mobilenetv2_100` built from `model_attribution.py
--collect-order` per_node output — fusible + extern tracks on one execution
clock, no oracle overlay. Track lengths sum to the model's `sum_parts`.

### `PROOF_priced_overlay.perfetto.json`
Same true-e2e mobilenetv2 timeline plus the opt-in oracle-ceiling track — the
first proof-of-concept of the overlay format.

## How to regenerate

Run from the repo root. The GPU step (~4 min/model) needs the lock + pinned
clocks; the overlay step is a pure CPU JSON join (seconds).

```bash
# 1. Ordered per-node attribution (GPU, locked):
python scripts/with_gpu_lock.py --gpu 0 -- \
  python scripts/model_attribution.py --corpus-root repros \
    --suite hf --mode infer --models AllenaiLongformerBase --collect-order \
    -o attr_longformer.json

# 2. TRUE-e2e + oracle-ceiling overlay (CPU-only):
python scripts/rollup_to_perfetto.py --source attribution \
  --attribution attr_longformer.json --model Longformer \
  --oracle-timings results/all_oracle_timings_b200_v2.json \
  --out AllenaiLongformerBase_oracle_overlay.perfetto.json
```

Other models: same two commands with `--suite torchbench --models alexnet`
(`--model alexnet`) or `--suite timm --models mobilenetv2_100`
(`--model mobilenetv2`). For an accounting-mode trace (no GPU, re-runs the
partition pass, externs untimed):

```bash
python scripts/rollup_to_perfetto.py \
    --model mobilenetv2_100 \
    --timings results/all_oracle_timings_b200_v2.json \
    --out mobilenetv2_accounting.perfetto.json
```
