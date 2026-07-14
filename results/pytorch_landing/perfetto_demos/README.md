# Perfetto demo traces

Execution-ordered Chrome-trace JSONs produced by `scripts/rollup_to_perfetto.py`.
View: drag a file into https://ui.perfetto.dev (or `chrome://tracing` -> Load).

## Files

### `mobilenetv2_true_e2e.perfetto.json`
TRUE-to-e2e trace for `mobilenetv2_100` built from `model_attribution.py
--collect-order` per_node output. Two tracks on one execution clock:

- **fusible (real us)** — each fused Inductor partition as one slice, with its
  REAL measured time (from the already-locked attribution benches).
- **extern (real us: conv/GEMM/SDPA)** — non-fusible extern ops, real measured us.

Track lengths sum to the model's `sum_parts` (the honest e2e decomposition), so
you see where the model's wall time actually goes, kernel by kernel, in run order.

### `PROOF_priced_overlay.perfetto.json`
Same true-e2e timeline for `mobilenetv2_100` plus the opt-in **oracle ceiling**
track (`--oracle-timings`): for every fusible kernel, an oracle slice at the SAME
timestamp as its compile slice with `dur = oracle_us` (reference-kernel floor from
`results/all_oracle_timings_b200_v2.json`). Externs have no oracle, so their real
us is mirrored onto the oracle clock (they cancel between totals). The vertical
per-kernel gap between the "fusible (current compile)" and "oracle ceiling" rows
is the headroom, in execution order.

## How to regenerate

All three tool invocations (run from the repo root):

```bash
# 1. ACCOUNTING mode (no GPU): oracle-vs-compile per-kernel trace, re-runs the
#    partition pass; externs appear untimed.
python scripts/rollup_to_perfetto.py \
    --model mobilenetv2_100 \
    --timings results/all_oracle_timings_b200_v2.json \
    --out mobilenetv2_accounting.perfetto.json

# 2. TRUE-e2e mode: first produce the ordered per-node attribution (GPU, uses
#    the locked bench paths), then convert.
python scripts/model_attribution.py \
    --corpus-root <corpus> --suite timm --mode infer \
    --models mobilenetv2_100 --collect-order \
    -o mobilenetv2_attr.json
python scripts/rollup_to_perfetto.py \
    --model mobilenetv2_100 --source attribution \
    --attribution mobilenetv2_attr.json \
    --out mobilenetv2_true_e2e.perfetto.json

# 3. TRUE-e2e + oracle-ceiling overlay (pure CPU JSON join, no benching):
python scripts/rollup_to_perfetto.py \
    --model mobilenetv2_100 --source attribution \
    --attribution mobilenetv2_attr.json \
    --oracle-timings results/all_oracle_timings_b200_v2.json \
    --out mobilenetv2_overlay.perfetto.json
```

Note: the timings file gained a gap-fill merge on 2026-07-13 (1,514 priced
families plus `_metadata` / `__failures__` keys); `load_timings` skips keys
starting with `_`, so the tool tolerates the new format unchanged. With the
gap-filled file, some mobilenetv2 inference families (e.g.
`pointwise_9595e73ff43b`) remain honestly unpriced (`no_valid_point` gate
rejections in `__failures__`) — those kernels show on the untimed track.
