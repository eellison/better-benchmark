# Longformer band-assembly gap — HANDOFF to scatter-pass PR owner

**Target:** repros/canonical/amax_sum_528a3c274a41 (AllenaiLongformerBase sliding-window attention).
#1 oracle-headroom result: ~37% of the model, 2.46x gap (compile 784us vs oracle 319us), fires 12x.
Scope-FAIR (bit-exact --check + a pure-torch gather reconstruction is bit-identical to the oracle —
NOT a scope-mismatch / not on the scope_mismatch_oracles.md blacklist).

**Root cause:** Inductor lowers the slice_scatter/select_scatter/constant_pad_nd/copy band-assembly
chain into ~7 kernels that MATERIALIZE the full [8,1024,12,513] band (~100.9 MB) to HBM; the softmax
reads it and the epilogue re-reads it (band touched 3x). Band assembly = 70% of time; softmax only 16%.

**Fix shape is proven:** expressing the band as a GATHER from the [288,512,512] bmm output lets
Inductor's EXISTING generic codegen hit 296.7us — already ≤ the 319us oracle. No new kernel template
needed; the missing piece is an FX-level scatter-assembly→gather rewrite so the band buffer is never
materialized and the gather fuses into the amax/sum reduction.

## Recommendation: HANDOFF (not build-standalone) + one immediate bug-fix

### ASK 1 — NOW, small/high-confidence: fix a real regression in `diagonal_skew_elimination`
`diagonal_skew_elimination` (commit d8e9914094a, default-on, targets these ~87 repros) is a *read-bypass*
pass — it reroutes unwritten reads to the base buffer but NEVER removes the 100.9MB band materialization.
On THIS point it fires (scatter_read_bypass=12) and makes it **+24% slower** (634us → 795us). It is missing
a "does this actually eliminate a materialization / min-bytes profitability" gate. Adding that gate stops
the regression on band-assembly points where the base buffer stays live, AND is a prerequisite so it
doesn't fight the gather pass below. Evidence: results/longformer_amax_sum/pass_toggle.json, FINDINGS.json.

### ASK 2 — the build (multi-week, fragile): band→gather rewrite as forward dual of structured_scatter_decode
- **Where:** post-grad FX pass in torch/_inductor/fx_passes/ (same registration/gating as existing scatter passes; no new infra).
- **Model it on `structured_scatter_decode.py`** — it decodes affine-iota index_put scatters into a
  consumer-agnostic overlap-add of masked loads. The Longformer band→gather pass is its FORWARD DUAL
  (positional slice_scatter/select_scatter tower instead of index_put). Copy that design.
- **NOT** `diagonal_skew_elimination` (read-bypass only, wrong lever), `slice_scatter_elision`
  (single-level slice-consumer only), or `scatter_reduce_fusion` (index_put + [0,2,3] reduction only) — none fit.
- **Generic index map** derivable from box/view geometry (node.meta['val'] shapes + scatter/pad/slice
  offsets) — NO hardcoded 256/257/513. Proof obligations: disjoint-cover/last-writer, affine source trace,
  mask preservation (see DESIGN_scatter_to_gather.md §2.3-2.5).
- **Order BEFORE diagonal_skew_elimination**, behind a new default-on flag, gated:
  static-shape + CUDA + all-consumers-rewritable + min-bytes.
- **Correctness traps:** bf16 cast boundary; the -3.39e38 additive key bias vs -inf invalid-key fill
  (must stay distinct); post-div query-mask zeroing; prims.rev edge-window reversal. Test matrix:
  first/interior/last windows x padded queries x padded keys.

**Full design:** results/longformer_amax_sum/DESIGN_scatter_to_gather.md
**Evidence:** FINDINGS.json, baseline_bench.json, config_sweep.json (nothing closes it: best CD+combo 638us = 2.0x),
pass_toggle.json (the regression), gather_rewrite.json (296us proof), output_code_CD.txt (the 7 materialization kernels).
