# Slice-output compaction family (harvest + extension)

Date: 2026-06-10. Hardware: B200, GPU lock, fresh `TORCHINDUCTOR_CACHE_DIR`,
CUDAGraph timing, CD tuning (standard oracle `--bench` methodology).
Branch: pytorch-work `pr-184905`.

## Commits

- `bbf37d454c2` (landed base): compact `aten.slice.Tensor` user-visible outputs
  of single-use pointwise intermediates (sum_sum_b16afac198fb 1.88x -> 0.97x).
- `3b746fce660` (prior agent WIP, KEPT): extend to `aten.select.int` and
  multi-view bases (one pointwise base feeding several compactable output
  views, grouped by base; combined view numel <= base/2).
- `3bf69043be0` (this work, follow-up): finishes the WIP and adds a
  `_MIN_SAVED_BYTES = 4MB` profitability gate. The WIP extension itself was
  structurally complete and correct; what it was missing was the gate — on
  small bases (llama RMSNorm last-token select, ~2MB saved) the clone's extra
  kernel launch is a measured ~1.5us REGRESSION (1.21x -> 1.49x), so the
  blind extension was a net loss on small-tensor inference repros.

## WIP disposition

KEPT and finished. The select.int + multi-view grouping logic from
`3b746fce660` is unchanged; the follow-up adds:

1. `_MIN_SAVED_BYTES` (4 * 1024 * 1024): skip when
   `(base_numel - sum(view_numels)) * element_size < 4MB`. Calibration points
   (B200): 2MB saved -> ~1.5us regression (mean_9c0fd9fb28b1); 12MB saved ->
   ~2.5us win (sum_sum_98c4811f6ddf); 75MB saved -> ~10us win
   (var_mean_c5067e6e3750).
2. Docstring update describing the gate.

## Harvest table

Source: static scan of `repros/canonical/*/repro.py` for returned
`slice.Tensor` / `select.int` / `narrow` views, cross-referenced with the
`needs_work_remeasure_2026-06-09.jsonl` worklist (ratio >= 1.15). All ratios
below re-measured fresh on 2026-06-10 with the final pass unless noted.

| repro id | worklist ratio | fresh ratio (pass ON) | fires? | why / why not |
|---|---|---|---|---|
| sum_sum_b16afac198fb | 1.85 | **0.95** | yes (1) | original target; slice of BN-bwd pointwise epilogue |
| sum_sum_98c4811f6ddf | closed (1.12 fresh OFF) | **0.82** (OFF: 1.12) | yes (1) | DenseNet BN-bwd: bare `slice` of `mul` epilogue returned; 12MB saved |
| sum_sum_3d76bb8bc6b6 | 1.59 | **1.29** (OFF: 1.63) | yes (1) | DenseNet BN-bwd: returns `slice(add(slice_of_input, epilogue))` — pass compacts it; residual gap is the split-K reduction itself (separate issue, oracle_cooperative_split_k) |
| var_mean_c5067e6e3750 | n/a (1.14 fresh OFF) | **0.88** (OFF: 1.14) | yes (2) | DeiT-distilled: `x[:,0]` + `x[:,1]` selects of one LayerNorm epilogue; multi-view base; 75MB saved |
| var_mean_2ec780efd8cf | n/a (1.34 fresh OFF) | **1.08** (OFF: 1.34) | yes (2) | same DeiT dual-select pattern, plus a third (non-view) output |
| mean_9c0fd9fb28b1 | n/a (1.21 fresh OFF) | 1.17 (no fire) | gated | llama: last-token `select` of RMSNorm epilogue; only 2MB saved -> below 4MB gate (ungated it was 1.49x, a regression) |
| var_mean_1bab7e80cec1 | n/a | 1.02 (no fire) | no | DeiT infer dual-select: decompositions already push the select INTO the epilogue (post-grad outputs are `add(mul(select(...)))`, no view at output) — already compact, at floor |
| var_mean_9da5c1fb26f0 | n/a | 0.97 (no fire) | no | same as 1bab7e80cec1 — select sunk by decomps, at floor |
| pointwise_4ebc1e147929 | n/a | n/a | no | siglip attention-pool `x[:,0]` of `[128,1,768]` add: select numel == base numel/1 (dim has size 1) -> fails numel<=base/2; nothing to save |
| pointwise_7517fa77d424 | n/a | n/a | no | duplicate of 4ebc1e147929 (same siglip pattern) |
| pointwise_3229f6e30711 | n/a | n/a | no | BERT dual-dropout: base `mul` has a second consumer `reshape` that is also an output -> full base needed anyway (correctly skipped) |
| sum_sum_e6f9af9e33b1 (flagged) | 0.98 | 0.98 (no fire) | no | output is `add(slice, slice)` — real pointwise op output, already compact; AT FLOOR (worklist entry was stale) |
| sum_sum_2f2d7ba72160 (flagged) | 0.94 | 0.96 (no fire) | no | same `add(slice, slice)` family; at floor |
| sum_sum_5c0f382244aa (flagged) | 0.97 | 1.03 (no fire) | no | same family; at floor |
| sum_sum_100fff569d2e | floor | 0.97 (no fire) | no | sibling; `add(slice,slice)` output, compact by construction |
| sum_sum_63e248035ceb | 1.68 | (other agent: 1.03) | no | slice of GRAPH INPUT consumed internally, not returned — out of scope; closed separately by bdc289b3644 |
| amax_sum_9940b361e5b4 | 2.60 | not re-benched | no | longformer slice_scatter/select_scatter chain; views feed `copy_`/scatter, not user outputs — different fix class |
| amax_sum_sum_{e2f518f0a274,6fd07d12d98a,86d05d6810f4} | 1.67/1.42/1.42 | not re-benched | no | `clone(slice(constant_pad_nd(input)))` at graph START (already cloned), not an output view |
| sum_sum_sum_e2388f04f7c2 | 1.67 | not re-benched | no | `select` of `view_as_real` consumed by `add` mid-graph, not returned |
| sum_sum_sum_f0377fc40fe2 | 1.39 | not re-benched | no | slices feed `sum`/`permute` mid-graph, not returned |
| amax_sum_d112f48ea917 | 1.33 | not re-benched | no | slices feed `cat` (roll pattern), not returned |
| amax_sum_4b162354068b | 1.26 | not re-benched | no | slice of graph INPUT (attention mask) feeding `where`, not returned |
| sum_sum_sum_d8e421a07bf7 | 1.30 | not re-benched | no | slice of input consumed by epilogue + `sum`; output is the sum, not a view |
| pointwise_27183a793fcd | not in worklist | n/a | no | `slice(slice(mm))` — base is a slice (of mm), not pointwise; chained-view base not handled (single static case, mm output) |
| pointwise_70c0751eb408 | not in worklist | n/a | no | pyhpc: slices of `full`/chained slices with scatter consumers — aliasing-heavy, correctly skipped |
| pointwise_531d72f1b34a | not in worklist | n/a | no | pyhpc: 60 outputs, selects of `select_scatter` bases — not pointwise bases, correctly skipped |
| sum_31bf563cdf96 / sum_f160d1f03c1f / sum_4de5b759d4d1 | not in worklist | n/a | no | slices of `reshape`/`view` of mm/reduction outputs — base is a view op, not pointwise |

## Regression results (final pass, B200)

| check | result |
|---|---|
| sum_sum_b16afac198fb | 0.95x AT_FLOOR (must be ~0.97x: PASS) |
| sum_sum_100fff569d2e | 0.97x AT_FLOOR (PASS) |
| sum_sum_e6f9af9e33b1 / 2f2d7ba72160 / 5c0f382244aa | 0.98x / 0.96x / 1.03x — unchanged, no fire |
| mean_9c0fd9fb28b1 | 1.17x with gate (vs 1.21x baseline OFF, 1.49x ungated) — regression fixed |
| `pytest -k "GPUTests and slice and not scatter"` test_torchinductor.py | 19 passed, 2 skipped |
| `pytest -k "GPUTests and select and not scatter"` | 1 passed |
| training w/ saved activations #1 (sigmoid output saved, slice returned, backward) | no fire; grads exact vs eager |
| training w/ saved activations #2 (tanh saved, select returned, backward) | no fire; grads exact vs eager |
| same shapes in inference mode | fires (confirms the training no-fire is the user_visible gating, not size) |
| output view of graph input | no fire; `out.data_ptr() == input.data_ptr()` aliasing preserved |
| numerics on all firing repros | max diff <= 3.1e-2 on f32 BN-backward (reduction-order tolerance), <= 2e-6 on LayerNorm selects |

## View-form inventory for the generic IR fix (graph.py:1990 view-output liveness)

This is the test-case inventory for the planned generic fix: compose the view's
index map into the producer's `inner_fn` at view-output realization instead of
realizing the base. The FX pass should then be deleted.

Forms that actually occur in the corpus, by frequency/importance:

1. **`slice.Tensor` (single dim, contiguous channel range) of a pointwise
   epilogue that depends on sibling reductions over the same inputs.**
   DenseNet BN-backward tail. The big wins (2x area). Cases:
   sum_sum_b16afac198fb, sum_sum_98c4811f6ddf, sum_sum_3d76bb8bc6b6 (here the
   epilogue also adds a slice *of a graph input* — the generic fix must compose
   the output-slice index map through an `add` whose other operand is itself a
   view of an input).
2. **Dual `select.int` (two indices on the same dim) of one LayerNorm
   epilogue.** DeiT-distilled cls+dist tokens: var_mean_c5067e6e3750,
   var_mean_2ec780efd8cf. Multi-view: ONE producer, TWO live elements on the
   selected dim. The generic fix needs element-level output liveness that
   unions several view index maps (cf. MEMORY: generic dataflow over pattern
   matching).
3. **Single `select.int` (last token) of an RMSNorm epilogue.** llama:
   mean_9c0fd9fb28b1. IMPORTANT: at this size (2MB saved) the FX-pass clone
   approach is a measured regression because it splits the fused kernel; the
   generic IR fix would NOT split the kernel (the reduction and the
   restricted epilogue stay fusable) and so should win here where the pass
   cannot. This is the case that most motivates subsumption.
4. **Decomp-sunk selects** (var_mean_1bab7e80cec1, var_mean_9da5c1fb26f0):
   inference decompositions sometimes already push the select into the
   epilogue. The generic fix gets this for free; any solution must not
   double-handle it.
5. **Chained views** (`slice(slice(mm))`, `slice(view(...))`,
   `slice(reshape(...))`): pointwise_27183a793fcd, sum_4de5b759d4d1,
   sum_f160d1f03c1f, sum_31bf563cdf96. Bases are mm/reshape, not pointwise —
   the FX pass skips them; a generic index-map composition handles chains
   naturally but the producer is an extern kernel (mm), so liveness must stop
   at realized extern outputs.
6. **NOT in scope / different fix class**: slice_scatter/select_scatter
   overlap chains (amax_sum_9940b361e5b4, pointwise_531d72f1b34a,
   pointwise_70c0751eb408) and input-slices consumed mid-graph
   (amax_sum_4b162354068b, sum_sum_63e248035ceb family).
7. **`narrow` / `narrow_copy`: zero occurrences** in the corpus as graph
   outputs (narrow decomposes to slice before post-grad).

## Status

- Extension landed as follow-up commit `3bf69043be0` on `pr-184905`
  (WIP `3b746fce660` kept underneath; an intervening commit `bdc289b3644`
  by another agent in choices.py made squashing inappropriate).
- New closed: var_mean_c5067e6e3750 (1.14x->0.88x), var_mean_2ec780efd8cf
  (1.34x->1.08x), sum_sum_98c4811f6ddf (1.12x->0.82x).
- Improved: sum_sum_3d76bb8bc6b6 1.63x->1.29x (residual gap is the dual
  channel-reduction split-K decision, tracked separately).
- Avoided regression: mean_9c0fd9fb28b1 via the 4MB gate; this case is the
  flagship argument for the generic graph.py fix.
