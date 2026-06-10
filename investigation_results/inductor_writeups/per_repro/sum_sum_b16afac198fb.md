# sum_sum_b16afac198fb Investigation

## Summary
- **Model**: torchbench_densenet121_train_001 (DenseNet BN-backward tail)
- **Verified 2026-06-10 (B200, GPU lock)**: oracle 120.4us, compile (CD) 226.9us, ratio ~1.87x
- **Classification**: OUTPUT-VIEW MATERIALIZATION (not CSE, not tiling)

## Root cause (kernel-body evidence, 2026-06-10)

Inductor emits ONE fused kernel `triton_red_fused_le_mul_sub_sum_unsqueeze_where_0`
(xnumel=256, r0_numel=200704) with TWO sequential `for r0_offset in tl.range(...)`
loops:

1. Loop 1 (reduction pass): loads `in_ptr0/in_ptr2/in_ptr3` at
   `r0_1 + 3136*x0 + 802816*r0_2` (evict_last), accumulates both sums
   (`_tmp12` = sum(where*centered), `_tmp15` = sum(where)).
2. Loop 2 (epilogue pass): RELOADS the same three buffers at the IDENTICAL index
   expression (`r0_1 + 3136*x0 + 802816*r0_2`, evict_first) to recompute
   `where_self` / `centered` and stores the FULL [64,256,56,56] epilogue to
   `out_ptr3 + (r0_1 + 3136*x0 + 802816*r0_2)`.

So num_load=12 is NOT a CSE failure across sibling reductions (the two sums share
loads inside loop 1) and NOT a layout/view index mismatch (indices are identical).
It is the *reduction -> dependent-pointwise* fusion: the epilogue consumes the
reduction results, so it must run after `tl.sum`, forcing a second full pass over
all 3 x 51.4MB inputs. On top of that, the graph only returns
`slice(mul_tensor_7, 1, 224, 256)` (1/8 of channels), but because the output is a
*view*, Inductor realizes the full base buffer:
`buf3 = empty((64,256,56,56))` ... `return reinterpret_tensor(buf3, (64,32,56,56), ..., 702464)`.
That is 205MB written + 308MB read = ~514MB traffic where the oracle moves ~257MB
(reduction reads 154MB once; epilogue touches only the 32 sliced channels: ~77MB
read + 26MB write).

Decision point: `torch/_inductor/graph.py:1995-2046` — for an output node whose
`meta["val"]._is_view()`, Inductor only calls `require_stride_order` (comment at
graph.py:2028 "to avoid converting possible view ops to a copy kernel"), i.e. it
deliberately keeps the slice as a view of the fully-realized base. Nothing in the
scheduler can then shrink the epilogue: the base IS the buffer being returned
(through reinterpret_tensor).

## Experiment that proves it

Appending `clone` to the slice output in the repro graph (so the output is compact)
flips codegen to TWO kernels:
- `triton_red_fused_le_mul_sub_sum_where_0`: 6 loads in ONE r0 loop, stores both sums.
- `triton_poi_fused_clone_slice_1`: xnumel=6422528, reads inputs at
  `702464 + x3 + 802816*x2` (slice region only), writes compact [64,32,56,56].

Measured: 118.6us (CD) vs oracle 120.4us — at floor. Sibling sum_sum_100fff569d2e is
at floor "accidentally" for the same reason: its returned epilogue is
`add(slice, slice)` — a real pointwise op, hence compact output, hence separate
slice-only kernel.

## Fix

New post-grad FX pass `torch/_inductor/fx_passes/slice_output_compaction.py`
(config flag `compact_slice_outputs`, default True, env
`TORCHINDUCTOR_COMPACT_SLICE_OUTPUTS`): when a graph output is
`aten.slice.Tensor` of a single-use pointwise intermediate, and the slice numel is
at most half the base numel, insert `aten.clone` after the slice and return the
clone instead. The clone gives the output a contiguous (compact) layout, so
Inductor (a) never realizes the full base, (b) computes the epilogue only for the
sliced region, and (c) keeps the dual reduction in its own single-pass kernel.
Output stride metadata (`original_output_strides`) is updated to the clone's
contiguous strides; this is legal because Inductor only guarantees stride ORDER
for view outputs anyway (graph.py:2030-2038).

## Timings (B200, GPU lock, fresh cache, CD tuning)

| variant | time | ratio vs oracle 120.4us |
|---|---|---|
| before | 226.9us | 1.88x |
| clone experiment (manual graph edit) | 118.6us | 0.99x |
| after FX pass (two runs) | 118.6us | 0.99x |

Codegen after: `triton_red_fused_le_mul_sub_sum_where_0` (6 loads, one r0 loop,
stores both sums) + `triton_poi_fused_slice_1` (slice-region-only epilogue,
compact [64,32,56,56] output). Numerics verified vs eager (max diff 1.1e-5 on
the [64,32,56,56] output; sums match within reduction-order tolerance).

## Regression checks (pass ON vs TORCHINDUCTOR_COMPACT_SLICE_OUTPUTS=0)

| repro | ON (CD) | OFF (CD) | verdict |
|---|---|---|---|
| sum_sum_100fff569d2e (sibling, at floor) | 112.3us | 112.5us | unchanged |
| sum_sum_e6f9af9e33b1 | 51.0us | 51.1us | unchanged |
| sum_sum_5c0f382244aa | 33.7us | 33.7us | unchanged |
| mean_var_mean_cf650837b7b1 (layer_norm) | 14.1us | 15.2us | unchanged/noise |

(The family repros end in `add(slice, slice)` rather than a bare output slice,
so the pass correctly does not fire on them — confirmed kernel counts identical.)

Safety smoke tests (all pass):
- fires on bare pointwise-slice output, numerics exact
- does NOT fire when slice has another consumer
- does NOT fire on training graphs' saved activations (user_visible gating)
- does NOT fire under dynamic shapes (symbolic numel -> skip), still correct
- does NOT fire on slice-of-input outputs; input aliasing preserved
  (`out.data_ptr() == input.data_ptr()`)
- test_torchinductor.py GPUTests -k "slice and not scatter": pass

## Status
- FIXED: 1.88x -> 0.99x (at floor / slightly under oracle).
- Commit: bbf37d454c2 "[inductor] Compact slice outputs of pointwise epilogues
  (sum_sum_b16afac198fb, 1.87x->1.00x)" on branch pr-184905.
- Files: torch/_inductor/fx_passes/slice_output_compaction.py (new),
  torch/_inductor/fx_passes/post_grad.py (registration after
  materialize_constant_full_outputs), torch/_inductor/config.py
  (compact_slice_outputs flag, default True,
  env TORCHINDUCTOR_COMPACT_SLICE_OUTPUTS).
