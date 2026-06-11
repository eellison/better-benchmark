# Branch Commit Audit — June 11, 2026

**Branch:** `pr-184905` in /tmp/pytorch-work
**Range:** `049d1229d84..HEAD` (17 core commits + 1 WIP from another agent)
**Rubric:** UPSTREAM-READY / NEEDS-WORK / EXPERIMENTAL / LOCAL-ONLY (per `pytorch_upstream_audit.md`)
**Status:** IN PROGRESS — skeleton committed first per audit protocol; verdicts filled incrementally.

---

## ALARMING CORRECTNESS FINDINGS

1. **a26fc2c8bf4 (online-softmax fast combine, default ON) silently changes NaN
   semantics of user-visible `amax` outputs.** `tl.max` does not propagate NaN
   the way `triton_helpers.max2` (and eager `torch.amax`) do: with a NaN in a
   row, eager returns NaN for the max while the fast path can return the
   largest non-NaN value. The commit message acknowledges "only the raw max
   output may differ in the presence of NaNs" — but the amax IS a graph output
   in the amax_sum_* family (online-softmax fusion returns (max, sum) pairs);
   any model that consumes the saved row-max directly (e.g. stable-softmax
   backward saving `max` for recompute, or user code reading amax) gets a
   silently wrong non-NaN value under default config. Downstream
   softmax/CE/logsumexp stay NaN-poisoned via the sum, which bounds the blast
   radius, but this is a silent eager-divergence behind a default-ON flag.
   Recommendation: default OFF upstream, or restrict the tl.max substitution to
   reductions whose max result is not a kernel output.

2. **bbf37d454c2 + 3bf69043be0 (slice-output compaction, default ON) rewrite
   `original_output_strides` in place, so compiled user-visible outputs change
   stride/contiguity (and lose mutual aliasing) vs eager.** A channel slice
   that eager returns as a non-contiguous view comes back contiguous from the
   compiled fn; `graph.get_user_visible_output_strides` then enforces the
   clone's strides instead of eager's. Stride ORDER is preserved (the pass's
   stated contract) but exact-stride consumers (`.data_ptr()` arithmetic,
   custom ops asserting strides, `torch.compile` stride-matching tests) and
   in-place mutation of one output expecting to be visible through an
   overlapping sibling slice (multi-view bases, added in 3bf69043be0) diverge
   from eager. Inductor's public contract for user-visible outputs has
   historically been exact strides where possible — this pass deliberately
   weakens it, default ON, with no opt-out besides the whole-pass flag.

3. **a85d79a900a (segment-aligned split) has no upper bound tying the split to
   xnumel — possible large scratch allocations.** The override bypasses the
   `numel_hint >= 2 * num_sm` early no-split exit (deliberately: the anchor has
   xnumel=1000), so a segmented reduction with BIG xnumel (e.g. 1M rows,
   rnumel=K*L with K up to 4096) now allocates an xnumel*K partials buffer
   (potentially hundreds of MB to GB) where baseline allocated nothing. No
   `partials_bytes` or total-rows cap exists. Perf-heuristic-only, but the
   memory regression mode is silent and could OOM previously-working models.

(Items 1-2 are behavior divergences behind default-ON flags; item 3 is a
resource-safety gap. No silent wrong-arithmetic bug was found in any of the 17
diffs.)

---

## Verdict Table

| # | SHA | Title (short) | Verdict | Correctness risk | Notes |
|---|-----|---------------|---------|------------------|-------|
| 1 | 6703f38fa2d | rsqrt canonicalization | UPSTREAM-READY | low | Pure gating of pre-existing patterns behind new flag; identity is safe |
| 2 | 52d4cadfac0 | Loop-invariant hoisting (default off) | EXPERIMENTAL | medium (flag-on only) | CSE cache-hit can mis-mark in-loop var as hoisted (compile error, not silent); neutral perf, default OFF; also smuggles in rotate_half_gather config flag |
| 3 | 1406552b9d3 | Rotate-half gather (RoPE) | UPSTREAM-READY | low | Careful static-shape/dtype/sign guards; dynamic shapes safely rejected; dead `_INT64_MAX`; no unit tests |
| 4 | edbd4b67279 | Fix stale other_node in scatter_add_into | UPSTREAM-READY | low | Sound replacement-map fix with cycle guard; gating a bug fix behind a flag is odd |
| 5 | bbf37d454c2 | Compact slice outputs of pointwise epilogues | NEEDS-WORK | medium | Deliberately overrides original_output_strides: compiled output strides/contiguity diverge from eager for user-visible outputs |
| 6 | 5489b8c2bb9 | scatter_add -> gather-predicate rewrite | NEEDS-WORK | medium | Identity verified correct (incl. atomic collisions); but 4-D/[0,2,3]/scatter_dim==1 shape-specialized; hand-built meta has wrong dtype for bool-sum node |
| 7 | 3b746fce660 | WIP: slice-output compaction extension | LOCAL-ONLY (squash) | low (superseded) | Fully superseded by 3bf69043be0; no half-state left in tree; squash before upstream |
| 8 | bdc289b3644 | Split partially-saturated multi-output reductions | NEEDS-WORK | low | Heuristic-only; B200-tuned magic window (num_sm//2..num_sm, cc>=10 gate); flag name overloads undersaturated flag semantics |
| 9 | 3bf69043be0 | Extend slice-output compaction (select.int etc.) | NEEDS-WORK | medium (inherits #5) | Logic sound (multi-view grouping correct); inherits stride-contract concern of bbf37d454c2; _MIN_SAVED_BYTES B200-derived constant |
| 10 | f6ae31e6984 | WIP: r-contiguity for evict_first | LOCAL-ONLY (squash) | low (superseded) | Flag renamed/replaced by e45f846f8d1; its `return ""` inside old `eviction_policy='<EP>'` template emitted `eviction_policy=''` (Triton tolerates, but sloppy) — cleaned up in follow-up; squash |
| 11 | e45f846f8d1 | Drop evict_first misaligned strides | NEEDS-WORK | low-medium | Coherent; fixes f6ae31e6984's DelayReplaceLine string bug; torch.cuda.get_device_capability() queries current device not the kernel's device |
| 12 | 60431eca431 | Gate evict_first drop on in-loop side stores | NEEDS-WORK | low | has_store_with_rindex resolved at DelayReplaceLine time — ordering OK; flag semantics now 3 conditions deep, name stale |
| 13 | e9a67c98a8c | Restrict evict_first drop to >L2 iteration space | NEEDS-WORK | low | numel*itemsize uses ONE load's dtype as proxy for working set; chain of 3 commits should squash into one well-named flag |
| 14 | a26fc2c8bf4 | Fast combine for online-softmax loops | NEEDS-WORK | MEDIUM | Default-ON flag changes NaN semantics of user-visible amax outputs (tl.max vs NaN-propagating max2); textual surgery on emitted load lines is fragile |
| 15 | 60dd3839913 | dedupe_graph_outputs earliest-node canonical | UPSTREAM-READY | low | Correct crash fix; relies on documented topo-sort precondition; `canonical_idx` now dead |
| 16 | 56959375c33 | Overlapping-window index_put -> overlap-add | UPSTREAM-READY | low | Index math verified (greedy slab decomposition + nesting-stride guards); zero/negative strides and dynamic shapes safely rejected; needs unit tests |
| 17 | a85d79a900a | Segment-aligned split introduction | NEEDS-WORK | low (perf-only knob) | L-detection edges handled (zero stride -> no hint; multi-load divisibility; dynamic -> None); B200 magic constants; `except TypeError` can swallow real bugs |
| — | 6f7fd7d74ec | WIP: ungate scalar-acc configs from CD | IN-PROGRESS | another agent's active work | not reviewed deeply |

---

## Per-Commit Detail

### 1. 6703f38fa2d — rsqrt canonicalization — UPSTREAM-READY
The two graph patterns (`reciprocal(sqrt(x))`, `div(1, sqrt(x))`) pre-existed
in post_grad.py; this commit only adds the `extra_check` gate +
`config.rsqrt_canonicalization` flag. Identity is exact for all dtypes
(1/sqrt == rsqrt bit-for-bit in eager decomp terms; eager itself uses rsqrt in
BN). `div_one_sqrt_to_rsqrt` correctly rejects non-scalar/non-1 numerators.
No edge-case exposure: dynamic shapes/CPU/0-size irrelevant to an algebraic
node swap. Note: upstream PyTorch may prefer this in joint-graph or decomp
rather than a new config knob, but as-is it's clean.

### 2. 52d4cadfac0 — loop-invariant compute/gather hoisting — EXPERIMENTAL
Default OFF and measured neutral, so by the branch's own evidence it should
not go upstream (no benefit demonstrated). Two latent issues if enabled:
(a) `CSEProxy._default` calls `mark_hoisted_compute(csevar)` after
`cse.generate(compute_buffer, ...)`; on a CSE **cache hit** `generate`
returns the previously-emitted variable — which may have been emitted inside
the loop into `self.compute` — yet it is added to `outside_loop_vars`,
letting a later op hoist a reference to an in-loop variable above the loop
(use-before-def; Triton compile error, not silent corruption). The
`not isinstance(v, CSEVariable)` guard does not cover this: a cache hit on an
identical expression string still has `v` as a plain string.
(b) `get_load_buffer` hoists indirect loads to `self.body` without masking
the gather index against rnumel-dependent masks — guarded by
`_load_mask is None` and `not has_rindex`, which appears sufficient.
QUALITY: the commit also (accidentally?) introduces the
`rotate_half_gather` config flag belonging to commit 3 ("landed in
52d4cadfac0 due to a concurrent commit" per 1406552b9d3's message) —
commits are not self-contained; squash or re-slice before upstream.

### 3. 1406552b9d3 — rotate-half gather — UPSTREAM-READY
Matcher requires: static dim sizes, step==1 slices, exact [0,K)+[K,D)
coverage, same source tensor, same dim, src shape+dtype == out shape+dtype
(rejects slice_scatter broadcast/cast variants), zeros base for the
slice_scatter form. Dynamic shapes return None at `_static_dim_size`/
`_static_int` — safe rejection. `_unsafe_index` is justified: index is in
[0,D) by construction. Sign-select built as (2*cond-1) multiply specifically
to keep the gather single-use (documented interaction with
realize_indirect_indexing_on_reuse). cat-form: negative cat_dim normalized
via out_val.ndim — correct. Minor: dead `_INT64_MAX` constant; no test file;
`mk()` re-executing targets under fake_mode is fine but `aten.reshape` on a
non-contiguous fake could in principle produce a copy node — harmless here
since sign vector is freshly computed. Edge: D-H == H not required (unequal
halves supported and handled by the asymmetric where) — verified consistent.

### 4. edbd4b67279 — chained scatter_add_into fix — UPSTREAM-READY
Real bug, well diagnosed (stale `other_node` resurrecting dead adds, DCE
silently undoing the first rewrite). Fix threads a replacement dict through
both call sites; `while other_node in replacements` walk has a `seen` cycle
guard. Quality nit: a *bug fix* gated behind `scatter_add_into_fusion_chained`
(default True) adds a flag nobody should ever turn off — drop the flag when
squashing into the parent pass for upstream.

### 5+9. bbf37d454c2 / 3bf69043be0 — slice-output compaction — NEEDS-WORK
See ALARMING #2. Mechanics are otherwise careful: requires
`user_visible_output_idxs` metadata (no-op otherwise — i.e. the pass
correctly skips saved-for-backward activations), base must be pointwise with
ALL consumers among the compacted output views, static numels only, combined
views <= base/2, >=4MB saved. Multi-view grouping in 3bf69043be0 is logically
correct (per-base consumer-set closure). Remaining concerns:
- exact-stride/aliasing divergence vs eager (ALARMING #2);
- `_MIN_SAVED_BYTES = 4MB` and the 1/2 ratio are B200-derived constants with
  no device scaling;
- duplicate-position note says "duplicate output positions share one clone"
  but each view_node maps to its own clone; only literally-identical nodes
  share — fine, comment slightly misleading.
WIP 3b746fce660 is fully contained in the final version — squash all three.

### 6. 5489b8c2bb9 — scatter_add gather-predicate rewrite — NEEDS-WORK
The algebraic identity is correct including atomic collisions: gathering
`mask[dest(i)]` per source element and summing `src*!mask[dest]` is exactly
sum-over-destinations of the scattered accumulation, because addition
commutes and each source contribution is individually gated by its (unique)
destination's mask. Guards added in this commit (index shape == src shape,
mask shape == view shape, scatter_dim==1, 0-dim fill, B*C_full row factor,
plain sum only, CUDA only) close the holes the earlier experimental pass had.
Issues:
- Heavily shape-specialized: 4-D view, `sorted(dims) % 4 == [0,2,3]` literal,
  scatter_dim==1. Fine for the corpus, far from a general upstream pass.
- Hand-built `meta["val"]` uses `dtype=src_meta["dtype"]` for the
  `fill_sum`/partials of `sum(mask, [2,3])` — summing a bool mask actually
  yields int64; meta dtype mismatch is mostly benign (later consumers use
  shapes) but can confuse downstream passes reading val.dtype.
- `% 4` in dim normalization assumes ndim==4 — consistent with the
  view-shape guard but brittle if detection ever loosens.
- `_get_tensor_meta` change adds "device" — fine.
Pass-ordering: runs after structured_scatter_decode and before
scatter_add_into_fusion; mutual exclusion with the experimental umbrella pass
handled by an explicit no-op guard. Coherent.

### 7. 3b746fce660 — WIP slice compaction — LOCAL-ONLY (squash)
Diffed against 3bf69043be0: the final commit strictly extends it (adds
`_MIN_SAVED_BYTES` gate + base_val reuse). No leftover half-state in the
tree. Should never appear in an upstream series.

### 8. bdc289b3644 — partially-saturated split — NEEDS-WORK
Pure heuristic, no correctness exposure (split reductions are
numerics-equivalent modulo fp reassociation, an accepted Inductor behavior).
Window `num_sm//2 <= xhint < num_sm` + `r*x <= min_elements_per_device` is
B200-measured; gated cc>=10 so older GPUs unchanged. Flag named
`split_reductions_for_partially_saturated_gpu` but the code path still also
requires `split_reductions_for_undersaturated_gpu` — the double gate is
non-obvious; document or fold into one knob. CPU unaffected (device-prop
path). Dynamic shapes: numel_hint/reduction_numel_hint are hints — fine.

### 10-13. evict_first chain — f6ae31e6984 (WIP) / e45f846f8d1 / 60431eca431 / e9a67c98a8c — NEEDS-WORK (squash to one)
Composed final condition (drop evict_first only when ALL hold):
  reduction-loop load, no r-contiguous dim, constant misaligned stride,
  kernel has in-loop r-indexed store, iteration space > L2.
The composition is coherent — each follow-up strictly narrows the drop — and
the DelayReplaceLine deferral correctly reads `has_store_with_rindex` after
all stores are emitted (set in store(), resolved at final string render).
Issues:
- f6ae31e6984 alone emitted `eviction_policy=''` via `return ""` inside the
  `'<EP>'` template (fixed by e45f846f8d1 moving the full
  ", eviction_policy=..." text into decide_later). WIP must be squashed.
- `torch.cuda.get_device_capability()` (no arg) uses the CURRENT device, not
  the kernel's device — wrong policy choice possible in multi-GPU
  heterogeneous setups (cosmetic perf, not correctness). e9a67c98a8c
  correctly passes the graph device to get_device_properties but the
  capability check above it still doesn't.
- L2 working-set proxy `prod(numels) * dtype.itemsize` uses the itemsize of
  whichever load is being decided — mixed-dtype kernels get inconsistent
  per-load policies; acceptable for a heuristic, deserves a comment.
- Flag name `evict_first_requires_aligned_strides` no longer describes the
  3-condition semantics. Rename on squash.
- 0-size/CPU/dynamic: guarded (cuda check, symbolic strides keep
  evict_first, hint fallback=0 keeps evict_first for unbacked).

### 14. a26fc2c8bf4 — online-softmax fast combine — NEEDS-WORK
See ALARMING #1 (NaN semantics of raw max output, default ON).
Verified `triton_helpers.maximum` adds `mask |= a != a` (NaN-propagating)
while `tl.max` is not — divergence is real, and the running-max accumulator
update still uses triton_helpers.maximum, so a NaN that survives the block
max DOES propagate; the loss is only within a block (where tl.max picks a
non-NaN over NaN). So: max output is wrong only when a row contains NaN.
Second concern: `_rewrite_masked_load_other_neg_inf` does TEXTUAL surgery on
already-emitted load lines (string match on `tmpX = tl.load(` + count of
`, other=0.0`). It bails correctly on native matmul (CSE key inspection) and
on multiple matches, and requires mask_vars == reduction masks; but any
upstream change to load-line formatting silently disables (fail-safe) or —
worse — a future second consumer of the same load var that relies on
masked-out lanes being 0.0 would read -inf. Today reductions re-mask with
their own default and stores are masked (per the docstring), so no concrete
breakage found; still the most fragile code in the series.
Also: rewrite mutates `self.loads._lines` of the CURRENT subkernel only —
correct scope. Tests: relies on test_online_softmax.py parity ("failures
identical before/after" — note there ARE pre-existing failures on this
branch).

### 15. 60dd3839913 — dedupe earliest-node canonical — UPSTREAM-READY
Correct fix for a real crash (later node replacing earlier node with
non-output users). Precondition (graph topo-sorted by the
stable_topological_sort that runs immediately before in post_grad) is
documented in-code. `canonical_idx` variable is now unused — remove on
squash with its parent commit (the dedupe pass itself, from the previous
audit's #46).

### 16. 56959375c33 — structured scatter decode (overlap-add) — UPSTREAM-READY
The strongest new pass of the series. Index math audited:
- `_trace_affine_index`: bounded walk through view/reshape/clone to
  as_strided(iota); start=0/step=1 enforced (BUG-ADJACENT: for
  `aten.arange.default` the start/step check only handles the
  `arange.start_step` overload; `arange.default` with a single `end` arg is
  fine, but `aten.arange.start` (start,end) overload is NOT in the accepted
  set — conservative, OK).
- `_decode_overlap_structure`: rejects size<=0, stride<=0 (so ZERO strides —
  fully-overlapping dims — are rejected rather than mis-decoded; a zero
  stride is non-injective but unsupported, correctly filtered by `s <= 0`),
  offset<0, out-of-bounds max address. Overlap-dim detection via
  stride[d] <= inner_extent in stride-descending order is the right
  injectivity criterion for nonneg strides. Remaining dims must strictly
  nest (stride divisibility, innermost==1) and the per-window greedy slab
  decomposition checks `rem == 0` and per-dim bounds — any failure aborts
  the rewrite. <=8 window unroll cap. Dynamic ranges: all sizes/strides must
  be literal ints; symbolic -> None. CUDA-only, functional index_put only,
  bool dtype excluded (accumulate=OR), dtype match required.
- fp reassociation: pad-sum reorders additions vs atomic order — but atomic
  scatter order is already nondeterministic, so this is an improvement
  (deterministic) not a regression. Commit honestly reports 9.2e-5 fp32
  delta on the reduction consumer.
Missing: unit tests; interaction note: runs BEFORE scatter_add_gather_reduce,
so an affine-iota scatter that both passes could claim is taken by the
(consumer-agnostic, strictly better) decode — correct priority.

### 17. a85d79a900a — segment-aligned split — NEEDS-WORK
L-detection (`inner_segment_length` in ir.py):
- multi-load different L: hint forwarded only if ALL inner loads produced a
  segment and min divides all — sound (chunk aligned for every load).
  Loads where detection fails (stride-1 missing, dynamic size) contribute
  no hint AND `len(hints) == num_inner` fails, disabling the override —
  conservative and correct.
- zero strides: `inner_segment_length` iterates reversed(strides); a 0
  stride != segment(1) breaks immediately -> segment=1 -> None. Squeezed
  size-1 dims (non-Symbol vars) skipped. Good.
- dynamic ranges: `_is_static` check returns None -> no hint. The hint call
  is additionally wrapped in try/except TypeError for out-of-tree
  InductorChoices subclasses — pragmatic but `except TypeError` around a
  user-overridable function can mask genuine TypeErrors INSIDE the
  override (it would silently fall back to the un-hinted split). Narrow it
  (inspect signature instead).
- choices.py `segment_aligned_split`: only overrides no-split or misaligned
  splits; keeps aligned ones; rnumel>8192, 64<=L<=16384, phase-1 rows >=
  2*num_sm, divisor fallback <=4096. `sympy.divisors(num_segments)` on a
  plain int is fine. MISSING: no cap linking split to xnumel — see
  ALARMING #3 (xnumel * K partials buffer can be huge).
- Interaction with the other two split layers in choices.py: order is
  (1) cooperative-would-trigger early return 1 — NOT segment-wrapped, so
  cooperative still wins; (2) no-split threshold path returns
  segment_aligned_split(1) — override CAN resurrect a split the
  undersaturation logic declined, which is the commit's point but means
  THREE interacting heuristics now decide split; a single decision table
  would serve upstream better.

### — 6f7fd7d74ec — WIP scalar-acc ungate (other agent) — IN-PROGRESS
Not reviewed deeply per scope. Touches config.py + triton_heuristics.py
(CD gating of scalar-accumulator configs + rnumel ceiling). Flagged only so
the next auditor knows it postdates this audit's range.

## Spot-Checks (3 headline claims)

(done last, after table complete)

| Claim | Repro | Claimed | Measured | Verdict |
|-------|-------|---------|----------|---------|
| TBD | | | | |

## Top-5 Prioritized Issues

1. **[a26fc2c8bf4] Default-ON NaN-semantics change in online-softmax max
   output.** Make `online_softmax_fast_combine` default OFF, or keep the
   load-fill/-inf rewrite (safe) and only use tl.max when the max accumulator
   is not a kernel output. (ALARMING #1)
2. **[bbf37d454c2/3bf69043be0] Slice-output compaction weakens the
   user-visible output stride/aliasing contract, default ON.** Needs an
   explicit decision (and tests) that contiguous-instead-of-view outputs are
   acceptable; per the artifact-likelihood overlay these motivating patterns
   may be partition artifacts anyway — re-validate before upstreaming.
   (ALARMING #2)
3. **[a85d79a900a] Segment-aligned split lacks a partials-buffer/total-rows
   cap** — add `xnumel * split * itemsize` bound before overriding no-split.
   (ALARMING #3)
4. **[chain f6ae31e6984→e9a67c98a8c] evict_first chain must be squashed**: the
   WIP emits malformed `eviction_policy=''` codegen if cherry-picked alone,
   the flag name describes only 1 of 3 conditions, and
   `torch.cuda.get_device_capability()` checks the wrong device in multi-GPU.
5. **[52d4cadfac0] Hoisting pass (if ever enabled): CSE cache-hit can mark an
   in-loop variable as outside_loop_vars** → use-before-def compile error;
   also the commit smuggles in commit 3's config flag (re-slice). Since it is
   default-off and measured neutral, candidate for LOCAL-ONLY/drop.

## Upstream-Readiness Summary

- UPSTREAM-READY (5): 6703f38fa2d, 1406552b9d3, edbd4b67279 (squash into
  parent), 60dd3839913 (squash into parent), 56959375c33 (needs tests).
- NEEDS-WORK (8): bbf37d454c2+3bf69043be0 (stride contract + artifact
  re-validation), 5489b8c2bb9 (shape specialization, meta dtype),
  bdc289b3644 (flag semantics), e45f846f8d1+60431eca431+e9a67c98a8c (squash,
  rename, device fix), a26fc2c8bf4 (NaN default), a85d79a900a (memory cap,
  except-TypeError).
- EXPERIMENTAL (1): 52d4cadfac0 (default off, neutral perf — keep local or
  drop).
- LOCAL-ONLY/squash (2): 3b746fce660, f6ae31e6984 (superseded WIPs).

## Artifact-likelihood overlay (user assessment, 2026-06-11)

User: "some of the stuff, like slice compaction etc, are likely just artifact
errors." Applying the partitioner-artifact lens to the audited commits:

| Commit(s) | Pattern it serves | Artifact likelihood |
|---|---|---|
| bbf37d454c2 + 3bf69043be0 (slice compaction) | graph RETURNS slice/select view of computed intermediate | HIGH — real models rarely return raw slices of intermediates; the "output" is likely a partition cut (same mechanism as squeezenet scatter case). The stride-contract concern compounds: semantics bent for a possibly-synthetic pattern. |
| 5489b8c2bb9 (scatter gather-reduce) | scatter consumed ONLY by reductions | CONFIRMED artifact for squeezenet; 10 other repro wins unvalidated |
| 56959375c33 (overlap-add decode) | affine-iota scatter structure | Index-structure detection is real (Longformer's as_strided windows exist in the model); but consumer-set economics may shift with uncut consumers |
| Heuristic/codegen fixes (splits a85d79a900a/bdc289b3644, evict chain, fast combine a26fc2c8bf4, scalar-acc ungate 9dde2c59a51, rsqrt, dedupe) | kernel-level properties | LOW — fire on load strides/op patterns/config sets regardless of where graphs were cut; genai suite additionally artifact-free by construction |

Implication for upstreaming order: heuristic/codegen commits first (artifact-
robust + mostly UPSTREAM-READY/light-NEEDS-WORK); graph-shape passes
(slice compaction, scatter passes) wait for the user's partitioner fixes to
re-validate their motivating patterns. The planned IR-level view-liveness
generalization (task #7) inherits the same caveat: its llama flagship case
(last-token select IS a real model output pattern: logits[:, -1]) is the
exception worth keeping even if the slice-compaction family proves synthetic.
