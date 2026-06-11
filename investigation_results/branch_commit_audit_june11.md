# Branch Commit Audit — June 11, 2026

**Branch:** `pr-184905` in /tmp/pytorch-work
**Range:** `049d1229d84..HEAD` (17 core commits + 1 WIP from another agent)
**Rubric:** UPSTREAM-READY / NEEDS-WORK / EXPERIMENTAL / LOCAL-ONLY (per `pytorch_upstream_audit.md`)
**Status:** IN PROGRESS — skeleton committed first per audit protocol; verdicts filled incrementally.

---

## ALARMING CORRECTNESS FINDINGS

(filled as found — empty means none found yet)

- TBD

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
| 14 | a26fc2c8bf4 | Fast combine for online-softmax loops | PENDING | | |
| 15 | 60dd3839913 | dedupe_graph_outputs earliest-node canonical | PENDING | | |
| 16 | 56959375c33 | Overlapping-window index_put -> overlap-add | PENDING | index math | |
| 17 | a85d79a900a | Segment-aligned split introduction | PENDING | L-detection edges | |
| — | 6f7fd7d74ec | WIP: ungate scalar-acc configs from CD | IN-PROGRESS | another agent's active work | not reviewed deeply |

---

## Per-Commit Detail

(filled incrementally)

## Spot-Checks (3 headline claims)

(done last, after table complete)

| Claim | Repro | Claimed | Measured | Verdict |
|-------|-------|---------|----------|---------|
| TBD | | | | |

## Top-5 Prioritized Issues

TBD

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
