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
| 7 | 3b746fce660 | WIP: slice-output compaction extension | PENDING | superseded-or-half check | |
| 8 | bdc289b3644 | Split partially-saturated multi-output reductions | PENDING | | |
| 9 | 3bf69043be0 | Extend slice-output compaction (select.int etc.) | PENDING | | |
| 10 | f6ae31e6984 | WIP: r-contiguity for evict_first | PENDING | superseded-or-half check | |
| 11 | e45f846f8d1 | Drop evict_first misaligned strides | PENDING | eviction-chain coherence | |
| 12 | 60431eca431 | Gate evict_first drop on in-loop side stores | PENDING | eviction-chain coherence | |
| 13 | e9a67c98a8c | Restrict evict_first drop to >L2 iteration space | PENDING | eviction-chain coherence | |
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
