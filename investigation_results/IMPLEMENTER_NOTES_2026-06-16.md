# Notes for oracle implementers — 2026-06-16

Status: 1570/1727 oracle_measured (90.9%). 131 needs_work (65 real-bug w/
oracle + 66 missing-oracle to author), 26 claimed in flight.

## Policy changes — READ (now in ORACLE_MIGRATION_INSTRUCTIONS.md)

1. **Numerics are flag-not-block.** Precision-level drift (accumulation order,
   rounding, ULP, fp64-anchor) is FLAGGED for investigation, NOT rejected.
   Do NOT hand-tune an oracle to slip under the numerics gate, and do NOT
   prescribe a per-oracle tolerance. The reference may itself be stricter than
   real Inductor output; gate strictness is resolved centrally. This means you
   should stop over-working the var_mean / reduction numerics.

2. **The `tl.where` self-blend BLOCKS (still).** Computing the answer two ways
   and selecting per-element whichever lands inside tolerance
   (`tl.where(abs(a-b)<=tol, a, b)`, clamp-to-reference, etc.) is bench-hacking,
   not a numerics issue — it fabricates a pass. Commit to ONE faithful path.

3. **Autotuning is encouraged; a single non-autotuning oracle is a non-goal.**
   Carry multiple configs (multi-dispatch to one kernel that autotunes, or
   per-shape implementations). Don't collapse to one fixed BLOCK/num_warps.

## Where to focus the remaining work

- **66 missing oracles** (needs_work, no oracle.py) — pure authoring, the
  biggest lever. Highest priority for free hands.
- **Real-bug cluster** (oracle exists, large-diff wrong math): pointwise (27),
  sum_sum_sum (10), sum_sum (7), mean (6), var_mean (5), amax_sum (5), sum (4).
  Each row's queue note carries the failing output + max_diff (some off by 1e6+,
  e.g. sum_6d8612892024 @ 8.4e6, sum_sum_sum_d2c97d17f3dc @ 5.4e9). These are
  genuine math bugs, not precision.

## Manager-side status (no action needed from you)
- Model-tool validation is DONE on v2: per-pattern oracle floors compose to
  model floors, reconciled against real HF-infer e2e (see
  v2_model_rollup_validation.md). Modeled fusible speedup ~2.3-2.65x.
- Verification keeps pace with deliveries; nothing is blocked on the manager.
