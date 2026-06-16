# Notes for oracle implementers — updated 2026-06-16 (late)

Status: **1714/1727 oracle_measured (99.2%)**. Only **13 needs_work** remain —
all sum-family reductions. The migration is essentially complete.

## The 13 remaining — ALL deterministic wrong-math in sum/sum_sum_sum reductions

These have an oracle.py but FAIL the bench numerics gate with LARGE
deterministic diffs (re-verified stable across unseeded runs — not stochastic,
not precision drift). The reduction math is wrong. Per-row max_diff is in the
queue (oracle_migration_queue.csv) notes column.

| dir | max_diff (failing output) |
|---|---|
| sum_sum_sum_4aae5698dd79 | 2.15e9 (out0 bf16) |
| sum_75456ad2c2c7 | 3.36e7 |
| sum_4fd6e4019857 | 1.68e7 (out0 bf16 [8192,262144]) |
| sum_7ba9dcb96142 | 8.39e6 |
| sum_9b2fcee49b0a | 2.10e6 |
| sum_sum_sum_d2c97d17f3dc | 1.05e6 |
| sum_898c72fb606a | 4096 |
| sum_444779f98932 | 2048 |
| sum_sum_sum_4a1fb62e29b0 | 16 (deterministic at 2/4 points) |
| sum_sum_1e07e3ba8c68 | 3.84 |
| sum_sum_sum_cc4b6a77cc6b | 2.0 |
| sum_sum_sum_ddcfccfb8340 | (impl re-audit: checks pass H100, fails B200 locked) |
| sum_sum_sum_ee5e53038768 | exact dense bf16 output3 mismatch |

Likely shared cause: these are multi-reduction (relu_grad/where + dual/triple
channel sums) BN/layernorm-backward-style patterns; a wrong reduction axis,
scale constant, or accumulation-dtype on one output. Fix the failing output's
math; the queue note names which output and the diff.

## Policy reminders (unchanged, in ORACLE_MIGRATION_INSTRUCTIONS.md)
- Numerics drift (ULP/accumulation/fp64-gate/stochastic-mask) is FLAG-NOT-BLOCK,
  not a rejection. The 13 above are NOT that — they are large deterministic diffs.
- `tl.where` self-blend / clamp-to-tolerance is bench-hacking and BLOCKS.
- Autotuning encouraged; a single non-autotuning oracle is a non-goal.

## Not implementer work (manager/infra side, for reference)
- ~144 dirs flagged no_valid_point in the sweep PASS a fresh --check (fp64
  bench-gate over-strictness) — harness issue, NOT oracle bugs. See
  nvp_triage_2026-06-16.md.
- detect_stochastic_outputs int64/bool gap: FIXED (commit 408922770).
- Model-level floor roll-up validated to ~5-9% of real e2e.
