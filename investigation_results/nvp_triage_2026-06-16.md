# no_valid_point triage — 2026-06-16

The corrected --oracles sweep flagged 252 dirs `no_valid_point` (check-gate
failed at every shape point so no floor recorded). Triaged all 252 by WHY:

| category | count | verdict |
|---|---|---|
| FP64_GATE_FN | 159 | `--check` PASSES on recheck — bench-gate false-negative, oracle FINE |
| STOCHASTIC_MASK | 64 | only bool dropout-mask (RNG) fails — flag-not-block, oracle FINE |
| NUMERICS_SMALL | 9 | tiny float drift <=0.2 — flag-not-block, oracle FINE |
| REAL_BUG | 20 | genuine wrong math — the actual remaining work |
| CUDA_ERROR | 0 | |
| NO_ORACLE | 0 | |

**232/252 (92%) are harness false-negatives (oracle is fine, gate too strict).**
**Only 20 are genuine bugs.**

## REAL_BUG list (genuine remaining work — implementer fix)

- `pointwise_4254ac4c0d96` — max_diff=unknown
- `pointwise_584a8c609627` — max_diff=unknown
- `sum_444779f98932` — max_diff=2050.0
- `sum_4fd6e4019857` — max_diff=16800000.0
- `sum_75456ad2c2c7` — max_diff=33600000.0
- `sum_7ba9dcb96142` — max_diff=8390000.0
- `sum_898c72fb606a` — max_diff=4100.0
- `sum_9b2fcee49b0a` — max_diff=2100000.0
- `sum_sum_1e07e3ba8c68` — max_diff=3.84
- `sum_sum_sum_4a1fb62e29b0` — max_diff=32.0
- `sum_sum_sum_4aae5698dd79` — max_diff=2160000000.0
- `sum_sum_sum_cc4b6a77cc6b` — max_diff=2.0
- `sum_sum_sum_d2c97d17f3dc` — max_diff=2280000000.0
- `sum_sum_sum_ddcfccfb8340` — max_diff=1020.0
- `var_mean_087d5b4f064d` — max_diff=unknown
- `var_mean_0d6f1eb6e0c6` — max_diff=unknown
- `var_mean_3bc311a8676a` — max_diff=unknown
- `var_mean_8187f7c4e01e` — max_diff=unknown
- `var_mean_a7b32508693f` — max_diff=unknown
- `var_mean_dc4abd356145` — max_diff=unknown

## Infra follow-up
- 159 FP64_GATE_FN dirs PASS a fresh `--check` but failed during the
  bench-time gate — the fp64-anchored bench gate is transiently stricter/less
  stable than standalone check. This costs ~that many valid floors in the
  timings file. Worth a harness fix (gate seeding/stability). NOT oracle bugs.
- STOCHASTIC_MASK dirs: the bench gate exact-compares a stochastic dropout
  mask; should skip/flag per the stochastic policy.
