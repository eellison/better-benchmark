# Numerical emulation policy

The final promoted-reference audit classified 4,977 frozen timeline points:

- 4,468 passed with default TorchInductor.
- 10 failed by default and passed with
  `emulate_precision_casts=True`.
- 28 remained degraded with and without emulation.
- 471 had only stochastic outputs and were excluded from the numerical
  verdict.

The compile policy is an explicit semantic opt-in, not an automatic retry.
It is stored on an exact `shapes.json` input point:

```json
"compile_policy": {
  "inductor": {"emulate_precision_casts": true}
}
```

Default-passing points have no annotation. Each result records its resolved
compile policy, and the same point cannot be compared across different
policies.

## Opted-in points

| Repro | Shape hash |
| --- | --- |
| `sum_sum_34b54dfb6c54` | `77702286` |
| `sum_sum_643db2887a01` | `18f48531` |
| `sum_sum_sum_c2ae956520f9` | `cf12eab2` |
| `sum_sum_sum_d0cb44a92f6d` | `09b68f1c` |
| `var_mean_mean_5f6d60b04d02` | `79146166`, `8881253b`, `c99f0cec`, `77734290`, `f7eda15e` |

These are the nine non-caveated rescued points. A follow-up audit also ran
emulation across all 81 shapes in the three mixed families; all remained
within the accepted verdict. That check was investigative only: the 78
default-passing points remain unannotated.

## Deliberately not opted in

- `pointwise_43588d3e9780`: emulation reproduces eager nonfinite behavior,
  while default compilation stays finite and is closer to the promoted FP32
  reference. This needs an explicit accuracy-objective decision.
- The 28 both-degraded points: emulation does not rescue them.
- The 471 stochastic-only points: these are excluded rather than failed.
