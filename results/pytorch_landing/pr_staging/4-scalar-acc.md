# PR 4 — Scalar-accumulator reduction configs + num_load≤4 gate

**Commits:** `ca8f961d6b0` (gate) + `9dde2c59a51` (ungate-from-CD + rnumel ceiling) — ship together.
**Flags:** `config.triton.scalar_reduction_accumulators` + `config.scalar_acc_configs_without_cd`.
**Status:** DEPENDS ON mega-commit — `scalar_reduction_accumulators` is introduced there; conflicts on bare base (triton_heuristics.py).

## Summary
Enables scalar-accumulator reduction configs (ungated from coordinate-descent, with
a raised rnumel ceiling), GATED to kernels with `num_load ≤ 4`. Ship the two commits
together: the gate is a regression-fixer — the ungated version caused measured
14–22% regressions, which the num_load≤4 gate eliminates.

## Measured impact
- **Kernel-geomean +0.59 / +0.44pp; per-model e2e +0.18 / +0.06pp.**
- THE genai lever: SoftmaxForward 3541→1905us (+45%), CrossEntropy fwd/bwd +64–67%.

## Test plan
Softmax/CE/reduction genai micros; verify the gate holds (no >2% regressions on the
num_load>4 cohort). Flag-gated A/B.

written with claude code
