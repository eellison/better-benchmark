# PR 1 — Canonicalize reciprocal(sqrt(x)) → rsqrt(x)

**Commit:** `a73d1583b34` · **Status:** STANDALONE, cherry-picks clean onto mainline.
**Branch:** `pr/rsqrt-clean` = `5e2ab3055de` + `a07f66f9996` (cherry-pick of
`a73d1583b34`). **How made mergeable:** nothing needed — clean standalone pick.
**Verification (2026-07-15):** roots on base; 0 conflict markers; the one touched
file `torch/_inductor/fx_passes/post_grad.py` ast-parses; patch applies clean onto
a pristine `5e2ab3055de`. **Caveat:** ast-parse only — no torch build here, not
compiled or run.

## Summary
A post-grad peephole that rewrites `reciprocal(sqrt(x))` and `div(1, sqrt(x))` to
`rsqrt(x)`. ~45 lines, one file (`torch/_inductor/fx_passes/post_grad.py`).

## Measured impact (base 5e2ab → this commit, 158-model corpus)
- **Kernel-geomean: +5.64pp.** **Per-model e2e: +2.11pp** — ~half the entire perf
  branch's model-level gain, from one peephole.
- Concentrated on conv/BatchNorm models: var_mean family ~2.2× per-kernel, ~1.15×
  per affected conv model (repvgg ~+26% e2e); ~0 on models without the pattern.
- Above both noise floors (kernel ±0.2%, model ±0.82pp).

## Mechanism
`reciprocal(sqrt(x))` lowers to a ~30-instruction correctly-rounded software
`sqrt_rn` + a divide, run per-element; `rsqrt(x)` emits a single hardware
`rsqrt.approx.ftz.f32` MUFU instruction. On the affected BN/var_mean kernels this
flips them from compute-bound (2.5–3.7× over SOL) to memory-bound (~1×).

## Numerics
`rsqrt.approx` is flush-to-zero and marginally less precise than the software
sqrt+divide. Measured deviation on the corpus is sub-bf16-ULP. Universally-correct
identity otherwise (1/sqrt(x) == rsqrt(x)).

## Test plan
Canonical repro families exercising it: `var_mean_*` (BatchNorm), conv-BN infer
models (repvgg_a2, mobilenetv2_100, inception). Numerics-gated against the repro.

written with claude code
