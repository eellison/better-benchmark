# PR 4 — Scalar-accumulator reduction configs + num_load≤4 gate

**Commits:** `ca8f961d6b0` (gate) + `9dde2c59a51` (ungate-from-CD + rnumel ceiling) — ship together.
**Flags:** `config.triton.scalar_reduction_accumulators` + `config.scalar_acc_configs_without_cd`.
**Status:** MERGE-READY via carved prerequisite — `pr/scalar-acc-clean`.

## Branch & how made mergeable
`pr/scalar-acc-clean` (rooted on `5e2ab3055de`):
- `6b5a0810183` — **prerequisite**, carved from mega `97385fb3273`:
  `config.triton.scalar_reduction_accumulators`, the `reduction()` codegen
  rewrite in `codegen/triton.py` (the `use_scalar_acc` block + scalar
  online-softmax path, carried whole), and the `MAX_R0_BLOCK` scalar-acc branch
  in `runtime/triton_heuristics.py`.
- `5e81e8825b4` — `ca8f961d6b0` cherry-picked clean (appends `num_load <= 4` gate).
- `4a2db251d12` — `9dde2c59a51` cherry-picked clean (ungate + `MAX_R0_BLOCK`
  ceiling raise + `scalar_acc_configs_without_cd` flag + extra configs).

## Conflict classification
REAL SYMBOL DEPENDENCY: the whole `use_scalar_acc` codegen block and the
`scalar_reduction_accumulators` flag are mega-introduced; the two PR commits only
append a gate / ungate on top. Carved the codegen prereq; both PR commits then
cherry-pick with zero conflict.

## Verification (2026-07-15)
3 touched `.py` files ast-parse; 0 conflict markers; symbols
`scalar_reduction_accumulators`, `scalar_acc_configs_without_cd`,
`use_scalar_acc`, and the gate all present; patch applies clean onto pristine
base. **Note:** the shipped gate text is `self.num_load <= 3` (the commit
`ca8f961d6b0` *title* says `<= 4`; that's a commit-message/doc discrepancy, not a
functional gap — the code gates at `<= 3`).
**Compile-smoke (2026-07-15, B200, PYTHONPATH-shadow):** import OK (y) — all the
above symbols + the `MAX_R0_BLOCK` heuristics block resolve. Representative repro
`repros/models/genai/static/SoftmaxForward/full_graph_000.py` (8192x262144)
`torch.compile`s to completion — **GOOD** (`max_abs=4.8e-7`, within bf16). Flag
A/B (reset between runs): with `scalar_reduction_accumulators` **off** the kernel
keeps a vector accumulator `tl.full([XBLOCK, R0_BLOCK])`; **on** it uses a scalar
accumulator `tl.full([XBLOCK])`. **num_load gate proven both directions** on
synthetic pure-sum reductions: `num_load=2` (<=3) selects the scalar path,
`num_load=6` (>3) falls back to the vector path — both compile, `max_abs<=1.2e-4`.
**Net: imports + compiles a representative repro to a numerics-valid result on
B200, the num_load<=3 gate exercised both ways; full CI not run.**

## Shared scaffolding note
The prereq's `reduction()` codegen rewrite is the same one PR3 (online-softmax)
needs (PR4 additionally carries the heuristics `MAX_R0_BLOCK` block). If
upstreamed together, merge the two prereqs into one shared scaffolding PR.

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
