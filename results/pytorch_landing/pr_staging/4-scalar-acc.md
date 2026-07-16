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
**Perf verification (2026-07-15, B200, A/B PYTHONPATH-shadow base 5e2ab vs branch
tip, fresh inductor cache per arm, bench_parallel locked path): REPRODUCES.**
genai full_graphs (8192×262144): SoftmaxForward 3656.8 → 2030.3us (**1.80x**;
claim was 3541→1905 ≈ 1.86x — same magnitude, base arm's 3657us confirms the
pre-fix level), CrossEntropyForward 2261.7 → 1024.8us (**2.21x**, beating the
"+60%+" expectation). Repeat run matched to <0.1% on both. Raw data:
`perf_verify/RESULTS.json` (this dir).

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

## Rebase onto current main + PR #190207 refresh (2026-07-16)
Rebased the change onto `origin/main` @ `ed8fd16c0e1` and force-pushed to fork
`bb/scalar-acc-reductions` (PR #190207, still **draft/[WIP]**). New tip
`9851b7b942b`.

**What moved (PRs #183277/#183278 refactor):** the reduction autotuning
heuristics were relocated from `runtime/triton_heuristics.py` into
`heuristics/triton_codegen/reduction.py` (now class-based:
`ReductionHeuristic.get_configs`; `_reduction_configs`/`reduction` in
`triton_heuristics.py` just delegate). Re-expressed there: the `MAX_R0_BLOCK`
scalar-acc branch (now at the `device_major` block, ~line 226), the
`_scalar_acc_configs_without_cd` helper, and the two extra large-`R0_BLOCK`
candidates wired into the `ReductionHint.INNER` return + `result_configs`.
`config.py` flags and the `codegen/triton.py` `reduction()` rewrite stayed in
place (that file did not move, though it diverged ~1000 lines — ported
hunk-by-hunk, not `git apply`).

**Simplifications:**
- Collapsed 3 commits → **2** (feature + gate).
- **Dropped the `online_softmax_cross_entropy` codegen scaffolding entirely** —
  it is not a real `ReductionType` on main (not in the `ops_handler` Literal, no
  lowering emits it), so all three of its branches were unreachable dead code.
- Fixed the `<=4`/`<=3` mismatch: code gates at `self.num_load <= 3` (as
  shipped/measured); corrected all commit-message/comment text to say `<= 3`.
- `frozenset({...})` for the reduction-type set → plain tuple (membership only;
  also sidesteps the `set_linter` autofix).

**Determinism-filter interaction (checked):** main added
`filter_reduction_configs_for_determinism` (`triton_heuristics.py`), called in
`reduction()` **after** `_reduction_configs`. Our extra scalar-acc candidates
only enlarge the candidate pool; in deterministic mode the filter dedups and
collapses to a single config via `_pick_config` (2nd-largest R0_BLOCK, then
num_warps, then XBLOCK). Nothing filters scalar-acc configs *by name* — no
correctness issue; det-mode may or may not pick a large-R0 config, which is the
intended det behavior. No change needed.

**Test:** added to `test/inductor/test_online_softmax.py` —
`test_scalar_acc_reduction_codegen` (asserts `[XBLOCK]` accumulator, in-loop
`tl.sum(...,1)`, no post-loop `tl.sum(_tmp`, numerics `same`) and
`test_scalar_acc_reduction_num_load_gate` (5 loads → `[XBLOCK, R0_BLOCK]` vector
path + post-loop `tl.sum(_tmp`, numerics `same`). Markers validated against
functionally-equivalent codegen from the installed torch (work2 tip, OLD
structure) — all assertions pass. This validates the codegen strings, not
rebased-tree perf.

**Verify honesty:** ast-parse (4 files) + `lintrunner -a` clean (only
environmental PYREFLY `No attribute in module torch._C` from unbuilt torch, repo-
wide, not our edits) + manual review. **Rebuild deferred** — no main-built `.so`;
the importable torch is the work2 tip `daa79cd`, a *different* tip, so it does
NOT cover the rebased scalar-acc tree. Perf (1.80x SoftmaxForward / 2.21x CE) is
cited as **PRIOR** (June base), not re-measured on this rebase.

written with claude code
