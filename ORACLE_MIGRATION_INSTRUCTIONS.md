# Oracle Migration — Agent Instructions

## Two roles (split as in the June flow)

**IMPLEMENTERS** (typically on other servers): claim queue rows, write
oracle.py files, validate numerics, measure floors, mark rows. The bulk of
this document is your contract — start at "Coverage target".

**THE MANAGER/OVERSEER** (one agent, on the corpus box): does NOT write
oracles. Responsibilities:

1. **Queue integrity**: regenerate `oracle_migration_queue.csv` from the
   corpus when it changes (`scripts/generate_oracle_queue.py`); resolve
   claim collisions (duplicate owners on a row: earliest commit wins);
   periodically sweep rows stuck in status=claimed with no commits from
   that owner (>1 day -> reset to unclaimed with a note).
2. **Batch verification** (the June "Batch-verify 84/100" role): for each
   batch marked oracle_measured, INDEPENDENTLY re-run check_oracle + the
   bench at a sample of points (all points for small batches) on this box.
   Numbers that don't reproduce (>10% drift) -> status back to
   needs_work with measured-vs-claimed numbers in notes. Trust nothing
   unmeasured locally.
3. **Compile-gap investigation**: for verified oracles, compute and record
   speedup-vs-compile per point (oracle_us vs Inductor compiled_us, same
   CUDAGraph methodology) — this is the gap table the whole exercise
   exists to produce. Mark slowdowns (oracle worse than compile) loudly:
   either the oracle is bad (back to needs_work) or Inductor is already
   at floor there (recorded as an at_floor note on the measured oracle
   row, never as a skip).
4. **Stale sweeps** (the June "Stale-sweep 110 entries" role): when
   methodology or pytorch-under-test changes, re-measure affected rows
   with the standard harness rather than trusting old numbers.
5. **Scope policing**: enforce the full-scope rule — an oracle measuring a
   SUBSET of the repro's computation (just the reduction, not the
   surrounding fused ops) is NOT a floor; reject with the missing scope
   named in notes.
6. **Progress reporting**: rows closed / by family / by status, weekly
   into investigation_results/.

The manager has GPU access on this box; implementers' numbers are
provisional until manager-verified. (History: INVEST_INSTRUCTIONS.MD is
the June-era predecessor of this document — same claim flow, this doc
supersedes it for the migration.)

You are migrating oracle kernels from the OLD corpus (`repros/canonical/`,
1482 dirs, fp32, ~1479 with oracle files) to the NEW corpus
(`repros_v2/canonical/ (in-repo)`, 1727 dirs,
bf16/AMP, new pattern hashes). This document is the contract. Read
`CORPUS_MIGRATION_PLAN.md` §5 (oracle migration) and §6 (atomic flip) for
the governing invariants; read `AGENT_INSTRUCTIONS.md` for repo-wide rules
(especially: never regex/AST over generated artifacts; verification means
consuming your own outputs; composability check before claiming closures).

## Reference documents (read the ones for your role)

BOTH roles:
- `AGENT_INSTRUCTIONS.md` — repo-wide rules (graphs not regex, verify by
  consuming outputs, bench_parallel for batches, composability check).
- `BENCHMARKING_FLOW.md` — the standard measurement pipeline.

IMPLEMENTERS additionally:
- `INVEST_INSTRUCTIONS.MD` §"What NOT to do" + §"Benchmarking" — the
  SAME-NUMERICS rule (verbatim binding here): oracles must use the same
  math formulation and dtype rounding boundaries as Inductor for the
  captured repro — no fast/approx exp, no exp2-based softmax, no fast
  GELU/tanh/sigmoid substitutions, no moved casts, unless Inductor
  generated that exact formulation; `aten.exp` means natural exp
  (libdevice.exp), and any nonstandard lowering choice is documented in
  the gap diagnosis. ALSO the no-inline-timing rule: bench only via
  oracle_harness.bench_oracle() — inline do_bench/perf_counter timing
  fabricates 2.3x fake gaps (verified) from dispatch overhead.
- `scripts/ORACLE_FORMAT.md` — the standardized format rationale (this
  document's slim format supersedes its template for NEW oracles, but its
  scope rules — full Repro() scope, no subset microbenchmarks — stand).

MANAGER additionally:
- `ORACLE_GAP_CLOSING_PLAYBOOK.md` — WHAT TO PROBE when verifying a
  measured gap: the 6-step process (characterize compile path: kernel
  census under fresh cache + CD; config sweep: coordinate_descent,
  max_autotune, combo_kernels, persistent-vs-looped, multi_kernel;
  algebraic-elimination check; oracle tuning check; classify
  closed/needs_work/confirmed; CSV update with configs-tried recorded).
  Includes the common-gap-patterns table with expected ranges per family.
  A gap nobody tried to close is not a confirmed gap.
- `ADVERSARIAL_REVIEW.md` — the pre-clean checklist (worker-queue
  recovery semantics, corpus validity, capture/partitioning rules,
  benchmark validity: exclusive-timing discipline, multi-worker variance).
  Restored 2026-06-12 from git history (deleted 05-21) — it is the
  reviewer's deep-probe list.
- `investigation_results/SESSION_SUMMARY.md` — the June campaign's
  execution framework: 3-5 parallel subagents sustained; any ratio >1.05x
  gets dedicated investigation with root-cause deliverables.
- `INVEST_INSTRUCTIONS.MD` §status/scope rules — what oracle_measured may
  and may not claim; subset-scope rejection criteria.

## The one invariant that governs everything

**Never a state with repros and no oracles.** The old corpus + old oracles
remain authoritative until a new pattern has (a) a benchmarked gap, (b) a
working oracle (or an explicit no-oracle-needed verdict), and (c) measured
oracle numbers. The §6 GC flip is gated on this mechanically. You are
ADDING coverage to the new corpus; you never delete or modify anything in
the old corpus.

## Why hashes don't match (read this before "porting by hash")

Only ~16 of 1479 oracle-bearing old pattern hashes appear in the new
corpus. This is EXPECTED: the hash function changed during recapture
hardening (dtype/str args, structural int dims, canonical node order,
stride in shape_hash) and the corpus moved fp32 -> bf16 (dtype is part of
identity). Hash equality is therefore NOT the migration key. Match
patterns by:

1. `meta.json` `origin_ops` multiset + `kind` (reduction types), then
2. shape compatibility (same rank/dims family in shapes.json points), then
3. reading both repro.py graphs side by side (they are small).

A helper may be built for candidate matching, but the CLAIM of a match is
yours to verify by reading the graphs — op multiset collisions are real.

## Migration cases (from plan §5)

For each new-corpus pattern you claim:

- **Trace-equivalent survivor** (same computation, same dtype): port the
  oracle file as-is, re-measure, done. Rare (dtype changed corpus-wide).
- **Same computation, new dtype** (the common case, fp32 -> bf16): port
  the kernel, re-validate numerics at bf16 tolerances, RE-TUNE configs —
  bf16 shifts the roofline (memory-bound -> instruction-bound); fp32
  configs are not presumed optimal. Re-measure the floor.
- **Genuinely new pattern** (no old counterpart): fresh oracle through the
  normal queue, priority by gap.
- **Old pattern with no new counterpart**: do nothing (it dies with the
  flip); note it in your batch log.

## NEW oracle format (mandatory — do not copy old boilerplate)

Old oracle files carry ~180 lines of boilerplate each. New-format oracle
files contain EXACTLY:

1. A docstring with the gap diagnosis (classification + why Inductor
   can't do this today + what the oracle does differently). Port this
   from the old file when migrating — it is the valuable part.
2. The Triton kernel(s). `import triton` unguarded — it is a hard dep;
   an eager fallback silently corrupts floor measurements.
3. One `@oracle_impl(...)`-decorated forward function.

DO NOT include: input validation (`_validate_inputs`, shape/dtype/rank
checks — the registration IS the contract; the harness feeds exactly the
repro's inputs), a `_torch_full_scope` reference reimplementation (the
repro IS the reference; `check_oracle` compares against `Repro()`), a CLI
`main()` (one shared runner exists), path manipulation
(`Path(__file__).resolve()...` — resolve by registration, not location),
or `if triton is not None` guards.

File name: `oracle.py` in the canonical dir (one known name — the
registry is enumerated by walking `canonical/*/oracle.py`). If a pattern
genuinely needs multiple alternative implementations, `oracle_<variant>.py`
additionally, but `oracle.py` is the registered floor.

Registration (SETTLED 2026-06-12, landed in oracle_harness):
`@oracle_impl(hardware="B200", point="<shape_hash>")` — the shape_hash is
THE key. It resolves against the SIBLING shapes.json at import time: an
unknown/stale hash raises immediately (never a silent dispatch miss).
Quote the human-readable signature in a comment above the decorator if
helpful; it is documentation, never parsed. Multi-point patterns: one
registration per point (one kernel body registered N times with per-point
configs), or shapes=None/point=None for a genuinely shape-general kernel.
Example:

    @oracle_impl(hardware="B200", point="6d28ca52",
                 BLOCK=2048, num_warps=8)
    # (T([512], bf16), T([128,512,7,7], bf16, stride=(25088,1,3584,512)))
    def oracle_forward(inputs, *, BLOCK, num_warps): ...

## Inputs: structured, never parsed

New-corpus shapes.json points carry structured `inputs` (compact codec:
`[[shape, dtype, {opts}]...]`) — `load_shape_configs` +
`make_inputs_from_config` is the ONLY way to build inputs. Respect
`gen` kinds (index bounds, permutations, constants, offsets) and
`alias_group`/`alias_group_nbytes` (members are views of ONE buffer —
benchmarking them as separate allocations misstates memory behavior).
Never eval signature strings; never regex repro.py.

## Mine the old corpus for inspiration — it usually transfers

The old oracles are 1479 worked solutions. Even when no exact pattern
match exists, the FAMILY-level techniques transfer (bf16 changes configs,
rarely the algorithmic idea). Before writing a kernel from scratch:

1. Check `old_oracle_candidates` in your queue row (pre-computed op-multiset
   matches), and ALSO search the old corpus by family: the old queue
   (`investigation_results/oracle_kernel_work_queue.csv`) has a `family`
   column (online_softmax_cross_entropy, structured_pool_upsample_backward
   _reduce, irregular_gather_reduce, multi_output_reduction_templates, ...)
   mapping families -> proven oracle files.
2. Check PER MODEL: your new pattern's shapes.json names its source models
   (`models` keys). The old corpus has per-model manifests — find the old
   patterns from the SAME model (`repros/models/<suite>/<mode>/<model>/
   manifest.json` -> pattern hashes -> canonical dirs -> their oracles).
   The same model's old partitions are the closest prior art for its new
   partitions, even across the hash change.
3. The gap-diagnosis docstrings in old oracles are a catalog of WHY
   Inductor loses (atomic contention, low-wave splits, dependent
   reduction re-reads, layout copies...). Matching your new pattern's
   profile against that catalog is faster than rediscovering the cause.

Port ideas freely; port configs skeptically (fp32-tuned, B200-roofline
shifted under bf16); port boilerplate never.

## Coverage target: EVERY canonical repro, EVERY shape point

The goal is exhaustive: all 1727 canonical patterns get an oracle, and
the oracle must cover EVERY point in the pattern's shapes.json —
validated and measured at each one. No prioritization gate: rankings
exist in the queue only so agents working top-down tend to hit high-value
patterns first; nothing is out-of-scope and the migration is done when
every row has a measured oracle, not when the "important" ones are. There
is no skip verdict for patterns that look hard, tiny, or already
optimized; "at floor" is a measurement note after an oracle exists.

Per-point discipline:
- `check_oracle` must pass at every point (one kernel, many shapes — or
  per-point registrations where configs genuinely differ).
- Measure at every point; a pattern with 8 points gets 8 floor numbers.
- If a kernel is correct everywhere but only FAST at some points, register
  it everywhere and note the slow points — a floor that ties Inductor at
  small shapes is still a floor; do NOT silently narrow coverage.

## Operational rules (carried from the June campaign, updated)

- Branch: investigations-june-2026. Pull latest before every claim batch.
- The PARENT process (per implementer instance) owns ALL csv edits,
  commits, rebases, pushes, and final validation. Subagents edit ONLY
  their assigned repros/canonical/<repro_id>/oracle.py — no CSV edits, no
  commits, no pushes, and never revert or touch others' edits.
- Claim before writing: set status/owner, commit AND PUSH the claim, then
  dispatch subagents.
- `python -m pip install --no-build-isolation -e .` once per checkout
  before any oracle work (imports resolve via the package, never
  sys.path/REPO_ROOT hacks).
- Preserve eps, correction, rsqrt placement, dtype casts, RNG semantics,
  alias/view behavior, strides, and the returned-output scope exactly.

### Required gap-diagnosis format (docstring paragraph 1, exactly)

    Gap diagnosis (classification: <CLASS>): this oracle <specific
    behavior>, whereas Inductor <specific current behavior>; Inductor
    cannot do this today because <specific scheduler/codegen/pattern
    limitation>; the fix is <CLASS>: <specific Inductor change>.

Allowed classifications ONLY: SCHEDULER_FUSION, SCATTER_REDUCE,
COOPERATIVE_SPLIT_K, ALGEBRAIC_ELIMINATION, NEW_PATTERN.

### Worker prompt requirements (what each subagent dispatch must say)

When the parent dispatches a per-repro subagent, the prompt MUST include,
beyond ownership/branch boilerplate:

1. **This is a MIGRATION, not greenfield**: before designing a kernel,
   look at the old corpus. Concretely: (a) the queue row's
   old_oracle_candidates dirs (exact/near op-multiset matches — read
   their oracle files and gap diagnoses first); (b) the SAME-MODEL prior
   art: this pattern's shapes.json models keys -> the old corpus's
   repros/models/<suite>/<mode>/<model>/manifest.json -> that model's old
   pattern dirs -> their oracles. An old oracle for a similar partition
   of the same model is usually the right starting kernel — port the
   idea, retune for bf16, never copy boilerplate.
2. **EVERY shape point**: the oracle must check_oracle-pass AND be
   benched at every point in shapes.json — a pattern with 8 points means
   8 validations and 8 floor numbers in the report. Correct-but-slow
   points are registered and noted, never dropped.

### Verification protocol (run and REPORT all of it)

    python -m py_compile repros/canonical/<id>/oracle.py
    python -m oracle_harness repros/canonical/<id> --check
    python -m oracle_harness repros/canonical/<id> --check --no-skip-stochastic
    INDUCTOR_GPU_BENCH_LOCK=1 python -m oracle_harness repros/canonical/<id> --bench

(The shared runner replaces the old per-file `--check/--bench` CLI; slim
oracles have no main().) If --bench prints ANY CUDAGraph warning the
measurement is INVALID — report it as invalid, never accept the number.
A subagent's final report must list: changed file path, classification,
PASS/FAIL per check, bench numbers/status per point, CUDAGraph warning
status, and a short numeric audit (which lowerings were chosen for
exp/tanh/sigmoid-class ops and why they match the repro).

## Claim protocol (same flow as the June queue)

The work queue is `investigation_results/oracle_migration_queue.csv`
(generated from the new corpus; one row per canonical pattern, ALL 1727
present; columns: queue_rank, new_pattern_hash, repro_dir, family,
n_points, status, old_oracle_candidates, owner, notes).

1. CLAIM: pick unclaimed rows (batch of 5-6), set status=claimed +
   owner=<your-name> in ONE commit ("Claim oracle batch NNNN-NNNN"). Any
   repro without an oracle and without an owner is eligible — no other
   gatekeeping.
2. IMPLEMENT: write `oracle.py` per the format above.
3. VALIDATE: `check_oracle` numerics at bf16 tolerances against the repro
   at EVERY shapes.json point. A failing point = not done.
4. MEASURE: bench through the standard harness (CUDAGraph + CD, min mode,
   `INDUCTOR_GPU_BENCH_LOCK=1`) at EVERY point — never hand-rolled timing
   loops; use bench_parallel --all-shapes for batches. Record per-point
   measured_oracle_us in the queue row (or a results file referenced by
   the row when points are many).
5. MARK: status=oracle_measured, one commit per batch. Every claimed
   pattern must land an oracle.py. If Inductor is already at launch floor
   / SOL at every point, record `at_floor` and the per-point evidence in
   notes/results, but still keep the row as a measured oracle.

## Verification gates (you are not done until)

- `check_oracle` passes at EVERY shapes.json point (bf16 tolerances).
- Measured floor <= Inductor's compiled time at each point (an
  "oracle" slower than the compiler at a point is a bug to fix or mark
  needs_work; an at-floor pattern should tie, not be skipped).
- The queue row carries per-point measured numbers, not estimates.
- For big closures: the composability check (AGENT_INSTRUCTIONS rule 6) —
  read the source model's full graph before claiming model-level impact.

## What you do NOT do

- Touch the old corpus (read-only reference).
- Touch capture/pipeline/harness code (file an issue in your batch notes
  if you find a bug; the corpus side is managed separately).
- Benchmark on a GPU without the bench lock.
- Mark anything done without measured numbers.
