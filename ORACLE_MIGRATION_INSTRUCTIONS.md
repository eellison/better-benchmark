# Oracle Migration — Agent Instructions

You are migrating oracle kernels from the OLD corpus (`repros/canonical/`,
1482 dirs, fp32, ~1479 with oracle files) to the NEW corpus
(`/tmp/scratch_space/recapture_corpus/repros/canonical/`, 1727 dirs,
bf16/AMP, new pattern hashes). This document is the contract. Read
`CORPUS_MIGRATION_PLAN.md` §5 (oracle migration) and §6 (atomic flip) for
the governing invariants; read `AGENT_INSTRUCTIONS.md` for repo-wide rules
(especially: never regex/AST over generated artifacts; verification means
consuming your own outputs; composability check before claiming closures).

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

Registration: `@oracle_impl(hardware="B200", point="<shape_hash>")` per
shapes.json point the oracle covers (dispatch is by (pattern_hash,
shape_hash) — the hash registration decision). Multi-point patterns:
either one kernel covering all points (preferred when configs allow) or
per-point registrations.

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

## Claim protocol (same flow as the June queue)

The work queue is `investigation_results/oracle_migration_queue.csv`
(generated from the new corpus after the bench sweep; columns:
queue_rank, new_pattern_hash, repro_dir, family, gap_vs_sol, status,
old_oracle_candidates, owner, notes).

1. CLAIM: pick the highest-ranked unclaimed rows (batch of 5-6), set
   status=claimed + owner=<your-name> in ONE commit ("Claim oracle batch
   NNNN-NNNN"). Any repro without an oracle and without an owner is
   eligible — no other gatekeeping.
2. IMPLEMENT: write `oracle.py` per the format above.
3. VALIDATE: `check_oracle` numerics at bf16 tolerances against the repro
   at EVERY registered point. A failing point = not done.
4. MEASURE: bench through the standard harness (CUDAGraph + CD, min mode,
   `INDUCTOR_GPU_BENCH_LOCK=1`) — never hand-rolled timing loops; use
   bench_parallel for batches. Record measured_oracle_us in the queue row.
5. MARK: status=oracle_measured, one commit per batch. Include any
   pattern you investigated and decided needs NO oracle (gap is launch
   floor / SOL-bound already) as status=no_oracle_needed with one line of
   why.

## Verification gates (you are not done until)

- `check_oracle` passes at every registered point (bf16 tolerances).
- Measured floor <= Inductor's compiled time at the same point (an
  "oracle" slower than the compiler is a bug or a no_oracle_needed).
- The queue row carries measured numbers, not estimates.
- For big closures: the composability check (AGENT_INSTRUCTIONS rule 6) —
  read the source model's full graph before claiming model-level impact.

## What you do NOT do

- Touch the old corpus (read-only reference).
- Touch capture/pipeline/harness code (file an issue in your batch notes
  if you find a bug; the corpus side is managed separately).
- Benchmark on a GPU without the bench lock.
- Mark anything done without measured numbers.
