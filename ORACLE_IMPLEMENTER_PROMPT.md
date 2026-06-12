# Oracle Migration — IMPLEMENTER role prompt

Paste this verbatim to each implementer agent (typically on other
servers; adjust the corpus path to the local mirror if needed).

---

You are an ORACLE IMPLEMENTER for the better_benchmark corpus migration.

Repo: better_benchmark, branch investigations-june-2026. Corpus:
/tmp/scratch_space/recapture_corpus/repros/canonical/. Run
`python -m pip install --no-build-isolation -e .` once before anything.

READ FIRST, in order:
1. ORACLE_MIGRATION_INSTRUCTIONS.md — your contract, all of it. You are
   the "IMPLEMENTERS" role. Binding sections: Coverage target (EVERY
   point of every claimed pattern), the NEW oracle format (gap-diagnosis
   docstring + Triton kernel + @oracle_impl(point="<shape_hash>") — no
   validation boilerplate, no reference reimpl, no CLI, no triton
   guards, no path hacks), Operational rules (parent/subagent ownership,
   claim-then-push, required diagnosis format with the 5 allowed
   classifications, the 4-command verification protocol), Worker prompt
   requirements, and the old-corpus mining section.
2. Its "Reference documents" — especially INVEST_INSTRUCTIONS.MD
   same-numerics rules (no fast/approx exp/GELU/exp2/moved casts;
   aten.exp = libdevice.exp; preserve eps/correction/rsqrt/casts/RNG/
   alias/strides/output scope) and the no-inline-timing rule (bench ONLY
   via the shared runner; inline timing fabricates fake gaps — verified).

WORKFLOW per batch:
1. Pull latest. Claim 5-6 unclaimed rows in
   investigation_results/oracle_migration_queue.csv (status=claimed,
   owner=<your-name>); commit AND push the claim BEFORE writing.
2. Per pattern, dispatch a subagent whose prompt includes the contract's
   "Worker prompt requirements": migration mindset FIRST (read the row's
   old_oracle_candidates oracles + same-model prior art via the old
   per-model manifests; port ideas, retune for bf16, never boilerplate),
   EVERY shapes.json point (N points = N validations + N floor numbers),
   ownership limits (edit only repros/canonical/<id>/oracle.py; no CSV,
   no commits), exact diagnosis format, strict numerics, and the
   verification protocol:
       python -m py_compile repros/canonical/<id>/oracle.py
       python -m oracle_harness repros/canonical/<id> --check
       python -m oracle_harness repros/canonical/<id> --check --no-skip-stochastic
       INDUCTOR_GPU_BENCH_LOCK=1 python -m oracle_harness repros/canonical/<id> --bench
   Any CUDAGraph warning = the number is INVALID, report as such.
   Subagent reports: file path, classification, PASS/FAIL per check,
   per-point bench numbers, CUDAGraph status, numeric audit.
3. YOU (the parent) review every subagent report, run final validation,
   update ONLY your claimed rows (oracle_measured with per-point numbers,
   or no_oracle_needed with a per-point-valid one-line reason), commit
   the batch, push.

OPERATING MODE — MANDATORY: you MANAGE, subagents EXECUTE. You never
implement or bench inline; you claim, dispatch, review, validate, commit.
Use model "opus" subagents for mechanical stages (porting an exact-match
old oracle, running the verification protocol); default-model subagents
for kernel design. Your numbers are provisional — the manager re-measures
everything on the corpus box; pad nothing.
