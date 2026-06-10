# Corpus Migration Plan — Recapture, Oracle Migration, Atomic Flip

Standing reference for migrating the corpus from its current state
(fp32 captures, pre-fidelity-fix partitions, 404 models without graphs)
to the new state (AMP-train/bf16-infer, fidelity-correct partitions,
full graph + constraint coverage, dynamic-shape-aware where it matters).

Status: DRAFT for review. Decisions marked **[SETTLED]** were agreed in
session; items marked **[OPEN]** need a call before the relevant phase.

---

## 0. Why we are migrating

The current corpus was captured before the capture infra had invariants.
What's wrong with it, concretely:

| Defect | Evidence | Consequence |
|---|---|---|
| Partition outputs dropped escaping values | squeezenet E2E: 3.60x repro closure, 0.1% model delta (`squeezenet_scatter_e2e_validation.md`) | Repros under-constrain the computation; elimination-style gaps/closures don't compose to models |
| Zero-user mutating ops could be dropped | fixed in `77a691d80` | Side effects (copy_ for running stats / caches) silently deleted from repros |
| fp32 everywhere | corpus-wide | Not how models run; bf16 shifts kernels memory-bound → instruction-bound, changes fusion profitability and floors |
| Dynamic-origin regions silently dropped | `_lift_shape_arg` emits fx.Node names into literals → NameError → region excluded (`dynamic_shapes_capture_design.md`) | Corpus has ZERO steady-state kernels from dynamic models; static benchmarking of e.g. opacus GroupNorm measures a ~2.6x different kernel |
| reshape/view hash fork | 167 vs 887 repros; confirmed dup pairs each with own oracle | Same partition counted as two patterns; duplicate oracle effort |
| 404/440 model dirs manifest-only (no full graphs) | inventory | No end-to-end cross-check, no repartitioning, no accounting for those models |
| Sidecar constraint guesses laundered | 19 opacus graphs backfilled with default 32 (real dims 16/8/4) | Graphs instantiate at fabricated shapes |

All of the *infra* causes are now fixed and gated (see §1). The corpus
itself still embodies the old defects — hence recapture.

---

## 1. Preconditions — infra gates (ALL LANDED)

| Gate | What it guarantees | Where |
|---|---|---|
| Partition fidelity | Escaping values are partition outputs; mutating ops never dropped | `77a691d80`, tests in `scripts/test_partition_outputs.py` |
| Canonical hashing | `hash(G) == hash(retrace(G))` for reshape/view + clone spellings; no dup patterns minted | `6ee455f4a`, fixed-point test `scripts/test_canonical_hash.py` |
| Round-trip invariants | Every written artifact reload-verified at capture time; sidecar stamped `"roundtrip": "ok"/"failed: <reason>"` | `e392d4422` + validate-before-write gate; `validate_corpus_invariants.py --full-graph-roundtrip` |
| Shared partitioner | Accounting partitions ≡ capture partitions, provably | `260d8f1a0` (`get_fusion_partitions`) |
| GC transaction | Atomic flip with hard verification gate (§5) | `c374b6553` (`scripts/gc_corpus.py`) |
| Oracle dispatch | Floors declare (hardware, signature); bench self-reports fallback | `oracle_impl`, on main |

**[OPEN — in flight]** Dynamic-shapes capture fix (dropped-region bug, expr
serialization). NOT a gate for Wave 1 (static suites); IS the gate for
Wave 2 (vllm/genai/dynamic). Design + serialization settled (§3).

---

## 2. Wave structure **[SETTLED]**

**Wave 1 — static suites: timm, torchbench, hf.** Start as soon as the
invocation script exists. The standard benchmark suites run fixed shapes;
static capture is faithful for them. The dynamic fix is irrelevant here —
EXCEPT that the write-gate will stamp any wave-1 model that unexpectedly
hits the symbolic path, telling us exactly which models actually belong in
wave 2 (no guessing).

**Wave 2 — dynamic-relevant: vllm, genai, opacus + anything wave 1 flags.**
Gated on the dynamic-shapes capture fix. These get expr serialization,
symbol tables, `captured_dynamic` provenance — enabling the questions that
motivate them ("how much does this fusion matter at decode BS=1 vs BS=64").

Pilot: a handful of small timm models first to shake out the invocation
script end-to-end (capture → gate stamps → merge → manifests → spot-bench)
before the long grind.

---

## 3. Capture configuration

**Dtype [SETTLED]: AMP for training, bf16 for inference.** The corpus is
UPDATED to this — one corpus, not an fp32/bf16 fork. Implementation note
[SETTLED]: precision is part of the *invocation* (autocast context / model
.to(bf16) in the run script), NOT a capture-hook concern — capture hooks
whatever compilation happens.

**Distributed [SETTLED]:** anything needing distributed state (process
groups, collectives — the moco class) is skipped with a recorded reason in
the run log. Blanket rule in the invocation script.

**Dynamic shape serialization [SETTLED, wave 2]:** per-repro `shapes.json`
sidecar:
```json
{
  "symbols": {"s0": {"hint": 8, "range": [1, 256]}},
  "family":  [{"shape_exprs": ["s0", "s1", 1024], "dtype": "bf16"}],
  "points":  [{"label": "...", "bindings": {"s0": 1}, "source": "captured"}]
}
```
- Shared symbol table expresses cross-input dim sharing (the thing concrete
  snapshots can't).
- Capture records only OBSERVED points; ladder points (BS=8/16/64...) are
  benchmarking policy added later.
- shapes.txt is GENERATED from shapes.json — existing consumers
  (`load_shape_configs`, oracle signatures, dispatch) unchanged; formats
  are mechanically interconvertible, so this is not blocking in either
  direction. Static repros get a degenerate shapes.json.
- Oracles/dispatch stay EXACT-SHAPE (settled oracle_impl design); families
  generate points, never fuzzy-match.
- Models that compile dynamically should also be *benchmarked* with
  dynamic compilation (`captured_dynamic` → bench honors it) — static
  benchmarking of dynamic models measures the wrong kernel.

**Hardware/run logistics [OPEN]:** which box runs the long grind (this one
monopolizes both B200s for ~days; pauses gap-closing work) vs the other
server. Suite order within wave 1 (suggest: timm pilot → timm → torchbench
→ hf).

---

## 4. What recapture produces (additive — nothing breaks)

Per model: `full_graph_NNN.py` + meta sidecar (constraints, roundtrip
stamp, provenance) + manifest.json (pattern list). Per new pattern: a
canonical repro dir (repro.py, meta.json, shapes.json→shapes.txt).

Key properties:
- **New pattern hashes throughout** — bf16 dtypes + fidelity-fixed outputs
  change content hashes, so new repros land ALONGSIDE old with zero
  collisions. Patterns that genuinely survive (dtype-generic, output-set
  unchanged) re-match by content hash and KEEP their existing repro dir +
  oracle automatically.
- Old corpus (repros, oracles, manifests, results) fully functional
  throughout. The old corpus remains valid for kernel-level work during
  the entire migration.
- Every artifact self-validated at write time (roundtrip stamp); a failed
  capture is visible immediately, not at benchmark time months later.
- `gc_corpus.py diff <model> <new_patterns>` prints the per-model
  transaction preview: +new (needs oracle), -dropped (orphan check),
  unchanged.

---

## 5. Oracle migration **[user invariant: never a state with repros and no oracles]**

Sequencing per pattern:
1. New pattern captured → repro exists, no oracle yet → it is NOT yet a
   "successor" for anything; old pattern + old oracle remain authoritative.
2. New pattern benchmarked (compile times, gap vs SOL proxy).
3. Oracle written/translated for the new pattern (oracle server; the
   `oracle_impl` registration declares hardware + signature from day one).
   Translation cases:
   - Content-hash survivor → oracle carries over untouched (it IS the same
     pattern).
   - Same computation, new dtype/outputs → port the kernel, re-validate
     numerics (bf16 tolerances), re-tune configs (bf16 shifts the roofline
     — instruction-bound vs memory-bound; floors need re-validation, not
     just dtype-swapped configs).
   - Genuinely new pattern → fresh oracle, normal queue.
4. ONLY THEN is old→new eligible for the flip (§6). The GC `prepare` gate
   enforces this mechanically: a successor without an oracle or without
   benchmark results hard-fails the plan.

The old oracles are deleted (well — quarantined) only in §6, after their
successors are covered. At no point does a pattern lose oracle coverage.

---

## 6. The atomic flip **[SETTLED: GC last, single revertible commit]**

When a model's (or suite's) new patterns are benchmarked + oracle-covered:

1. Build plan: `{"supersedes": {old_hash: new_hash | null}, "waivers": ...,
   "benchmark_results": ..., "oracle_status": ...}`
2. `gc_corpus.py prepare --plan` — HARD-FAILS unless every superseded
   pattern has a benchmarked + oracle-covered successor or explicit waiver,
   and no manifest still references an old hash the plan doesn't cover.
   You cannot produce an applicable plan from a half-finished migration.
3. `gc_corpus.py apply --plan` — stamps `"quarantined": {date, reason}`
   into superseded repros' meta.json. NEVER deletes (hand-written oracles
   are expensive; content-addressed patterns recur). Prints exact file list.
4. ONE git commit of that file list + updated manifests. Revert = full
   restore. The corpus is never half-migrated.

Flips can be per-suite (timm flip, then torchbench flip...) — each is its
own atomic commit with its own verified plan. **[OPEN]** flip granularity:
per-suite vs one corpus-wide flip at the end.

Pre-existing debt handled in the same machinery: the 48 current orphans
(47 with oracles) and the ≤167 reshape/view dup pairs just become entries
in a plan (supersedes/waivers) whenever we choose to clean them — no
special path.

---

## 7. After migration

- Fresh oracle sweep on the new corpus → new gap table (expect honest
  shifts: some old "wins" deflate per §0 row 1; whatever survives is real).
- Model accounting (shared partitioner + occurrence counts) over new full
  graphs → "for model X, which kernel fixes get within N% of
  demonstrated-best" with trustworthy composition.
- SOL column (roofline: max(bytes/BW, flops/peak)) as the second target
  alongside oracle floors — both for repros and gemm/conv. (Backlog.)
- Wave 2 dynamic families → the BS=1 vs BS=N fusion-impact questions.

---

## 8. Worklist inputs (already collected)

- 404 manifest-only model dirs (suite breakdown: 242 torchbench, 150 hf,
  12 genai) — need graphs regardless of anything else.
- Sweep v2 failure taxonomy (partial, 145 measured + 215 failures
  categorized; sweep ongoing) — `results/full_graph_sweep_v2_b200.json`.
- 19 opacus sidecars with laundered shape guesses (wave 2).
- 161 graph-dirs without manifests (invariant C unexercisable) — fixed
  automatically by recapture writing manifests.
- BERT_pytorch-class corpus gaps (partitions with no canonical repro,
  dropped by old eager-validation failures) — recapture recovers them.

---

## 9. Open questions (decisions needed, by phase)

| # | Question | Needed by | Suggestion |
|---|---|---|---|
| 1 | Run location: this box vs other server | Wave 1 start | Other server if available; this box otherwise with gap-work paused |
| 2 | Suite order | Wave 1 start | timm pilot → timm → torchbench → hf |
| 3 | Old-corpus work during migration | Wave 1 start | Keep using for kernel-level work; it stays valid until flip |
| 4 | Flip granularity: per-suite vs corpus-wide | First flip | Per-suite — smaller blast radius, earlier feedback |
| 5 | Oracle translation ownership: which floors port mechanically vs re-written | Phase §5 | Oracle server decides per pattern; dispatch registration makes either a 2-line landing |
| 6 | Ladder policy for wave-2 families (which BS/seq points) | Wave 2 bench | Per-family policy doc; decode ladder for vllm (1,2,4,...,64), seq ladder for LMs |
