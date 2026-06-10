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

**Wave 0 — single-model pipeline validation (GATES WAVE 1).** Before any
mass run, prove the ENTIRE value chain on ONE model (small timm CNN), not
just that capture executes. Acceptance criteria, all must pass:

1. **Capture**: invocation script runs the model (bf16/AMP); full graph +
   sidecars + manifest + canonical repros written; every artifact stamped
   `roundtrip: ok` by the write-gate.
2. **Reference counts**: `gc_corpus.py diff` shows a sane transaction
   preview (new patterns / content-hash survivors / dropped); the manifest
   references exactly the repro dirs that exist; `refcount` reports no
   dangling references.
3. **Accounting maps**: `model_graph_accounting.py` partitions the new
   full graph (shared partitioner), maps EVERY partition to a captured
   repro — zero UNMATCHED — with occurrence counts.
4. **Times roll up**: benchmark the new repros; verify
   Σ(partition_time × occurrences) + non-fusible ops ≈ end-to-end
   full-graph time within tolerance (the A-vs-B cross-check that catches
   composition bugs — exactly what exposed squeezenet).
5. **Improvement estimates produce**: with oracles on at least 2-3 of the
   model's patterns (ported or fresh), the accounting emits the target
   sentence: "fixing pattern X saves Y us × N occurrences = Z% of this
   model".

If any link fails on one model it fails on 440 — wave 0 is where it's
cheap to find out. Only a fully green wave 0 unlocks wave 1.

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

**Observed-value stats [SETTLED 2026-06-10]:** capture records per
integer/bool input the OBSERVED stats from the real execution —
`{"observed": {"min": ..., "max": ..., "n_unique": ...}}` in the input
spec / shapes.json gen block. Purpose: input to INFERENCE, not just
validation. Bound hierarchy:
  1. graph inference (known consumer patterns: embedding→vocab, gather→dim)
     — wins where available, gives the semantic bound;
  2. observed-value fallback (`high = observed.max + 1`,
     `source: "observed"`) — the universal fallback covering the unbounded
     tail of ops nobody pattern-matched (e.g. int8 maxpool offsets in
     [0,9) — not a mask, not a vocab index; no pattern matcher knew it);
  3. default/guess: ELIMINATED for new captures. The high=512-style guess
     class (device asserts, opacus laundering) becomes structurally
     impossible — observation is always available at capture time.
`n_unique` additionally disambiguates generator KIND (permutation:
n_unique==numel; binary mask in int clothing: n_unique==2; dense vs sparse
index). Raw stats stored unembellished; generation uses max+1 exactly.
observed-sourced bounds are conservative-safe (never OOB) but may
understate the semantic bound — fine for validity, graph inference still
preferred for realism. Full-value serialization (tier 2) only for: small
integer/bool tensors, `unverifiable` inference, or the existing
requires_exact class. Floats never serialized beyond existing tiny-constant
exact path.

**Constraint round-trip invariants (extend the A/B/C partition set):**
- **D — derivation**: re-running inference on the reloaded ARTIFACT
  reproduces the serialized bounds (or the input is stamped
  `unverifiable` — a guess cannot survive D, killing the laundering
  failure mode).
- **E — generation validity**: inputs generated from serialized
  constraints execute eager + compiled without device asserts.
- **F — tightness**: edge values (high-1; true permutations) are valid;
  where independent ground truth exists (model config vocab_size, gather
  target dims, observed stats), serialized bound is consistent with it.
D/E run in the write-gate (per-input `constraints: verified |
unverifiable(reason)` stamps alongside roundtrip:ok) and in
`--full-graph-roundtrip`; F where ground truth exists.

**Shape serialization [SETTLED]: `shapes.json` is the PRIMARY per-repro
format from wave 1 onward** (decision 2026-06-10: JSON is easier to reason
about and augment over time — model linkage, generations, occurrence
counts, and the entire dynamic-shape structure are real keys instead of
facts smuggled into txt label strings). One schema serves static and
dynamic; static repros simply omit the symbolic keys:

```json
{
  "symbols": {"s0": {"hint": 8, "range": [1, 256]}},
  "family":  [{"shape_exprs": ["s0", "s1", 1024], "dtype": "bf16"}],
  "points": [
    {"shape_hash": "75902420",
     "signature": "(T([320], bf16), T([128, 320, 7, 7], bf16, stride=(15680, 1, 2240, 320)), ...)",
     "bindings": {"s0": 1},
     "models": {"timm/infer/mobilenetv2_100": {"occurrences": 7, "generation": "2026-06-bf16"}},
     "source": "captured"}
  ]
}
```

- `signature` keeps the existing T()/S() string WITHIN the JSON — compact,
  human-readable, existing parser.
- **Oracle dispatch keys on the point hash [SETTLED]**:
  `oracle_impl(hardware="B200", point="75902420")` (or a list of points,
  with shared-body kwargs). Dispatch computes the input-derivable shape
  hash of the runtime inputs and matches by hash equality — no signature
  parsing or structural comparison. `shapes=` (verbose signature string)
  remains as fallback for standalone oracles. Unknown hash → loud
  NO_ORACLE_FOR_SHAPE listing registered points with their readable
  signatures from shapes.json. This also makes the GC prepare gate's
  oracle-coverage check mechanical: coverage is per point hash.
- `models` solves the provenance problem the txt labels smuggled
  (`<model>_<hash>:`), and carries occurrences + generation per model —
  the keys the migration/accounting need and txt could not hold.
- `symbols`/`family`/`bindings` (wave 2): shared symbol table expresses
  cross-input dim sharing; capture records OBSERVED points; ladder points
  (BS=8/16/64...) are benchmarking policy, `"source": "ladder"`.
- shapes.txt becomes DERIVED compat output (generated from shapes.json)
  until consumers (`load_shape_configs`, runners) read JSON directly; then
  dropped. Formats are mechanically interconvertible — no migration cliff.
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

---

## Appendix: Wave 0 findings (2026-06-10, mobilenetv2_100 bf16)

Wave 0 ran GREEN — full chain validated (capture/stamps → refcounts →
zero-UNMATCHED accounting → roll-up → per-fusion improvement table).
Found and FIXED before wave 1: write-gate TracingContext bug (every
sidecar would have stamped roundtrip:failed; fix `1a17b643f`).

Known properties to carry into wave-1 interpretation:
1. **Many-tiny-kernel models are a structural limit of standalone repro
   benchmarking, not a measurement artifact.** For models like
   mobilenetv2 (52 fusible launches of 7–23us), the real optimization
   lever is horizontal fusion / combo kernels / launch scheduling — which
   is INVISIBLE to a standalone repro by construction: a repro is one
   partition; horizontal-fusion profit exists only among sibling kernels
   in the model. Per-kernel headroom numbers miss the lever entirely.
   DECISION: benchmarking methodology stays untouched (pure CUDAGraph, no
   profiler mixing, no correction factors). For launch-granularity-bound
   models, the accounting should CLASSIFY them as such and route the
   question to the full-model runner (e2e with/without combo_kernels
   etc.) rather than emit per-pattern headroom. Launch-overhead
   accounting, if ever needed, lives in the analysis layer.
2. **Survivor floors don't carry across dtype/hardware**: ported
   BN-affine oracle (H100/f16 → B200/bf16) showed Inductor at/below the
   old floor (0.78–1.00x) — apparent headroom deflated to ~0. Floors are
   re-validated, never dtype-swapped (§5 confirmed empirically).
3. **Exact-shape dispatch refusal works**: survivor oracles correctly
   reported NO_ORACLE_FOR_SHAPE at the new capture point instead of
   emitting soft floors.
4. Soft-invariant candidate for capture: infer non-negativity for
   variance-position inputs (randn currently yields NaN through
   sqrt(var); harmless for timing, numerics checks must be NaN-aware).
