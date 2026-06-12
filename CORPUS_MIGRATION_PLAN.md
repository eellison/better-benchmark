# Corpus Migration Plan — Recapture, Oracle Migration, Atomic Flip

Standing reference for migrating the corpus from its current state
(fp32 captures, pre-fidelity-fix partitions, 404 models without graphs)
to the new state (AMP-train/bf16-infer, fidelity-correct partitions,
full graph + constraint coverage, dynamic-shape-aware where it matters).

Status: DRAFT for review. Decisions marked **[SETTLED]** were agreed in
session; items marked **[OPEN]** need a call before the relevant phase.

---

## STATUS 2026-06-11 — Wave 1 READY, awaiting two calls

All infra gates landed, adversarially reviewed (4 agent rounds + opus
verification + per-fix live captures), and dress-rehearsed end to end
(10 timm models: capture 173s → all hard invariants → three-leg identity
exact → occurrence counts match offline retrace → attribution coherent
0.99–1.08 corrected). Hardening landed since the §1 table was written:

- **Hash identity**: dtype/str/memory_format args, schema-typed structural
  ints (int/int? dims, both invocation forms), canonical node order
  (deterministic Kahn) + sorted outputs, stride in shape_hash. 64/74
  batch hashes changed — free now, a migration later.
- **Pipeline**: drops FAIL HARD (no partial model enters the corpus);
  atomic writes everywhere in merge; run_log flock + per-worker GPU env;
  validator `--corpus-root` (wave trees validatable pre-flip); exact
  occurrence counts in shapes.json (accounting joins from artifacts, no
  retrace).
- **Format**: one compact structured input encoding (`input_codec.py`)
  shared by model sidecars + shapes.json points (data-only; T()-string
  rendered on demand); repro v3 (default inputs = first shapes.json
  point, no inline config); signature flows as data capture→index→merge
  (regex extraction legacy-only).
- **Attribution**: e2e ≈ Σ(standalone×occ) − G×(occ − n_graphs) validated
  on 10 models; extern bench reuses harness input generation (inferred
  index bounds); deit in-context layernorm finding filed
  (`investigation_results/deit_incontext_layernorm_slowdown.md`).
- **Tests**: `tests/test_canonical_invariants.py` — 30 invariants
  (idempotence incl. lifted-params/reductions/mutations, Kahn tie-break
  ambiguity, hash semantics, extern signatures, node accounting), ~3s,
  CPU-only.

**The two calls that unlock launch**: (1) run location — this box
(dress-rehearsal-proven, ~2.5–3h for ~440 timm on 2 GPUs) vs other
server; (2) suite order — proposed timm infer+train → torchbench → hf
(timm = validated path; torchbench = biggest manifest-only hole, 242
dirs; hf last = newest extern paths + natural experiment for the deit
finding).

Pre-launch sweep (no decisions needed): qualify model names in canonical
meta.json (`timm/infer/resnet18`, not `resnet18` — free now, dirs are
re-minted); suppress __pycache__ in validation imports.

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
| Canonical hashing | **canonicalize-before-serialization**: one make_fx retrace BEFORE the artifact is written, so the serialized form IS canonical and every later trace is the identity (idempotent: retrace(serialized)==serialized). Replaces the alias-table approach (`_HASH_OP_SPELLING_ALIASES` was a code smell — 167 reshape/view + 79 dead-getitem repros prove the drift surface). One re-trace allowed; all subsequent traces and hashes identical. | `6ee455f4a` (alias-table v1) → upgrade in flight per 2026-06-10 decision; fixed-point test becomes the single property retrace∘serialize == serialize |
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

### Oracle file slimming (filed 2026-06-11, apply during the §5 rewrite)

The current oracle files carry heavy boilerplate (example:
`pointwise_60c504726d51/oracle_whisper_layout.py`, 264 lines for one
kernel). Since most oracles get rewritten/ported during migration anyway,
the new-format oracle should drop ALL of this:

- **No input validation** (`_validate_inputs`, `_shape_tuple` rank/dtype/
  contiguity checks): IMPLIED BY THE REGISTRATION ITSELF. The
  `@oracle_impl(shapes="(T([8,6,1500,64], f32), S([12000,384]))")`
  decorator already declares rank, dtype, shape, and layout — per-oracle
  checks re-assert the same contract, weaker and driftably. If checking
  is wanted at all, it is ONE generic harness-side assertion (inputs
  match the registered signature), written once, never per oracle.
- **No `_torch_full_scope` reference reimplementation**: the repro IS the
  reference; `check_oracle` already compares against `Repro()(*inputs)`.
  A second hand-written aten spelling of the same scope is drift waiting
  to happen.
- **No CLI entry point** (argparse `main()`, ~80 lines/file): one shared
  runner (`python -m oracle_harness run <repro_id>`) replaces every
  per-file CLI. Not factored today — each file re-declares the same
  parser.
- **No path manipulation** (`REPRO_DIR = Path(__file__).resolve().parent`
  etc.): the package is installed; the harness resolves repro dirs by
  registration key, never by file location.
- **No `if triton is not None:` guards**: triton is a hard dependency of
  the whole project; an oracle that can't import triton should fail
  loudly, not silently fall back to the eager scope (which silently turns
  a floor measurement into an eager measurement).

Target shape of a migrated oracle file: docstring (gap diagnosis — keep,
it's valuable), kernel(s), `@oracle_impl`-decorated forward, nothing else.
Harness owns: inputs, reference comparison, timing, CLI, dispatch.

**Standard, enumerable registration (filed 2026-06-11):** make the
registry itself the mass-bench surface. Each repro's oracle registers
under a standard key ((pattern_hash, shape_hash) point — matching the
oracle-dispatch-by-hash decision) and the registry is ENUMERABLE without
importing every file by convention (one known filename per canonical dir,
e.g. `oracle.py`, discovered by walking `repros/canonical/`). Then mass
benchmarking is one command — `bench_oracles --all` enumerates the
registry, joins each registration to its repro inputs + shapes.json
points, and reuses bench_parallel for multi-GPU fan-out. No per-file
CLIs, no hand-maintained oracle lists, coverage = registry keys vs corpus
points (the same join that reports which patterns still lack oracles).
Design during §5 oracle migration, not before.

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

**Noted refinement (not pre-launch):** shape-param lifting currently keys
on the specific view-like op set (reshape/view/expand) with the shape at
args[1] — matching the original inline lift exactly. A schema-typed
discriminator (lift any argument occupying a `SymInt[]`-typed parameter
slot per the op's own schema) is arguably more principled (no op list to
maintain, can't mis-lift int[] dims/permute args) — swappable inside
`lift_shape_params` whenever broader shape coverage is wanted.

## 7. After migration

- Fresh oracle sweep on the new corpus → new gap table (expect honest
  shifts: some old "wins" deflate per §0 row 1; whatever survives is real).
- Model accounting (shared partitioner + occurrence counts) over new full
  graphs → "for model X, which kernel fixes get within N% of
  demonstrated-best" with trustworthy composition.
- SOL column (roofline: max(bytes/BW, flops/peak)) as the second target
  alongside oracle floors — both for repros and gemm/conv. (Backlog.)
  Inspiration target: NVIDIA's sol-execbench
  (https://github.com/nvidia/sol-execbench) — speed-of-light execution
  benchmarking methodology; worth mining for roofline modeling, per-kernel
  SOL accounting conventions, and how they structure SOL-vs-achieved
  reporting when we build our SOL column.
- Wave 2 dynamic families → the BS=1 vs BS=N fusion-impact questions.
- **Special-value numerics validation (filed 2026-06-12)**: the oracle
  check + fp64-anchored gate run on well-behaved synthetic inputs, so they
  can't catch oracles that mishandle special values — NaN propagation
  (max/min tricks that drop NaNs, fast-math that flushes them), ±inf
  (softmax/logsumexp shift tricks), -0.0, denormals. Add an opt-in
  adversarial pass to check_oracle_all_shapes: salt float inputs with
  NaN/±inf/-0.0 at a few positions (respecting gen kinds — never salt
  index/offsets/permutation inputs) and require bitwise-class agreement
  with eager (NaN positions match; inf signs match). Eager is the
  semantics reference, same as today. Needs care for repros where eager
  itself is special-value-undefined (e.g. uninitialized-read patterns);
  those get a recorded skip, not silence.
- **Model oracles (future)**: the natural generalization of repros — a
  hand-optimized END-TO-END implementation as the floor for a whole model
  (a model is structurally a big repro: graph + input specs + measured
  time; `oracle_impl(point=<graph shape hash>)` carries over unchanged).
  Closes the gap kernel floors structurally can't see: cross-kernel
  scheduling / horizontal fusion for launch-bound models. Accounting gains
  a third row — compile vs Σ(kernel oracles) vs model oracle — whose
  spread MEASURES the cross-kernel opportunity. Not until the migrated
  corpus + kernel floors + accounting are trustworthy.
- **LLM training-graph capture (filed 2026-06-12, HIGH VALUE — wave 2 or
  sooner)**: the 7 modern HF generation benchmarks (Llama-3.2, gemma-2/3,
  Qwen3, Mistral-7B, whisper-tiny, gpt-oss-20b) are inference-only in the
  upstream suite BY DECLARATION (ci_expected_accuracy training CSVs mark
  them `eager_fail_to_run` — their inputs carry no labels; runner
  loss=pred[0]=logits, backward structurally unsupported). Wave 1 captures
  their inference graphs and policy-skips train with this reason recorded.
  BUT these are the most modern models in the suite, and their TRAINING
  graphs (attention backward, vocab-size cross-entropy backward, RMSNorm
  backward at real LLM dims) are among the most valuable patterns the
  corpus could hold. The extension is small: for HF_LLM_MODELS in train
  mode, construct loss args ourselves (labels=input_ids — HF models
  compute their own LM loss when labels are present) and capture; mark
  provenance as our-construction (not upstream-blessed) so benchmark
  comparisons stay honest. Do when wave-2 planning lands, or earlier if
  LLM backward patterns become the accounting's coverage gap.
- **Custom generation constructors for op-specific input semantics
  (established 2026-06-11, extend as needed)**: when an op's inputs have
  validity constraints beyond a value range, the generation spec language
  grows a NEW KIND rather than approximating with bounds. Three seams,
  ~20 lines each, test-pinned: gen dict kind (full_graph_harness
  inference emits it), compact spelling (input_codec), constructor
  (repro_harness.make_inputs_from_config). Precedents: `constant`
  (maxpool window-center offsets — random offsets OOB at padded edges,
  device-side assert), `offsets` (embedding_bag: sorted non-decreasing),
  `permutation`, `index`. The inference side decides WHICH kind applies
  (consumer-op walk); the constructor owns HOW to build it. If an op
  needs richer semantics than a kind+args can express, the same seams
  accept a custom constructor keyed on the op.
- **Graph invocation counts in full-model accounting (filed 2026-06-12,
  deferred)**: the attribution identity currently assumes each captured
  full graph runs ONCE per model step — but a model can invoke the same
  compiled graph multiple times (loops over identical blocks, repeated
  calls into one dynamo artifact, generate-style decode steps). Where
  that happens, e2e ≈ Σ(graph contributions × invocation_count), and
  per-graph occurrence counts under-account. Detecting it needs a
  RUNTIME signal, not graph inspection — i.e. augmenting the compile/
  runtime wrapper to count invocations per compiled artifact during the
  capture run (a counter in the compiled fn's wrapper, recorded into the
  graph sidecar as invocations_per_step). Static suites mostly run each
  graph once per step so wave-1 numbers stand; revisit when accounting
  shows unexplained e2e > Σ(parts) gaps or when generate-style workloads
  (wave 2) enter the corpus.
- **Semantic tags on shapes/repros (filed 2026-06-11, not for wave 1)**:
  attach reusable semantic labels to specific points/repros — e.g.
  "split-k" (K-dominant gemm reduction shape), "decode" (BS=1 attn),
  "prefill", "imbalanced-reduction", "layout-copy" — that benching and
  analysis can key on ("bench all split-k shapes", "compare oracle wins
  on decode points vs prefill"). Natural home: an optional `tags: [...]`
  list per shapes.json point (and/or per canonical meta.json for
  pattern-level semantics), human/agent-curated, additive, schema already
  versioned so it lands without migration. Distinct from the structural
  metadata (shape/stride/gen) — these are MEANINGS, not facts derivable
  from the artifact alone.
- **Repro linter (future, post-migration)**: modeled on PyTorch's lintrunner
  setup — a dedicated lint config for `repros/` enforcing the corpus
  invariants statically (repro file structure, shapes.json schema,
  meta.json completeness, no stale `_repro_version`, header/class-name
  conventions), runnable as a single `lintrunner`-style command and
  wireable into the same pre-commit/CI hooks as
  `validate_corpus_invariants.py`. Separately, adopt the standard linter
  set (the non-repro-specific ones from PyTorch's config: flake8/ruff,
  format, spelling) for the NON-repro code (`scripts/`, `tests/`, harness
  modules) — repros are generated artifacts and get the corpus linter, not
  the style linter. Explicitly deferred until after the migration lands;
  noted here so it isn't lost.
- **Corpus/benchmark versioning for cross-change comparability (filed
  2026-06-11, add after migration)**: schema versions cover artifact
  FORMAT drift; this is about CONTENT drift — comparing benchmark results
  across corpus changes. Two distinct mechanisms, don't conflate:
  (a) *additive drift* (new shapes.json points, new models joining
  existing points) — cheap: a `generation` stamp per point (the §3 schema
  already reserves it) lets any results file say which corpus generation
  produced it. (b) *removals* — the hard case: once a model/pattern is
  GC'd, "re-run the old comparison set" has nothing to run, so
  comparability across removals cannot be a property of the corpus
  itself. It needs the benchmark SET as a first-class versioned artifact:
  a pinned list of (pattern_hash, shape_hash) points (extending
  `scripts/select_benchmark_set.py`), checked in, referenced by results
  files by name+version. Git history of `repros/` gives artifact
  reproducibility; the pinned set gives result comparability; results
  JSONs already stamp tool+commit+timestamp in `_metadata`. Design when
  the first real cross-generation comparison is needed, not before.

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
