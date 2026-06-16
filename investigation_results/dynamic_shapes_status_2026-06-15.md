# Dynamic-shape capture: status, references, open design questions

Date: 2026-06-15. Branch: `dynamic-shapes-capture` (off
`investigations-june-2026`, 12 commits, 69 tests in
`tests/test_canonical_invariants.py`, all green; 2 GPU-gated). Companion
design doc: `investigation_results/dynamic_shapes_capture_design.md`
(§2.1–2.7 + the 2026-06-15 updates §2.5b / upstream-bug note).

This is the "what we did / what's left" map. The big-picture goal: the
corpus must be able to capture and faithfully replay the kernels a model
runs under DYNAMIC shapes (vLLM-style bucketing, variable seq-len), not
just the cold-start static kernel — which is a measurably different kernel
(design §1.4: GroupNorm static 13.7us/1 persistent kernel vs dynamic
35.5us/2 looped kernels, a 2.6x gap our static-only corpus was blind to).

---

## 1. What was broken (baseline)

Per the June-10 audit (design §1): the capture pipeline collapsed every
SymInt to its `.node.hint`, regex-parsed shape annotations
(`_parse_intish("64*s0*s53") -> 64`, design §1.2), and — worst — every
region extracted from a dynamic compilation FAILED eager validation and was
dropped. Net: the corpus contained zero dynamic-shape repros.

Two concrete bugs anchored it:
- `make_inputs_from_config` had no `symint` branch → a live symint input
  (`['sym', h]`) fell through to `spec['shape']` → `KeyError` → region
  dropped (the silent-drop, design §1.1 bug F).
- `_lift_shape_arg` stored the reshape shape list raw; under dynamic shapes
  it held `fx.Node`s → generated `[64,32,2,mul]` as source → `NameError` at
  exec, OR a frozen `List[int]` that defeats reuse.

---

## 2. What we built (faithful capture → merge → load → run)

Principle that governs all of it (the user's repeated instruction): **we
have the live SymNode at capture, so store its exact sympy expression and
evaluate with sympy when a concrete value is needed — never regex an int
out of rendered text, never collapse a stride to a hint, share one utility
across capture paths.**

### 2.1 Shared SymNode→expr extractor (one source of truth)
`full_graph_harness.py` — imported by `capture_hook.py` (verified same
object by `test_capture_hook_reexports_shared_extractor`):
- `sym_expr_str(x)` :110 — exact sympy string for a live SymInt/SymFloat
  (`'s0'`, `'64*s0*s53'`), None for plain/constant ints.
- `symbolic_block_from_value(value)` :136 — per-tensor
  `{shape_exprs, stride_exprs}` (per-slot expr or None).
- `harvest_shape_env(shape_env)` :166 — graph-level
  `{symbols:{name:{hint,range[,unbacked,hint_source]}}, guards,
  captured_dynamic}`. Backed symbols from `backed_var_to_val`; UNBACKED
  symbols (data-dependent `nonzero`/`item`/…) harvested too, with
  `hint_source` tier (`observed` real-runtime > `size_hint` derived >
  `range_fallback` placeholder) so a consumer knows how much to trust the
  hint. Specialized symbols (range `[k,k]`) dropped. Guards filtered to
  those over kept symbols. Finite range upper bounds survive; `int_oo`→None.
- `shape_env_from_gm(gm)` :252 — finds the live ShapeEnv in a graph.

### 2.2 Expr-preserving capture (capture_hook.py)
- `_record_placeholder` :347 — records hint shape/stride AS BEFORE, plus an
  additive `symbolic` block from the shared extractor; live symint inputs
  get `expr` recorded. Static placeholders are byte-identical (no block).
- `_lift_shape_arg` :477 + `_entry_is_symbolic_node` :468 — **static** shape
  lists lift to `_shape_param_N` exactly as before; a shape list containing
  **symbolic** dims is kept INLINE (`reshape(x,[64,32,2,mul])`) referencing
  the lifted symint input nodes (already in scope as forward args). This is
  the form the model's graph actually had; our old lift had replaced it
  with a frozen constant list that re-specializes per binding (§4 below).

### 2.3 Codec is the single serialization boundary (input_codec.py)
- `compact_from_spec` :111 / `spec_from_compact` :182 — overlay per-slot
  exprs into shape/stride slots and reconstruct them; `symint` with an expr
  ⇄ `['I', hint, expr]`, constant ⇄ `['sym', hint]`. Round-trip lossless
  (`test_symbolic_stride_codec_roundtrip_is_exprs_not_hints`,
  `test_symint_input_expr_codec_roundtrips_as_I`).
- `evaluate_symbolic_entry` :362 / `evaluate_spec` :389 — evaluate a
  (compact entry | verbose spec) at a binding via sympy (`_eval_dim`).
- `validate_bindings` :339 — range + guard check, LOUD on violation.
- `instantiate_point` :416 — materialize a point at any binding.

### 2.4 Consume + merge
- `repro_harness.load_shape_configs(symbol_bindings=)` :51 → instantiates
  symbolic points at a binding; `make_inputs_from_config` :468 has the
  symint/scalar branches (silent-drop fix).
- `merge_captures._write_shapes_json` :75 — writes symbols/guards
  GRAPH-LEVEL and per-point `bindings`/`captured_dynamic`; static captures
  get none of these (`test_merge_static_point_has_no_dynamic_fields`).

### 2.5 The serialized artifact (real example, GroupNorm/opacus var_mean)
```json
{
  "symbols": {"s53": {"hint": 16, "range": [2, null]},
              "s0":  {"hint": 16, "range": [2, null]}},
  "guards":  ["Eq(16380*s0*s53 + 4*s0*(s53 - 1) + 4*s0, 4194304)"],
  "points": [{
    "shape_hash": "2e02cdda", "captured_dynamic": true,
    "bindings": {"s53": 16, "s0": 16},
    "inputs": [
      [[64,64,"s53","s0"],"f32",{"st":["64*s0*s53","s0*s53","s0",1]}],
      ["I",256,"s0*s53"], ["I",16,"s53"], ["I",16,"s0"],
      [[64],"f32"], [[64],"f32"],
      ["S",[64,32,2,"s0*s53"]], ["S",[64,64,"s53","s0"]]
    ]}]
}
```
`["I",hint,expr]` = live symint input; `"64*s0*s53"` = stride expr;
`["S",[...,"s0*s53"]]` = shape param with coupled product preserved.

### 2.6 Verified end to end (GPU)
- `test_dynamic_capture_merge_load_roundtrip_gpu`: capture → 0 dropped →
  merge → `load_shape_configs` at hint (16×16) runs, guard-RESPECTING
  rebind (8×32, product preserved) runs, guard-VIOLATING (24×24, product
  576) loudly rejected. No hand-built shapes.json.
- The captured guard `Eq(...)` factors to `16384*(s0*s53 - 256)` i.e.
  `s0*s53==256`; it re-parses through sympy and gates correctly (manual
  proof in conversation).
- `test_static_capture_has_no_symbolic_artifacts_gpu`: a static compile
  captures with zero symbolic content (additive-path pin).

---

## 3. The compile question — RESOLVED in principle, not yet wired

The point of capturing dynamic shapes is to BENCH the dynamic kernel
faithfully. We investigated how to compile the captured repro so it has the
model's real dynamic structure.

- `mark_dynamic` alone is INSUFFICIENT (design §2.5b): it re-derives an
  approximation — dynamic dims + ranges, but guards are freshly re-traced
  and symbols re-allocated, so coupled constraints (`s0*s53==256`) are NOT
  restored. The `--dynamic` bench path today uses `mark_dynamic` on the
  recorded dims (`repro_harness.dynamic_dims_for_repro` :309) — a strict
  improvement over blanket `dynamic=True`, but it still recompiles per
  binding for the lifted-symint repro form.
- The FAITHFUL mechanism is `torch.compile(mod, shapes_spec=ShapesSpec(...))`
  — **this is the user's own PR stack** (pytorch:
  `[ShapesSpec] Support derived expressions as leaf specs… (#185154)`,
  `Support ShapesSpec.assumptions for cross-input shape invariants (#185161)`,
  `unify variables with same shape var (#184853)`, STATIC sentinel #185165,
  args/*args/**kwargs #184129; groundwork PR #187160). Our capture maps 1:1:
  | shapes.json | ShapesSpec |
  |---|---|
  | symbols + ranges | `ShapeVar(name, min=, max=)`, ONE object per symbol |
  | per-dim/stride exprs | `TensorSpec([64,64,s53,s0])`; derived `s0*s53` |
  | coupling / sharing | reuse the SAME ShapeVar object across slots |
  | live symints `['I',..]` | `IntVar(...)` params (bare slots bind symbols) |
  | guards | `assumptions=[s0*s53 == 256]` |
  | hint | `optimization_hint=` |
- VERIFIED: driving the ACTUAL captured repro (post inline-reshape change)
  through `torch.compile(shapes_spec=...)` reuses ONE dynamic graph across
  the coupled 16×16→32×8 rebind. Before the inline change, the constant-list
  `_shape_param` made this impossible (1-vs-2-graphs spikes, design §2.5b).

So §2.2's inline-reshape change is the necessary enabler, and ShapesSpec is
proven sufficient. What remains is the adapter + wiring.

---

## 4. Why lifting symbolic shapes as LIST params is the crux (settled)

The user wanted to KEEP lifting (it aids repro reuse). Spikes settled the
exact boundary:
- Lift the SYMINTS as scalar args — KEEP (they're the bare `IntVar` slots
  that bind the symbols). Already captured.
- A shape lifted as a standalone constant-LIST arg
  (`_shape_param=[64,32,2,256]`) re-specializes per binding regardless of
  symbol binding (`ParamsSpec` has no list-element spec; the list literal is
  a leaf). So for symbolic shapes, the reshape target is spelled INLINE from
  the symint args — same reuse, and the only ShapesSpec-compatible form.

---

## 5. Open design questions (need user input)

1. **shapes.json ↔ ShapesSpec connection** (the paused question). Spec
   types have `to_jsonable` but NO `from_jsonable`. Options:
   (a) reuse a JSON→ShapesSpec loader if the PR stack has one;
   (b) write `shapes_spec_from_shapes_json()` adapter on our side
   (ShapeVar/IntVar/TensorSpec/assumptions from our symbols/exprs/guards);
   (c) add `from_jsonable` to ShapesSpec upstream (canonical round-trip).
   This decides how the bench builds the spec. NOT YET DECIDED.

2. **Oracle ↔ dynamic family dispatch** (explicitly deferred). Oracle
   `oracle_impl(point=<shape_hash>)` dispatches by EXACT concrete shape
   tuple, but a dynamic point is a family (one shape_hash, many bindings).
   Options: (a) shape_hash-keyed dispatch + a shape-general dynamic oracle
   kernel; (b) per-binding registration (N per family); (c) both. Needs the
   "how do oracles relate to families" decision before building.

3. **Upstream telemetry bug** (filed, design §2.5b). `shapes_spec=` triggers
   a non-fatal `SymInt is not JSON serializable` traceback every compile
   from `_get_dynamo_config_for_logging` (caught by pytorch's telemetry
   guard — NOT us). Fix: add `_shapes_spec` to that blocklist upstream, or
   filter the log line our side. File upstream when the bench path lands.

4. **Unbacked-symbol bench treatment** (design §2.5b / hint_source). An
   `unbacked` symbol's hint is a data-dependent fallback, not an observed
   shape — a `range_fallback` point should be swept-or-excluded from the
   static-vs-dynamic gap table (no static baseline exists). Rare in fusion
   regions (data-dependent ops are extern); flagged, not yet special-cased
   in accounting.

5. **Dynamic-vs-static IDENTITY** (adversarial review round 1, Finding E —
   see `finding_e_identity_analysis.md`). MEASURED: a dynamic capture does
   NOT currently dedup to the static capture of the same hint shapes — both
   shape_hash (symint input placeholders inflate it; excluding them makes it
   match, verified, and changes ZERO existing corpus hashes) and pattern_hash
   (inline-reshape vs lifted `_shape_param`; dynamic subgraphs skip
   canonicalization) fork. Plus symbol allocation is trace-context-sensitive,
   so re-captures scatter. DECISION: Option 1 (make them dedup — exclude
   symints from shape_hash + canonicalize dynamic subgraphs) vs Option 2
   (distinct entries, join static-vs-dynamic at the accounting layer via
   origin_ops+models, retract the §E dedup claim). Recommend Option 2 + the
   safe shape_hash change. Not yet implemented — needs the call.

---

## 6. Remaining implementation (once #5.1 is decided)

- `shapes_spec_from_shapes_json()` builder (form depends on #5.1).
- Wire `repro_harness` `--dynamic` to use `shapes_spec=` instead of
  `mark_dynamic`; keep both numbers (static-specialized vs dynamic-artifact).
- Bench-side ladder generator (`wave2_ladder_policy.md`) — turn a captured
  family into swept points; separate concern from the compile wiring.

Corpus-population follow-ups (separate from capture machinery): opacus
recapture (design §2.6), restore the untrusted-sidecar skip for
annotation-backfilled symints (§2.5 last bullet).

---

## 7. Commit trail (branch `dynamic-shapes-capture`)
```
5c660b2 design: file upstream ShapesSpec telemetry bug
152a5fd dynamic capture: keep symbolic reshape targets INLINE, don't lift to list param
6efe1b6 design: keep lifting — build reshape target from symint args
4a0979b design: ShapesSpec is the faithful re-creation; lifted shape-params block it
6242129 bench --dynamic: mark recorded dims, not blanket dynamic=True
5f2edb1 tests: promote the GPU end-to-end verifications into the suite
e64ca99 unbacked symbols: record hint_source tier (observed/size_hint/range_fallback)
31e39df dynamic capture: symint-input expr round-trips; harvest unbacked symbols
69121a6 merge: write dynamic symbols/guards/bindings into canonical shapes.json
81aa380 dynamic capture: strides/dims are EXPRESSIONS end to end; kill the parse path
bc4de07 dynamic capture: loop-close verified + harvest unit tests + static regression
3084660 WIP dynamic capture: ShapeEnv harvest + expr-preserving placeholders/lift
```
Plus the symint make_inputs fix on `investigations-june-2026` (cherry-pick
`5f2b45b`, the silent-drop unblock).
