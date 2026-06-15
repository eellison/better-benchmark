# Dynamic Shapes in the Capture Pipeline: Audit + Symbolic Serialization Design

Date: 2026-06-10. Status: audit complete, design proposed, NOT implemented.

## TL;DR

The capture pipeline collapses every SymInt to its `.node.hint` at capture
time and serializes concrete ints. Symbolic exprs, symbol-sharing between
placeholders, value ranges, and guards are all discarded. Worse, regions
extracted from dynamic-shape compilations **always fail eager validation and
are silently dropped** (the generated input literal references FX node names
like `mul` that don't exist at module scope), so the corpus contains *zero*
repros of the kernels a dynamic-shape model actually runs in steady state.
The one dynamic-vs-static experiment (below) shows this is not cosmetic: for
`var_mean_b1feb9d09685` the static repro measures **1 persistent kernel @
13.7us** while the model's real dynamic compilation runs **2 kernels
(looped reduction + pointwise) @ 35.5us** — a 2.6x different number.

---

## Part 1: Audit

### 1.1 `capture_hook.py` — every hint-collapse site

| Site | What happens | What's lost |
|---|---|---|
| `capture_hook.py:87-91` `_resolve_sym` | `SymInt/SymFloat -> x.node.hint` | the sympy expr (`s0`, `s0*s53`), free symbols, everything |
| `capture_hook.py:97-98` `_record_placeholder` | tensor shape/stride mapped through `_resolve_sym` per-dim | which dims were symbolic; that two placeholders share `s0`; symbolic strides (`64*s0*s53`) |
| `capture_hook.py:102-110` | SymInt *placeholder* recorded as `{"dtype": "symint", "hint": h}` | the expr (`Sym(s0*s53)` vs `Sym(s0)`) — a derived symint (product node) is indistinguishable from a root symbol |
| `capture_hook.py:300-312` | regex substitutes `s\d+` symbols in `print_readable` annotations with concrete dims | the repro source itself is rewritten to concrete; symbolic origin unrecoverable from the file |
| `capture_hook.py:377-381` | symint placeholder emitted as a plain int literal in inputs | family membership; only the snapshot point survives |
| `capture_hook.py:417` | symint serialized to `_shapes_config` as `S([hint])` | **conflation bug**: `S([...])` is the shape-param spec; `parse_shapes_config` (`repro_harness.py:183-184, 265-266`) returns a *list* `[hint]` where the graph needs an *int*. Any symint placeholder that survived to a repro would get the wrong python type |

Nothing in `capture_hook.py` ever touches `shape_env` — `var_to_range`,
`guards`, `var_to_val` are never read (verified: zero grep hits for
`shape_env|guards|ValueRange` across capture_hook.py, full_graph_harness.py,
repro_harness.py, merge_captures.py).

**Shape-param lifting breaks on symbolic shapes.** `_lift_shape_arg`
(`capture_hook.py:157-171`) lifts reshape/view shape literals into
`_shape_param_N` placeholders. Under dynamic shapes the shape list contains
`fx.Node`s (e.g. `[64, 32, 2, mul]` where `mul: Sym(s0*s53)`), and:

- `shape_params[param_name] = shape_list` stores the raw list *with Node
  objects in it*;
- the generated input line (`capture_hook.py:318`) renders as
  `[64, 32, 2, mul],  # _shape_param_0` — a NameError at exec;
- `_shapes_config` renders `S([64, 32, 2, mul])` and
  `S([64, 64, arg0_1, arg1_1])` — unparseable.

Reproduced end-to-end (GroupNorm model, second call with different H,W
triggers dynamic recompile):

```
[capture_hook] Failed to extract region: Captured repro
region_001_mean_var_bc169823856a_d47251a5.py failed eager validation:
name 'mul' is not defined
```

The file is left on disk but **excluded from index.json**, so
`merge_captures.py` never sees it. Net effect: **every region from a
dynamic compilation is silently dropped**, and only the print to stdout
records that it happened. The first (static) compile's regions are kept, so
the corpus systematically contains the cold-start kernels, not the
steady-state ones.

Also lost in the dynamic-origin region file (visible in the dropped
`region_001`): symint placeholders are emitted as *dead* forward args
(`mul: "Sym(32*32)", arg0_1: "Sym(32)"` — the annotation regex at :300 only
rewrites tensor annotations, and the symbol substitution turned
`Sym(s0*s53)` into the absurd `Sym(32*32)`).

### 1.2 `full_graph_harness.py` — `Sym(...)` annotations and the default=32 guess

- `_concrete_int` (`full_graph_harness.py:86-92`): `hint` if the live
  SymNode is available, else `int(value)`, else **default=32**. Used at
  :106-107 (shape/stride), :119 (symint hint), :338/345/347 (tensor spec).
  When called on a live FakeTensor the hint is real; the 32 only appears
  when parsing *text*.
- `_parse_intish` (`full_graph_harness.py:541-555`) is the text path and is
  worse than "guess 32": after failing `int()` and the `Sym(...)` unwrap, a
  bare identifier returns 32, but for compound exprs it greps the **first
  integer in the string** — including digits inside symbol names:

  | annotation text | `_parse_intish` result |
  |---|---|
  | `s0`, `s16` | 32 (default) |
  | `s16*s82` | **16** (digits of "s16") |
  | `s0*s53` | **0** |
  | `2*s0*s53` | 2 |
  | `64*s0*s53` | 64 |

  A stride annotation `[64*s0*s53, s0*s53, ...]` parses to `[64, 0, ...]`.
  (Verified by direct call; today this is latent because all 19 symbolic
  graphs got sidecars, see below.)
- `_symbolic_dims` (`:558-568`) records `{dim, symbol, default}` per
  non-integer dim — this is the **only place symbolic identity is kept**,
  and it keeps the symbol *string* (`"s16"`, or the compound `"s16*s82"`)
  but no ranges, no hint provenance, no cross-input sharing map (sharing is
  implicitly recoverable by string equality of symbols within one graph).
- The "19 symbolic_dim skips": `bench_parallel.py:430-444` classifies
  full graphs whose specs are annotation-only (no `.meta.json` sidecar) as
  `category: "symbolic_dim"` and skips them. All 19 graphs with `Sym(`
  annotations are `repros/models/torchbench/infer/opacus_cifar10/
  full_graph_{003..023}` (the 6 static siblings are 000,001,002,006,019,024).
  Since commit `6ed04a8b8` ("Backfill .meta.json batch 3") they have
  sidecars and are **no longer skipped** — they are now *silently
  instantiated at made-up shapes*: every symint value and symbolic dim in
  the opacus sidecars is exactly **32** (e.g. `full_graph_004.meta.json`:
  `arg2_1=32, arg3_1=32`, `arg4_1.shape=[64,64,32,32]`), i.e. the
  `_parse_intish` default, while the model's real per-layer dims are 16/8/4
  (cf. static sibling `full_graph_001`: `f32[64, 64, 16, 16]`). The
  backfilled sidecar laundered an annotation guess into "trusted" metadata
  — arguably a regression vs. the skip.
- `make_inputs_from_full_graph_specs` (`:1326-1342`) materializes
  `kind=symint` as `int(spec["value"])` and tensors at the (guessed)
  concrete shape. No way to ask for a different family point; `symbolic_dims`
  is carried in the spec but never used for generation, only for relaxing
  sidecar-vs-annotation validation (`:775-779`).
- `scripts/repartition_from_graphs.py:147-151, 322-325` (the path that
  re-partitions saved full graphs into canonical repros) substitutes **32**
  for every symint and every unparseable dim (`:174`), then re-traces with
  `make_fx` — so any canonical repro re-derived from a symbolic full graph
  is a synthetic 32-point of the family, not a real observed shape.

### 1.3 Captured repros: what S()/`_shape_param_N` actually record

- 1085/1482 canonical repros have `_shape_param_N` args and matching `S()`
  entries. These are **lifted reshape/view shape literals**, not lifted
  symbolic dims. In a static compilation the list is fully concrete and the
  mechanism works; the multi-line `shapes.txt` then varies them per captured
  call site (e.g. `var_mean_b1feb9d09685/shapes.txt` has 5 lines: `S([64,
  32, 2, 256])` … `S([64, 32, 8, 1024])` — five *different layers* of
  opacus_cifar10, each a separate static capture, NOT five points of one
  symbolic family).
- `S()` entries record **only concrete snapshot values**. There is no
  marker for "this dim was symbolic". Verified: zero non-numeric `S()`
  entries across all canonical shapes.txt.
- Dims that share a symbol are NOT recorded as shared. In
  `var_mean_8e122e47d0a5` (Albert, `s0=8`, `s1=512` originally), the `4096 =
  8*512` in `T([4096, 4096])`, the `8, 512` in `T([8, 512, 4096])`, and both
  `S()` entries all encode the same two symbols, but the repro has no record
  that scaling one requires scaling the others. A user adding a new
  shapes.txt line must re-derive the coupling by reading the graph.
- Provenance: `meta.json` has `models: [...]` labels but no
  "compiled-with-dynamic-shapes" flag; you cannot query "which repros came
  from dynamic compilations" — and per 1.1, the honest answer today is
  "none; they were all dropped".

### 1.4 CRITICAL: static repro of a dynamic compilation measures a different kernel

Experiment (B200, `coordinate_descent_tuning=True`), repro
`var_mean_b1feb9d09685` (opacus GroupNorm; the model's steady-state graphs
003-023 are dynamic over `s16, s82`):

| configuration | kernels | time |
|---|---|---|
| A. repro, `torch.compile` (static — **what our benchmark measures**) | **1** (`triton_per_fused...`, persistent reduction) | **13.7us** |
| B. repro, `torch.compile(dynamic=True)` | **2** (`triton_red_fused...` looped Welford + separate pointwise) | **40.4us** |
| C. actual dynamic full graph (`full_graph_004`, symint args, compiled as the model ran it) at the same 16x16 point | **2** | **35.5us** |
| C at 32x32 (same compiled artifact, other family point) | 2 | 46.2us |

Mechanism: with static shapes Inductor knows `rnumel=512` and emits a
persistent reduction with the epilogue fused (1 kernel). With symbolic
`rnumel = 2*s16*s82` it cannot pick the persistent variant, falls back to a
looped Welford reduction, and the epilogue (which re-reads the input) cannot
fuse — 2 kernels, plus `ks0..ks7` runtime shape args and runtime grid
computation. **The 13.7us our sweep reports for this repro is a kernel the
model only executed during its first (static) iteration; the model's
steady-state kernel is ~2.6x slower.** Any "Inductor gap" or oracle floor
computed against the static number is answering a different question than
"how fast does this model's kernel run".

Scope of impact today: confirmed dynamic compilations exist only for
opacus_cifar10 (19/553 saved full graphs; recapture scripts call
`torch.compile(model)` and run a single fixed input, so only models with
internally varying shapes go dynamic). But the silent-drop in 1.1 means any
future capture of a real workload (variable seq-len inference, vLLM-style
bucketing) will quietly produce the same static-only blind spot.

---

## Part 2: Design — symbolic serialization

### 2.1 Per-placeholder symbolic metadata

At capture time the SymNodes are live, so everything below is one
`shape_env` read away (verified with a post-grad probe: expr, hint,
`var_to_range`, `var_to_val`, `guards` are all accessible inside
`post_grad_custom_pre_pass`). Extend `placeholder_info` (and the
`full_graph_*.meta.json` input specs — same schema, bump
`schema_version` to 2) with an optional `symbolic` block; fully static
captures are unchanged:

```json
{
  "kind": "tensor",
  "name": "arg2_1",
  "shape": [64, 64, 16, 16],            // hints, exactly as today
  "stride": [16384, 256, 16, 1],        // hints, exactly as today
  "dtype": "float32",
  "symbolic": {
    "shape_exprs":  [null, null, "s16", "s82"],   // null = static dim
    "stride_exprs": ["64*s16*s82", "s16*s82", "s82", null]
  }
}
{
  "kind": "symint",
  "name": "mul",
  "value": 256,                          // hint, as today
  "expr": "s16*s82"                      // NEW: root symbol or compound
}
```

Plus one graph-level block (per repro `meta.json` / per full-graph sidecar),
NOT per placeholder, since symbols are shared across placeholders:

```json
"shape_env": {
  "symbols": {
    "s16": {"hint": 16, "range": [2, "inf"]},
    "s82": {"hint": 16, "range": [2, "inf"]}
  },
  "guards": [],                          // residual guard exprs, strings
  "captured_dynamic": true
}
```

Design points:

- **Exprs are sympy strings** rendered from `sym.node.expr`. Compound exprs
  (`s0*s53`, `128*((s0//128))`) stay compound — do not pre-evaluate. They
  are re-parseable with `sympy.sympify` against the `symbols` table.
- **The symbol-sharing map is free**: two dims sharing `s16` simply name the
  same key in `symbols`. No separate "sharing" structure needed — this is
  the same convention the `Sym()` annotations already use, now made
  machine-readable and extended to repros (whose code text is concretized).
- **Ranges** come from `shape_env.var_to_range` (e.g. `VR[2, int_oo]`);
  serialize `int_oo` as `"inf"`. They bound what alternate family points are
  valid.
- **Guards**: serialize `shape_env.guards` expr strings. For typical
  captures this is empty (specialization shows up as the symbol simply not
  existing), but when present (e.g. `s0 % 128 == 0` from a view), generating
  an off-guard point would benchmark a graph the model never ran — the
  generator must reject points that violate guards.
- **`_resolve_sym` keeps returning the hint** for the legacy fields. The
  symbolic block is additive; v2 repros without it parse exactly as today.

### 2.2 Shape-param lifting under dynamic shapes (fixes the silent drop)

`_lift_shape_arg` must handle `fx.Node`/SymInt entries in the shape list:

- concrete entry -> keep the int;
- symbolic entry -> record its **hint** in `shape_params[param]` (so the
  emitted literal is valid Python) and its **expr** in a parallel
  `shape_param_exprs[param]` list (e.g. `["64", "32", "2", "s16*s82"]`),
  serialized as the `S()` symbolic extension below.

Symint placeholders that become dead after lifting (the `mul, arg0_1,
arg1_1` dead args in the dropped region_001) should be pruned from the new
graph rather than emitted. Symint placeholders that remain live must be
emitted as ints — never as `S([hint])` (the type-conflation bug in 1.1);
give them their own compact form `I(hint, expr="s16")`.

### 2.3 `make_inputs`: hint by default, any valid family point on request

`make_inputs(shape_config=None)` behavior is unchanged — it generates at the
hint (the concrete snapshot), so every existing benchmark and result stays
comparable. Add an optional binding parameter:

```python
def make_inputs(shape_config=None, symbol_bindings=None):
    # symbol_bindings: {"s16": 24, "s82": 24}
```

Generation algorithm (lives in `repro_harness.py` / `full_graph_harness.py`,
shared):

1. start from the `symbols` table; override hints with `symbol_bindings`;
2. check each binding against its `range` and re-evaluate `guards` under
   the binding (sympy); raise on violation — loud beats silently
   benchmarking an impossible shape;
3. for every tensor input, evaluate `shape_exprs`/`stride_exprs` (falling
   back to the concrete hint fields for static dims), then generate exactly
   as today;
4. for symint inputs and lifted shape params, evaluate their exprs.

Because dims sharing `s16` are evaluated from one binding, coupled inputs
stay consistent by construction — the property the corpus can't express
today.

### 2.4 shapes.txt and the oracle signature format

A symbolic signature is a *family* of concrete signatures. Per the settled
dispatch design (`scripts/oracle_dispatch_design.md`: exact-only matching,
no fuzzy shape fallback), **oracle registration stays concrete**: an
`oracle_impl(shapes=...)` line continues to name one fully concrete
signature, and dispatch never interpolates. The symbolic info lives in
capture metadata, not in dispatch:

- `shapes.txt` lines stay concrete — each line is one evaluated family
  point. New optional header comments document the family so humans and
  tooling can mint new valid lines:

  ```
  # symbols: s16=[2,inf] hint=16; s82=[2,inf] hint=16
  # family: (T([64, 64, s16, s82], f32), T([64], f32), T([64], f32), S([64, 32, 2, s16*s82]), S([64, 64, s16, s82]))
  torchbench_opacus_cifar10_infer_001_ded9fcb8: (T([64, 64, 16, 16], f32), T([64], f32), T([64], f32), S([64, 32, 2, 256]), S([64, 64, 16, 16]))
  ```

  `_parse_shapes_txt` already skips `#` lines (`repro_harness.py:122`), so
  this is backward compatible with every existing parser.
- A tool (`scripts/expand_symbolic_shapes.py`, future work) evaluates the
  family at requested bindings and appends concrete lines; oracles then get
  registered per-line exactly as today. The "finite, known set of shapes"
  premise of exact-only dispatch is preserved — the family is just where
  new members of that finite set come from.
- The machine-readable copy of the family lives in `meta.json`
  (`shape_env` block + per-input exprs); the shapes.txt comment is a
  rendering of it, not a second source of truth.

### 2.5 Recording dynamic provenance; benchmarking with dynamic=True

Yes — repros should record it. Concretely:

- `meta.json` gains `"captured_dynamic": true` plus
  `"dynamic_dims": {"arg2_1": [2, 3]}` (inputs/dims that were symbolic).
  capture_hook sets it whenever any placeholder carries a SymInt.
- `benchmark_repro` / `bench_parallel` gain `--dynamic` mode: for flagged
  repros, compile with `torch._dynamo.mark_dynamic(tensor, dim)` on exactly
  the recorded dims (preferred over blanket `dynamic=True`, which would also
  make genuinely-static dims dynamic — measurably different again: B vs C in
  the experiment, 40.4 vs 35.5us). Report both numbers:
  `inductor_us` (static, comparable to history) and `inductor_dynamic_us`.
- Gap accounting can then distinguish "Inductor static gap" from "dynamic
  penalty" (the latter is its own Inductor investigation class — e.g. the
  persistent-reduction fallback above is a real, fixable
  PERSISTENT_THRESHOLD/dynamic-rnumel issue, currently invisible to us).
- Full-graph sweeps: `_classify_full_graph_definition` should treat
  sidecar symint values that came from annotation backfill as untrusted
  (restore the skip, or require a `"hint_source": "captured"` field) —
  the opacus sidecars' all-32 values are guesses (1.2) and currently pass
  as real.

### 2.5b UPDATE (2026-06-15): mark_dynamic is insufficient; use ShapesSpec — and lifted shape-params block it

Implementing 2.5 surfaced two things. (1) `mark_dynamic` only re-derives an
*approximation* of the ShapeEnv: it sets dynamic dims + ranges but the
guards are freshly re-traced and the symbols are re-allocated, so coupled
constraints (e.g. `s0*s53==256`) are NOT restored — dynamo re-discovers
whatever the re-trace yields, which can diverge. (2) Blanket `dynamic=True`
over-dynamizes and, worse, the lifted symints arrive as plain Python int
ARGUMENTS that dynamo specializes on → a recompile per binding.

The faithful mechanism is `torch.compile(mod, shapes_spec=ShapesSpec(...))`
(landed in the local pytorch as the ShapesSpec stack;
`torch/fx/experimental/dynamic_spec.py`). It re-creates the ShapeEnv rather
than re-deriving it — our capture maps 1:1:

| capture (shapes.json) | ShapesSpec |
|---|---|
| `symbols` + ranges | `ShapeVar(name, min=, max=)` — ONE object per symbol |
| per-dim/stride exprs (`s0*s53`) | `TensorSpec([64,64,s53,s0])`; derived dims `s0*s53` over the shared objects |
| coupling / symbol sharing | reuse the SAME ShapeVar object across slots |
| live symint inputs (`["I",..]`) | `IntVar(...)` params |
| guards (`s0*s53==256`) | `assumptions=[s0*s53 == 256]` (wired into the env, runtime-asserted) |
| hint | `optimization_hint=` |

VERIFIED (spike, EagerAndRecordGraphs): a model that derives shapes INSIDE
forward compiles to **1 graph** across a coupled rebind (16×16 → 32×8) — the
ShapeEnv is faithfully restored, no recompile. 

THE BLOCKER: our captured repros LIFT reshape/view shapes into
`_shape_param_N` **list** arguments (`[64,64,h,w]`). `ParamsSpec` has spec
types for tensors and scalar ints, but NOT for "a list whose elements are
dynamic" — so the changing list values re-specialize and force a recompile
(spike: same model with lifted params = 2 graphs; without = 1). Shape-param
lifting is a STATIC-capture hygiene mechanism; under dynamic shapes it
converts a symbolic dim into a frozen list literal and defeats ShapesSpec.

REFINED (2026-06-15, spikes C/D): the fix is NOT "stop lifting" — lifting
is good for reuse and we keep it. The culprit is specifically a shape param
lifted as a standalone *constant list* placeholder (`_shape_param_0 =
[64,32,2,256]`); the changing list literal re-specializes regardless of
symbol binding (spike C: symbols bound via scalar slots + plain-int list
params still = 2 graphs). The resolution, VERIFIED at 1 graph across the
coupled 16×16→32×8 rebind (spike D):

- KEEP lifting the symints as scalar args (`mul`, `arg0_1`, `arg1_1`) —
  these are the bare `IntVar` slots that bind the symbols. Already captured.
- For a lifted shape-param LIST, build the reshape target IN the forward
  body from those symint args — `reshape(x, [64, 32, 2, mul])` — instead of
  emitting a separate constant-list `_shape_param` placeholder. The list
  elements are then live SymInts and one artifact covers the family.

This is strictly a dynamic-capture GENERATION change (how repro.py spells
the reshape target), gated on `captured_dynamic`; static lifting is
untouched. We already capture both the symint args (`["I",hint,expr]`) and
the shape-param exprs (`["S",[64,32,2,"s0*s53"]]`), so the generator has
everything: emit the list as `[64, 32, 2, <symint-arg-or-expr>]` over the
in-scope lifted symints. ParamsSpec has no list-element spec, so this is
also the ONLY way to make a lifted reshape target dynamic under ShapesSpec.

Status: ShapesSpec builder from shapes.json + this generation change are
the remaining implementation. Until then, `--dynamic` marks the recorded
tensor dims (strict improvement over blanket; still recompiles per binding
for the constant-list-param form).

### 2.6 Retroactive recovery vs. recapture

Recoverable retroactively (no model rerun):

- **The 19 opacus full graphs**: exprs, symbol identity, and the sharing map
  are all in the `Sym()`/`f32[64, 64, s16, s82]` annotation text — a parser
  upgrade (`parse_full_graph_inputs` keeping exprs instead of
  `_parse_intish`) recovers the family structure. True hints are *not* in
  the text but are cross-referenceable from the static sibling graphs
  (000/001/002/006/019) and from canonical shapes.txt lines for the same
  model. Value ranges default to `[2, inf]` (the dynamo default for sizes,
  confirmed by probe); guards are unrecoverable but almost certainly empty
  here.
- **Stride exprs** in opacus graphs (annotations include `[64*s16*s82, ...]`
  where present) — same parser upgrade. Must NOT go through `_parse_intish`
  (the `s0*s53 -> 0` bug).

Requires recapture:

- **All canonical repros from dynamic-capable models**: repro code text was
  concretized at capture (1.1), and the dropped dynamic regions were never
  written. opacus_cifar10 is the only confirmed one (its var_mean/pointwise
  families: `var_mean_b1feb9d09685`, `pointwise_073278de7552`,
  `pointwise_e129f028e0b8` carry opacus labels). -> goes on the
  "models that are broken / need recapture" list.
- **shape_env extras** (ranges beyond defaults, guards, exact hint
  provenance) for *any* existing capture — only live SymNodes have them.
- **Any model whose dynamic regions were silently dropped**: undetectable
  from the corpus (the drop only printed to stdout). The recapture pass
  should re-run captures with the 2.2 fix and a counter of
  dynamic-origin regions, so "0 dynamic regions" becomes an assertable
  fact per model rather than an unknown.

### 2.7 Implementation order (when implementation is green-lit)

1. Fix `_lift_shape_arg` + symint emission (2.2) — stops the silent drop;
   purely additive to static captures.
2. Capture-side `symbolic` blocks + `shape_env` graph block (2.1) in
   capture_hook and `graph_constraints_from_gm` (schema_version 2).
3. `make_inputs` symbol_bindings (2.3) + shapes.txt family comments (2.4).
4. `captured_dynamic` provenance + `--dynamic` bench mode (2.5).
5. Retroactive opacus parser upgrade + recapture of opacus_cifar10 (2.6).
6. Restore the untrusted-sidecar skip for annotation-backfilled symints
   (2.5, last bullet).

---

## SETTLED: repro-level serialization is shapes.json (decision 2026-06-10)

Per discussion: JSON sidecar instead of comment-headers in shapes.txt, for
design flexibility. Per repro:

```json
{
  "symbols": {"s0": {"hint": 8, "range": [1, 256]}, "s1": {"hint": 512, "range": [1, 4096]}},
  "family": [{"shape_exprs": ["s0", "s1", 1024], "dtype": "bf16"},
             {"shape_exprs": ["s0", "s1"], "dtype": "i64", "gen": {"kind": "index", "high": 32128}}],
  "points": [
    {"label": "vllm_llama_decode_bs1", "bindings": {"s0": 1, "s1": 1}, "source": "captured"},
    {"label": "vllm_llama_decode_bs64", "bindings": {"s0": 64, "s1": 1}, "source": "ladder"}
  ]
}
```

- `symbols`: shared table — cross-input dim sharing is expressible (both
  inputs binding s0 move together). Ranges from shape_env when available.
- `family`: per-input shape exprs (ints for static dims, symbol names or
  expr strings like "2*s0" for dynamic). dtype + gen constraints live here
  once, not per point.
- `points`: a concrete point is just a symbol binding. `source` records
  provenance: "captured" (observed) vs "ladder" (added for evaluation).
  Point SELECTION is a benchmarking-policy decision, not capture's job —
  capture records only what it observed.
- TRANSITION: shapes.txt is GENERATED from shapes.json (concrete lines,
  same format) so load_shape_configs / oracle signatures / dispatch keep
  working unchanged. Consumers migrate to the JSON at leisure.
- Static repros get a degenerate shapes.json (no symbols, one captured
  point). Model-level sidecars (full_graph meta.json) carry the same
  symbols/shape_exprs structure per the design above.
- Oracles and dispatch stay EXACT-SHAPE per the settled oracle_impl design;
  the family exists to generate points and document structure, never to
  fuzzy-match.
