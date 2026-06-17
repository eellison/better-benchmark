# compile_fx-direct: the cleaner alternative to ShapesSpec (2026-06-16)

User hypothesis: "how does the AOT repro run with specific expressions? it
wouldn't surprise me if it used compile_fx directly... that's the other
alternative to ShapesSpec." — CONFIRMED, and it is structurally better.

## What runs today
A full-graph / dynamic repro is re-exec'd into a fresh nn.Module and run via
`torch.compile(instance)` (scripts/bench_parallel.py:1699,
model_attribution.py:296). That goes back THROUGH dynamo with concrete
inputs, so the symbolic structure is RE-DERIVED — the exact limitation that
made us look at ShapesSpec (dynamo-level symbol/guard reconstruction).

## The experiment (VERIFIED, /tmp/dyn_capture_work/compile_fx_probe2.py)
`torch._inductor.compile_fx(gm, example_inputs)` takes an FX GraphModule +
example inputs and runs Inductor (+AOTAutograd) on it, BYPASSING dynamo.
Crucially it calls `detect_fake_mode(example_inputs)` — so if the example
inputs are FAKE tensors carrying SymInt shapes, it compiles the graph
SYMBOLICALLY.

Fed the captured post-grad gm (placeholders: `arg0_1=Sym(s77)`,
`arg1_1=Sym(s27)`, `arg2_1: f32[s77, s27]`) + its symbolic fake placeholder
vals straight to compile_fx:
- ONE compiled artifact ran BOTH (64,100) and (128,200) -> out (64,1)/(128,1),
  no dynamo, no recompile, no ShapesSpec.
- Call convention: the artifact wants args in placeholder order — the lifted
  SYMINTS passed explicitly (`compiled(s77_val, s27_val, tensor)`). This is
  EXACTLY the `["I", hint, expr]` lifted-symint inputs we already capture, in
  the order the forward lists them. Our capture shape is already correct for
  this path.

## Why it beats ShapesSpec
| | ShapesSpec | compile_fx direct |
|---|---|---|
| level | dynamo (re-trace a reconstructed module) | post-AOT (run the captured graph itself) |
| fidelity | re-derives symbols+guards, hopes they match | uses the model's ACTUAL symbolic graph |
| input | hand-built ShapeVar/IntVar/assumptions from shapes.json | the FX graph + fake symbolic inputs |
| telemetry bug | hits the `_shapes_spec` JSON-dump crash | avoided |
| coupling/guards | must rebuild as `assumptions` | already baked in the graph |

## q1 RESOLVED (2026-06-16): GM source = re-run the repro's own dynamic
## compile, intercept the post-grad gm, compile_fx that.
Tried and REJECTED: hand-building a symbolic graph via make_fx +
create_unbacked_symint (q1-a) — it fights make_fx's unbacked-symint tracing
mode (`_set_unbacked_bindings` AssertionError) and reallocates symbol names.
Tried and REJECTED: serialize the post-grad gm (q1-b) — we don't store one.

WHAT WORKS (verified, /tmp/dyn_capture_work/q1_min.py + compile_fx_probe2.py):
the captured repro's forward ALREADY takes the lifted symints as args and is
written symbolically (reshape(x,[64,32,2,mul])). So: compile the repro with
the same dynamic=True + mark_dynamic on the recorded dims the capture used,
intercept the post-grad gm at the post_grad hook (the SAME mechanism capture
uses), and hand THAT gm + its symbolic placeholder fake-vals to compile_fx.
One artifact then serves every binding (proved: hint 16x16 -> (64,64,16,16);
64x100 & 128x200 from one artifact, no recompile).

Per-binding ARG construction: do NOT hand-roll {name:val} maps (brittle —
the post-grad graph reallocates/derives symbols like s40, mul). Use the
codec's instantiate_point(point, symbols, bindings) to build the full,
consistent concrete input list for each binding, in placeholder order, then
call compiled(*args). instantiate_point already evaluates every symint/dim/
stride expr under one binding — exactly what the artifact's call convention
needs.

Remaining (smaller) open items:
2. compile_fx is a FRAGILE internal API (bench_parallel.py:37 already flags
   "breaks across PT versions"). Acceptable for a bench harness, but pin the
   call site + guard a clear version-skew error.
3. BENCH METHODOLOGY. The artifact is a plain callable; CUDAGraph
   capture/replay + the lock still apply. The bench wrapper changes from
   "torch.compile(module)" to "compile_fx(gm, fake_inputs) then call with
   per-binding concrete args".
4. STATIC PARITY. The static corpus path stays torch.compile — only the
   dynamic `--dynamic` path would use compile_fx. Confirm the numbers from
   the two paths are comparable (same CUDAGraph+CD methodology).

## Recommendation
Make compile_fx-direct the dynamic-bench mechanism instead of ShapesSpec:
faithful (the model's own symbolic graph), no spec-builder, no telemetry
bug, and the lifted-symint capture shape already matches its call
convention. Decision + design needed on the GM source (q1) before building.
ShapesSpec stays a documented fallback.
