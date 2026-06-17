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

## DECISION (2026-06-16): per-binding compile_fx + kernel-identity, NOT
## literal one-artifact reuse.
Experiment surfaced a real fork. Reusing ONE compile_fx artifact across
bindings requires building call-args in POST-GRAD placeholder order at an
arbitrary binding. The post-grad graph (verified) is reordered vs the repro
forward AND carries AOT-introduced FREE symints (e.g. s62, s73 — the lifted
arg0_1/arg1_1 symint inputs, which dynamo does NOT unify with the tensor's
own dim symbols s40/s32). Mapping every free graph symbol back to a
shapes.json binding is fragile and PT-internal-version-sensitive.

What the bench actually needs to MEASURE is the dynamic kernel's runtime at
each binding. A dynamic kernel's codegen is binding-INDEPENDENT, so a fresh
compile_fx at each binding produces the SAME kernel; the timing is identical
to reusing one cached object. The only thing literal reuse adds is proving
the kernel is physically the same — which we get MORE robustly by checking
KERNEL-NAME IDENTITY across the per-binding compiles. Same kernel name(s) at
every binding == it is one dynamic kernel (the vLLM-bucketing property),
established without the fragile arg-mapping.

So the implemented design (--dynamic-mode=compile_fx):
  for each binding:
    - make_inputs_from_config at the binding (codec; concrete forward inputs)
    - mark the recorded dynamic dims
    - intercept the post-grad gm at THIS binding via the post_grad hook
    - compile_fx(gm, ex); realize ex at this binding's placeholder hints
      (the hints ARE this binding, since we compiled at it -> no
      arg-reordering, no free-symbol reverse-engineering)
    - time under the shared CUDAGraph+lock methodology
  then assert the dynamic kernel-name SET is identical across all bindings
  (a divergence means it is NOT one dynamic kernel -> flag, not silently
  average). VERIFIED: 3 bindings (16x16/24x24/8x32) each compiled+ran.
Literal single-artifact reuse is noted as future work (only needed if a
consumer wants the exact cached-object semantics, not the timing).

## UPDATE (2026-06-16, deeper experiments): WHICH gm matters — make_fx alone
## specializes; the DYNAMO post-grad gm is what compiles dynamically.
Pursuing "make_fx once -> compile_fx once" (clean forward-order
correspondence, our own symbols, no AOT reordering) ran into a hard wall,
established by experiment:

- `make_fx(mod, tracing_mode="symbolic")(*marked_inputs)` DOES give a clean
  graph: 6 placeholders == 6 forward inputs, in forward order, our symbols
  (tensor [64,64,s,s], mul=s**2, symint args=s). Correspondence is perfect.
- BUT `compile_fx(make_fx_gm, inputs)` SPECIALIZES to the trace shapes:
  AOTAutograd's runtime_wrapper emits `assert_size_stride(arg0, (…trace
  dims…))`, so calling at any OTHER binding raises
  `AssertionError: expected size 16==32`. `ignore_shape_env=True` does NOT
  remove it (the assert is in the AOT runtime wrapper, not gated by it).
  Also: tracing at a hint where two dynamic dims are EQUAL (16==16) unifies
  them into ONE symbol (square constraint) — must trace at distinct dims.
- The earlier "ONE artifact served both 64x100 and 128x200" result
  (compile_fx_probe2) worked because it fed compile_fx the gm dynamo +
  AOTAutograd ALREADY produced under torch.compile(dynamic=True) — that gm
  carries the proper dynamic-compilation context (no specializing asserts),
  but its placeholders are REORDERED and carry AOT-introduced FREE symints
  (s62/s73) that don't map to our shapes.json — the fragile arg problem.

So the two are in tension:
  (A) make_fx gm: clean correspondence, but compile_fx specializes it -> can't
      rebind.
  (B) dynamo post-grad gm: compiles dynamically (rebinds), but reordered +
      free symbols -> fragile per-binding args.

REMAINING OPTIONS (decision needed — this is deep PT-internals integration):
 1. Per-binding compile (the robust fallback already proven): a fresh
    dynamic compile + post-grad intercept PER binding, realize args at that
    binding's own placeholder hints (no cross-binding mapping). Works today
    (verified 3 bindings). Recompiles per point — fine for timing (we don't
    time compile), and kernel-name identity across bindings shows it's one
    dynamic kernel. DOWNSIDE: not a literal single artifact; the kernel-name
    scan reads inductor's generated .py (a parse — undesirable).
 2. Drive AOTAutograd directly (aot_module_simplified, what dynamo's inductor
    backend calls) on the make_fx gm so the dynamic context is set up
    WITHOUT the specializing asserts. Cleanest if it works; needs an
    experiment to confirm the asserts are avoidable this way.
 3. Strip/relax the assert_size_stride guard on the compiled make_fx artifact
    (post-process the generated code or a config). Hacky.

CURRENT LEAN: option 2 (aot_module_simplified on the clean make_fx gm) is the
principled target — clean correspondence AND dynamic. Needs one more
focused experiment before committing. The --dynamic-mode=compile_fx scaffold
is wired but the gm-source helper is NOT finalized pending this.

## CONCLUSION (2026-06-16, option-2 experiment done): make_fx-based paths
## SPECIALIZE; only the dynamo-produced gm compiles dynamically.
Tested aot_module_simplified(make_fx_gm, inputs, fw_compiler=compile_fx_inner)
— the exact entry dynamo's inductor backend uses. It got past the output
convention (needs tuple output — trivial wrapper) but STILL emits
`assert_size_stride(arg0, (…trace dims…))` and fails to rebind (16==32). So
the specializing asserts are NOT specific to compile_fx; they come from
feeding a make_fx graph CONCRETE example inputs — the dynamic-compilation
context that suppresses them is established by how DYNAMO sets up the
ShapeEnv/inputs, which neither make_fx nor a direct AOT call reproduces.

DEFINITIVE: the dynamo-produced post-grad gm (option B) is the ONLY graph
that compiles to a rebindable dynamic kernel. Its cost is the placeholder
reordering + AOT-introduced free symints (s62/s73) — the per-binding-args
problem. There is no clean "make_fx once + compile_fx once" with our forward
correspondence; that always specializes.

THEREFORE the viable designs reduce to:
 (1) PER-BINDING dynamo-compile + post-grad intercept, realize args at THAT
     binding's placeholder hints (proven working, 3 bindings). One compile
     per point; kernel-name identity across points evidences one dynamic
     kernel. This is the robust, shippable path. (The kernel-name read is a
     .py scan — acceptable as a cross-check, not on the data path.)
 (2) Keep the EXISTING mark_dynamic --dynamic path (already shipped) — it
     also recompiles per binding for lifted-symint repros, and is simpler
     (no compile_fx internals, no fragile-API risk). compile_fx buys nothing
     over it UNLESS we solve literal one-artifact reuse, which the
     experiments show needs the fragile free-symbol arg mapping.

RECOMMENDATION (revised): compile_fx-direct does NOT cleanly beat the
existing mark_dynamic path for the per-binding bench — both recompile per
point; compile_fx adds internal-API fragility for no timing benefit at the
per-binding granularity. Its ONLY advantage (one physical artifact across
bindings) is exactly the part that's fragile (free-symbol args). So: KEEP
mark_dynamic as the --dynamic mechanism; treat compile_fx-direct /
one-artifact-reuse as future work gated on a clean solution to the
free-symbol arg mapping (or an upstream API that exports a rebindable
artifact). The --dynamic-mode=compile_fx flag is left UNWIRED (scaffold
only) with this finding as the rationale. Net: this investigation's value
is the clear NEGATIVE — we now know compile_fx isn't the easy win it looked
like, and why.
