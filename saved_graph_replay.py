"""Faithful symbolic reconstruction for saved dynamic FX graphs.

This is not a capture path. Live graphs are captured by capture_hook.py. This
module rebuilds the symbolic inputs needed to re-trace an already-saved graph
without baking its symbols to hints, which keeps recapture a fixed point.
"""

import json
from pathlib import Path


def _specs_have_symbols(specs) -> bool:
    """True if any parsed input spec carries a symbolic dim/stride/symint —
    i.e. the saved graph is dynamic and must be re-traced symbolically."""
    for s in specs:
        if not isinstance(s, dict):
            continue
        if s.get("kind") == "symint" and s.get("expr") is not None:
            return True
        for key in ("shape", "stride"):
            for d in (s.get(key) or []):
                if isinstance(d, str):
                    return True
        sym = s.get("symbolic") or {}
        if sym.get("shape_exprs") or sym.get("stride_exprs"):
            return True
    return False


def _sidecar_symbol_definitions(graph_path: Path) -> dict:
    """Per-symbol metadata recorded in a saved graph's sidecar."""
    meta_path = graph_path.with_name(graph_path.stem + ".meta.json")
    if not meta_path.exists():
        return {}
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return {}
    return {
        name: dict(defn)
        for name, defn in (meta.get("symbols") or {}).items()
        if isinstance(name, str) and isinstance(defn, dict)
    }


def _coerce_hint(defn) -> int | None:
    from input_codec import _symbol_point_value
    hint = _symbol_point_value(defn)
    # bool is an int subclass but never a valid value. Do not coerce floats:
    # replay must preserve the serialized point exactly, not normalize it.
    if isinstance(hint, int) and not isinstance(hint, bool):
        return hint
    return None


def _sidecar_symbol_hints(graph_path: Path) -> dict:
    """Per-symbol observed point values from a saved graph's sidecar.

    Backed symbols store this as ``hint``; unbacked symbols store the
    independent ``observed_value``. Optimization hints are deliberately not
    returned as points. Missing/corrupt sidecar -> {}."""
    return {
        name: hint
        for name, defn in _sidecar_symbol_definitions(graph_path).items()
        if (hint := _coerce_hint(defn)) is not None
    }


def _build_symbolic_inputs(specs, dev, symbol_hints=None,
                           symbol_definitions=None):
    """Rebuild the forward inputs for a dynamic saved graph with ONE shared
    ShapeEnv symbol per symbol NAME, so a symint input (arg2_1:Sym(s16)) and a
    tensor dynamic dim (arg4_1:[...,s16,...]) trace as the SAME symbol — the
    structure the model actually had. Returns (FakeTensorMode, inputs), or None
    if a symbol could not be resolved (caller falls back to real-mode trace).

    Backed/unbacked provenance, observed values, ranges, and optimization
    hints come from the sidecar symbol definitions. Strides come from the
    annotation's expressions evaluated over the same symbols, so a
    non-contiguous dynamic tensor round-trips.
    """
    import torch
    from torch.fx.experimental.symbolic_shapes import ShapeEnv, DimDynamic
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch._dynamo.source import LocalSource

    # Collect symbol names + a hint per symbol. Precedence: the sidecar
    # meta.json's NATIVE hints (symbol_hints) first — they set the recaptured
    # point's bindings and shape_hash, which must match the live capture —
    # then a symint spec's own trace value, then the same benign template value
    # (2) used by live capture. This value only
    # picks which concrete size make_fx traces at — the symbol stays symbolic
    # regardless — so the default is safe for structure, wrong for the point.
    definitions = dict(symbol_definitions or {})
    hints: dict[str, int] = {}
    for name, definition in definitions.items():
        observed = _coerce_hint(definition)
        optimization_hint = definition.get("optimization_hint")
        if observed is not None:
            hints[name] = observed
        elif (isinstance(optimization_hint, int)
              and not isinstance(optimization_hint, bool)):
            # Trace seed only: this must not become an observed point.
            hints[name] = optimization_hint
    hints.update(symbol_hints or {})

    import ast as _ast

    def _expr_symbols(text: str) -> set:
        """Symbol identifiers referenced by a closed arithmetic expr, via AST
        (no regex, lossless over the printed grammar): 's0*s53' -> {'s0','s53'},
        's53' -> {'s53'}, '64' -> set(). Call-position names are NOT symbols —
        'CeilToInt(s0/3)' references only s0; seeding 'CeilToInt' would mint a
        bogus unused ShapeEnv symbol. A token the grammar doesn't cover
        (SyntaxError) contributes no symbols, so the caller later falls back to
        real-mode rather than seeding a half-parsed name."""
        try:
            tree = _ast.parse(text, mode="eval")
        except (SyntaxError, ValueError):
            return set()
        called = {n.func.id for n in _ast.walk(tree)
                  if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name)}
        return {n.id for n in _ast.walk(tree)
                if isinstance(n, _ast.Name)} - called

    # A live symint carries an expr ('s53' or a product 's0*s53'); seed EVERY
    # symbol it references, not just bare-identifier exprs — a composite symint
    # whose components appear in no tensor dim must still resolve.
    for s in specs:
        if not isinstance(s, dict):
            continue
        if s.get("kind") == "symint" and isinstance(s.get("expr"), str):
            for nm in _expr_symbols(s["expr"]):
                hints.setdefault(nm, int(s.get("hint", 2) or 2))
    # Seed any still-unhinted symbols that appear in tensor SHAPE *or* STRIDE
    # tokens. Strides are seeded too: a symbol that occurs ONLY in a stride
    # expr (e.g. a permuted/non-contiguous dynamic tensor) would otherwise be
    # unknown to eval_dim, dropping the strided view to a contiguous fallback.
    for s in specs:
        if not isinstance(s, dict) or s.get("kind") == "symint":
            continue
        for tok in list(s.get("shape") or []) + list(s.get("stride") or []):
            if isinstance(tok, str):
                for nm in _expr_symbols(tok):
                    hints.setdefault(nm, 2)
    if not hints:
        return None

    shape_env = ShapeEnv()
    symnodes: dict[str, object] = {}
    trace_hints: dict[str, int] = {}
    for nm, h in hints.items():
        # ShapeEnv 0/1-specializes size hints of 0 and 1 — the created "symbol"
        # collapses to a constant, so a dynamic dim would silently recapture as
        # static. The hint only picks the concrete size make_fx traces at (the
        # symbol stays symbolic regardless), so trace at >=2 to keep it dynamic.
        trace_h = h if isinstance(h, int) and h >= 2 else 2
        trace_hints[nm] = trace_h
        defn = definitions.get(nm) or {}
        if defn.get("unbacked") is True:
            symint = shape_env.create_unbacked_symint(source=LocalSource(nm))
            symnodes[nm] = symint

            # Preserve the captured range instead of replacing it with the
            # unconstrained default of a fresh unbacked symbol.
            recorded_range = defn.get("range")
            if recorded_range is not None:
                if (not isinstance(recorded_range, list)
                        or len(recorded_range) != 2):
                    return None
                lo, hi = recorded_range

                def valid_bound(value):
                    return (
                        value is None
                        or (isinstance(value, int)
                            and not isinstance(value, bool))
                    )

                if (not valid_bound(lo) or not valid_bound(hi)
                        or (lo is not None and hi is not None and lo > hi)):
                    return None
                import sympy
                from torch.utils._sympy.numbers import int_oo
                from torch.utils._sympy.value_ranges import ValueRanges
                shape_env._update_var_to_range(
                    symint.node.expr,
                    ValueRanges(
                        -int_oo if lo is None else sympy.Integer(lo),
                        int_oo if hi is None else sympy.Integer(hi),
                    ),
                )
            observed = _coerce_hint(defn)
            if observed is not None:
                shape_env.set_real_tensor_prop_unbacked_vals(
                    symint.node.expr, observed)
            optimization_hint = defn.get("optimization_hint")
            if (isinstance(optimization_hint, int)
                    and not isinstance(optimization_hint, bool)):
                shape_env._set_unbacked_var_to_hint_override(
                    symint, optimization_hint)
        else:
            sym = shape_env.create_symbol(
                trace_h, source=LocalSource(nm),
                dynamic_dim=DimDynamic.DYNAMIC)
            symnodes[nm] = shape_env.create_symintnode(sym, hint=trace_h)

    # Evaluate a dim/stride token to a torch SymInt (or int), covering the
    # SAME closed expression grammar input_codec accepts everywhere else —
    # not just +/-/* products but the torch shape functions a saved
    # annotation can legitimately carry ('s0//2', 'CeilToInt(s0/3)',
    # 'PythonMod(s0, 4)', 'ModularIndexing(...)'). The old evaluator's
    # charset gate rejected those, so a VALID symbolic dim returned None and
    # the whole graph silently recaptured STATIC — dynamism lost and
    # f(f(x)) != f(x). _sympify_expr is the shared
    # safe-grammar boundary (AST allowlist gates the string before sympify
    # eval()s it; torch's sympy function classes parse natively); the parsed
    # expr is rebuilt over the ShapeEnv's OWN backing symbols and minted into
    # a torch SymInt via create_symintnode, so composite dims stay tied to
    # the same symbols as every other dim. An expr outside the grammar or
    # over unknown symbols returns None and the caller fails ingestion
    # loudly rather than guessing.
    from input_codec import _sympify_expr

    def eval_dim(tok):
        if isinstance(tok, int):
            return tok
        if not isinstance(tok, str):
            return tok
        tok = tok.strip()
        if tok.lstrip("-").isdigit():
            return int(tok)
        if tok.isidentifier():
            return symnodes.get(tok)  # bare symbol (None if unknown)
        try:
            expr = _sympify_expr(tok)
        except ValueError:
            return None  # outside the closed safe grammar — never guess
        free = list(getattr(expr, "free_symbols", ()))
        if any(s.name not in symnodes for s in free):
            return None
        # Concrete value at the trace hints — create_symintnode needs it, and
        # a non-integer fold means a function we can't evaluate: refuse.
        hint_val = expr.subs({s: trace_hints[s.name] for s in free})
        if not getattr(hint_val, "is_Integer", False):
            return None
        if not free:
            return int(hint_val)  # constant expr folds to a plain int
        sub = expr.subs({s: symnodes[s.name].node.expr for s in free})
        return shape_env.create_symintnode(sub, hint=int(hint_val))

    fake_mode = FakeTensorMode(shape_env=shape_env)

    def _is_contiguous(shape, strides):
        """Row-major contiguous check that works symbolically: compare each
        stride to the running product of trailing dims using == over SymInts
        (guarded). If any comparison can't be decided, treat as non-contiguous
        (build the explicit strided view) rather than guess."""
        running = 1
        for st, d in zip(reversed(strides), reversed(shape)):
            same = (st == running)
            if isinstance(same, bool):
                if not same:
                    return False
            else:
                return False  # symbolic-undecided -> use as_strided
            running = running * d
        return True

    def build(spec):
        kind = spec.get("kind")
        if kind == "symint":
            nm = spec.get("expr")
            if nm is None:
                # Constant-valued symint (Sym(256)): no symbol — keep the
                # recorded concrete value, don't collapse it to the default.
                return spec.get("value", spec.get("hint", 8))
            # Bare symbol ('s53') or composite ('s0*s53'): evaluate over the
            # shared symbol nodes so it stays a torch SymInt tied to the SAME
            # symbols as the tensor dims. An unknown component -> None -> caller
            # falls back to real-mode tracing (never silently bakes a guess).
            return eval_dim(nm)
        if kind == "scalar":
            return spec.get("value", 1)
        shape = [eval_dim(d) for d in (spec.get("shape") or [])]
        if any(d is None for d in shape):
            return None
        dtype = spec.get("dtype") or torch.float32
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype, torch.float32)
        device_text = spec.get("device") or ""
        tdev = torch.device("cuda:0") if device_text.startswith("cuda") and \
            torch.cuda.is_available() else dev
        stride_spec = spec.get("stride")
        with fake_mode:
            if stride_spec and len(stride_spec) == len(shape):
                strides = [eval_dim(st) for st in stride_spec]
                if not any(st is None for st in strides) and \
                        not _is_contiguous(shape, strides):
                    # storage big enough for the (symbolic) strided view; the
                    # arithmetic stays in torch SymInt space (shape/strides are
                    # SymInts), so torch.empty/as_strided accept it.
                    storage = 1
                    for st, d in zip(strides, shape):
                        storage = storage + st * (d - 1)
                    base = torch.empty([storage], dtype=dtype, device=tdev)
                    return torch.as_strided(base, shape, strides)
            return torch.empty(shape, dtype=dtype, device=tdev)

    inputs = []
    for spec in specs:
        built = build(spec)
        if built is None:
            return None  # unresolved -> let caller fall back to real mode
        inputs.append(built)
    return fake_mode, inputs
