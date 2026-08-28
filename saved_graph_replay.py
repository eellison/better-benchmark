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


def _sidecar_symbol_hints(graph_path: Path) -> dict:
    """Per-symbol NATIVE hints recorded at capture time in the graph's sidecar
    meta.json (full_graph_XXX.meta.json -> {"symbols": {"s16": {"hint": 4}}}).

    The saved .py annotations are symbolic (Sym(s16)) and carry no hint, so
    without the sidecar the re-trace seeds every symbol at a benign default.
    The default is fine for FIDELITY (the symbol stays symbolic either way)
    but not for the recorded POINT: the point's bindings become the default
    bench binding and its shape_hash (which includes hints) is the join key
    back to live-captured occurrence counts — a non-native hint times the
    wrong problem size and orphans the accounting join. Missing/corrupt
    sidecar -> {} (caller falls back to the default)."""
    meta_path = graph_path.with_name(graph_path.stem + ".meta.json")
    if not meta_path.exists():
        return {}
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return {}
    hints = {}
    for name, defn in (meta.get("symbols") or {}).items():
        if not isinstance(defn, dict):
            continue
        hint = defn.get("hint")
        # bool is an int subclass but never a valid size hint (True would seed
        # a 1 -> ShapeEnv 0/1-specialization); a float that is exactly integral
        # is coerced losslessly (JSON may round-trip 4 as 4.0). Anything else
        # (None, non-integral float, str) is dropped -> caller's default hint.
        if isinstance(hint, bool):
            continue
        if isinstance(hint, int):
            hints[name] = hint
        elif isinstance(hint, float) and hint.is_integer():
            hints[name] = int(hint)
    return hints


def _build_symbolic_inputs(specs, dev, symbol_hints=None):
    """Rebuild the forward inputs for a dynamic saved graph with ONE shared
    ShapeEnv symbol per symbol NAME, so a symint input (arg2_1:Sym(s16)) and a
    tensor dynamic dim (arg4_1:[...,s16,...]) trace as the SAME symbol — the
    structure the model actually had. Returns (FakeTensorMode, inputs), or None
    if a symbol could not be resolved (caller falls back to real-mode trace).

    Symbols are created backed at their recorded hint (size_hint), mirroring
    capture; the strides come from the annotation's stride exprs evaluated
    over the same symbols, so a non-contiguous dynamic tensor round-trips.
    """
    import torch
    from torch.fx.experimental.symbolic_shapes import ShapeEnv, DimDynamic
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch._dynamo.source import LocalSource

    # Collect symbol names + a hint per symbol. Precedence: the sidecar
    # meta.json's NATIVE hints (symbol_hints) first — they set the recaptured
    # point's bindings and shape_hash, which must match the live capture —
    # then a symint spec's own hint, then a benign default (8). The hint only
    # picks which concrete size make_fx traces at — the symbol stays symbolic
    # regardless — so the default is safe for structure, wrong for the point.
    hints: dict[str, int] = dict(symbol_hints or {})

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
                hints.setdefault(nm, int(s.get("hint", 8) or 8))
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
                    hints.setdefault(nm, 8)
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
        sym = shape_env.create_symbol(
            trace_h, source=LocalSource(nm), dynamic_dim=DimDynamic.DYNAMIC)
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
