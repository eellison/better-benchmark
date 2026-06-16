"""Compact structured encoding for input specs — THE shared format.

One encoding used by BOTH the per-pattern shapes.json points and the
per-model full_graph_*.meta.json sidecars (previously: a verbose
dict-per-tensor in the model sidecar vs a T()/S() string in shapes.json —
same information, two formats, one bloated and one text-parsed).

Compact form (plain JSON, no string grammar):

    [[128, 128, 28, 28], "bf16", {"st": [100352, 1, 3584, 128]}]
    [[4096], "i64", {"gen": ["index", 0, 1000]}]
    ["S", [128, 512, 1, 1]]          # lifted shape param (repros only)
    ["sc", 0.5]                       # python scalar
    ["sym", 128]                      # symint hint

Tensor entry: [shape, dtype, opts?]. opts keys, ALL optional (defaults
omitted): "st" stride (absent = contiguous), "dev" device (absent =
cuda), "off" storage_offset (absent = 0), "gen" generation as
[kind, *args] (absent = randn for floats / small ints for int dtypes),
"data" exact payload list (small int/bool tensors).

The verbose per-tensor dict (full_graph_harness._tensor_spec_from_value
format) remains the IN-MEMORY working representation — this module
converts at serialization boundaries only, so loaders/validators are
unchanged. The human-readable T()/S() string is a RENDERING of this data
(repro.py's _shapes_config documentation line), never parsed back.
"""
from __future__ import annotations

from typing import Any

SHORT_DTYPE = {
    "float32": "f32", "float16": "f16", "bfloat16": "bf16",
    "float64": "f64", "int64": "i64", "int32": "i32",
    "int16": "i16", "int8": "i8", "bool": "b8", "uint8": "u8",
    "uint16": "u16", "uint32": "u32", "uint64": "u64",
    "complex64": "c64", "complex128": "c128",
    "float8_e4m3fn": "f8e4m3fn", "float8_e5m2": "f8e5m2",
}
LONG_DTYPE = {v: k for k, v in SHORT_DTYPE.items()}


def _short_dtype(name: str) -> str:
    name = str(name).removeprefix("torch.")
    return SHORT_DTYPE.get(name, name)


def _long_dtype(name: str) -> str:
    return LONG_DTYPE.get(name, name)


def _gen_to_compact(gen: dict) -> list | None:
    kind = gen.get("kind")
    if kind in (None, "randn"):
        return None  # default for floats — omitted
    if kind == "index":
        return ["index", int(gen.get("low", 0)), int(gen.get("high", 100))]
    if kind == "randint":
        return ["randint", int(gen.get("low", 0)), int(gen.get("high", 2))]
    if kind == "permutation":
        out = ["perm"]
        if gen.get("size") is not None:
            out.append(int(gen["size"]))
        return out
    if kind == "offsets":
        return ["offsets", int(gen.get("high", 1))]
    if kind == "constant":
        return ["const", gen.get("value", 0)]
    return [kind] + [v for k, v in sorted(gen.items()) if k != "kind"]


def _gen_from_compact(c: list) -> dict:
    kind = c[0]
    if kind == "index":
        return {"kind": "index", "low": c[1], "high": c[2]}
    if kind == "randint":
        return {"kind": "randint", "low": c[1], "high": c[2]}
    if kind == "perm":
        return {"kind": "permutation",
                "size": c[1] if len(c) > 1 else None}
    if kind == "offsets":
        return {"kind": "offsets", "high": c[1]}
    if kind == "const":
        return {"kind": "constant", "value": c[1]}
    return {"kind": kind}


# Keys the codec encodes structurally; anything ELSE on a spec passes
# through opts["x"] verbatim so no producer field is ever silently lost.
_KNOWN_KEYS = {"kind", "name", "shape", "dtype", "stride", "device",
               "storage_offset", "generator", "gen", "exact", "data",
               "alias_group", "symbolic"}


def _overlay_exprs(slots: list, exprs: list | None) -> list:
    """Overlay per-slot expr strings onto hint ints: where exprs[i] is a
    non-None string, the slot becomes that expr; otherwise the hint int
    stays. None/empty exprs -> slots unchanged (static)."""
    if not exprs:
        return slots
    return [e if e is not None else s for s, e in zip(slots, exprs)]


def _exprs_from_slots(slots: list) -> list | None:
    """Inverse of _overlay: per-slot expr strings (None where the slot is a
    plain int). Returns None if NO slot is a string (fully static)."""
    out = [s if isinstance(s, str) else None for s in slots]
    return out if any(e is not None for e in out) else None


def compact_from_spec(spec: dict, include_name: bool = False) -> list:
    """Verbose spec dict -> compact JSON entry (lossless)."""
    kind = spec.get("kind", "tensor")
    if kind == "shape":
        return ["S", list(spec["dims"])]
    if kind == "symint":
        hint = spec.get("value", spec.get("hint", 1))
        # A live symint input with an expr -> ['I', hint, expr] (rebindable);
        # a constant symint -> ['sym', hint].
        if spec.get("expr") is not None:
            return ["I", hint, spec["expr"]]
        return ["sym", hint]
    if kind == "scalar":
        return ["sc", spec.get("value")]

    # Symbolic block (dynamic capture): per-slot exprs overlaid onto the hint
    # shape/stride so the serialized entry IS the instantiate_point format
    # (a symbolic slot holds the expr string, static slots the hint int).
    # ONE overlay path for both region and full-graph capture.
    symbolic = spec.get("symbolic") or {}
    shape = _overlay_exprs(list(spec["shape"]), symbolic.get("shape_exprs"))

    entry: list = [shape, _short_dtype(spec["dtype"])]
    opts: dict = {}
    if include_name and spec.get("name"):
        opts["n"] = spec["name"]
    stride = spec.get("stride")
    stride_exprs = symbolic.get("stride_exprs")
    if stride and not _is_contiguous(spec["shape"], stride):
        opts["st"] = _overlay_exprs(list(stride), stride_exprs)
    elif stride_exprs is not None and any(e is not None for e in stride_exprs):
        # Symbolic stride that happens to be contiguous AT THE HINT must
        # still be recorded as exprs — a different binding may break
        # contiguity, and the expr is the exact data.
        opts["st"] = _overlay_exprs(
            list(stride or _contiguous_stride(list(spec["shape"]))),
            stride_exprs)
    # Device default = THE accelerator. Capture pins each worker to one
    # GPU via CUDA_VISIBLE_DEVICES, so any cuda ordinal ("cuda:0") is just
    # "the accelerator" — normalize to bare "cuda" (recording ordinals
    # would fork shape identity on which worker captured the model).
    # Anything non-cuda (cpu RNG state etc.) is a real deviation: recorded.
    dev = str(spec.get("device", "cuda"))
    if dev.startswith("cuda"):
        dev = "cuda"
    if dev != "cuda":
        opts["dev"] = dev
    off = spec.get("storage_offset", 0)
    if off:
        opts["off"] = off
    gen = spec.get("gen") or spec.get("generator")
    if gen:
        cg = _gen_to_compact(gen)
        if cg is not None:
            opts["gen"] = cg
    if spec.get("exact") and spec.get("data") is not None:
        opts["data"] = spec["data"]
    if spec.get("alias_group") is not None:
        # Members of one alias group are views of ONE storage (packed-qkv
        # saved views): generation must allocate a single buffer per group
        # and as_strided each member at its offset.
        opts["alias"] = spec["alias_group"]
    extras = {k: v for k, v in spec.items()
              if k not in _KNOWN_KEYS and v is not None}
    if extras:
        opts["x"] = extras
    if opts:
        entry.append(opts)
    return entry


def spec_from_compact(entry: list, name: str | None = None) -> dict:
    """Compact JSON entry -> verbose spec dict (the in-memory format)."""
    if entry and entry[0] == "S":
        return {"kind": "shape", "name": name, "dims": list(entry[1])}
    if entry and entry[0] == "sym":
        return {"kind": "symint", "name": name, "value": entry[1]}
    if entry and entry[0] == "I":
        # Live symint input: ['I', hint, expr] -> rebindable symint spec.
        spec = {"kind": "symint", "name": name, "value": entry[1]}
        if len(entry) > 2 and entry[2] is not None:
            spec["expr"] = entry[2]
        return spec
    if entry and entry[0] == "sc":
        return {"kind": "scalar", "name": name, "value": entry[1]}

    shape, dtype = list(entry[0]), _long_dtype(entry[1])
    opts = entry[2] if len(entry) > 2 else {}
    raw_stride = list(opts["st"]) if "st" in opts else None
    # Reconstruct the symbolic block from any expr strings in the slots, so
    # the round-trip is lossless and downstream can re-evaluate at any
    # binding. shape/stride themselves keep the expr strings — evaluation
    # (evaluate_symbolic_entry/instantiate_point) is the consumer's job, NOT
    # something we lossily collapse here.
    # Resolve the effective stride: recorded 'st' if present, else the
    # contiguous stride (which is SYMBOLIC when the shape has symbolic dims).
    stride = raw_stride if raw_stride is not None else _contiguous_stride(shape)
    symbolic: dict = {}
    sh_exprs = _exprs_from_slots(shape)
    if sh_exprs is not None:
        symbolic["shape_exprs"] = sh_exprs
    # stride_exprs from whatever stride we ended up with (recorded OR the
    # synthesized symbolic contiguous one) — so evaluate_spec knows to fold
    # it at a binding instead of leaving expr strings in the stride.
    st_exprs = _exprs_from_slots(stride)
    if st_exprs is not None:
        symbolic["stride_exprs"] = st_exprs
    spec: dict = {
        "kind": "tensor",
        "name": name,
        "shape": shape,
        "dtype": dtype,
        "stride": stride,
        "device": opts.get("dev", "cuda"),
        "storage_offset": opts.get("off", 0),
    }
    if symbolic:
        spec["symbolic"] = symbolic
    if "gen" in opts:
        gen = _gen_from_compact(opts["gen"])
        spec["gen"] = gen
        spec["generator"] = gen
    if "data" in opts:
        spec["exact"] = True
        spec["data"] = opts["data"]
    if "alias" in opts:
        spec["alias_group"] = opts["alias"]
    if "n" in opts and name is None:
        spec["name"] = opts["n"]
    spec.update(opts.get("x", {}))
    return spec


def render_T(entry_or_spec) -> str:
    """Human-readable T()/S() rendering of ONE input (documentation only —
    never parsed back; the data is the compact/verbose form)."""
    entry = (compact_from_spec(entry_or_spec)
             if isinstance(entry_or_spec, dict) else entry_or_spec)
    if entry[0] == "S":
        return f"S({entry[1]})"
    if entry[0] == "sym":
        return f"S([{entry[1]}])"
    if entry[0] == "sc":
        return f"Sc({entry[1]})"
    shape, dtype = entry[0], entry[1]
    opts = entry[2] if len(entry) > 2 else {}
    kwargs = []
    if "st" in opts:
        kwargs.append(f"stride={tuple(opts['st'])}")
    if "gen" in opts:
        g = opts["gen"]
        if g[0] == "index":
            kwargs.append(f"gen=Index({g[2]})")
        elif g[0] == "perm":
            kwargs.append(f"gen=Perm({g[1] if len(g) > 1 else ''})")
        elif g[0] == "offsets":
            kwargs.append(f"gen=Offsets({g[1]})")
    suffix = f", {', '.join(kwargs)}" if kwargs else ""
    return f"T({shape}, {dtype}{suffix})"


def render_signature(entries_or_specs: list) -> str:
    """Full human-readable signature line for a list of inputs."""
    return f"({', '.join(render_T(e) for e in entries_or_specs)})"


def _contiguous_stride(shape: list) -> list:
    """Row-major contiguous stride for a shape. Shape slots may be symbolic
    (expr strings) — then the stride is symbolic too, built as sympy product
    expr strings (NOT int()'d, which crashed on a symbol). A fully-int shape
    yields the usual int stride."""
    if any(isinstance(d, str) for d in shape):
        import sympy
        exprs = [_sympify_expr(d) if isinstance(d, str) else sympy.Integer(d)
                 for d in shape]
        stride = [None] * len(shape)
        acc = sympy.Integer(1)
        for i in range(len(shape) - 1, -1, -1):
            stride[i] = int(acc) if acc.is_Integer else str(acc)
            acc = acc * exprs[i]
        return stride
    stride = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        stride[i] = stride[i + 1] * max(int(shape[i + 1]), 1)
    return stride


def _is_contiguous(shape: list, stride: list) -> bool:
    return list(stride) == _contiguous_stride(shape)


# ---------------------------------------------------------------------------
# Dynamic shapes: symbolic dims/strides + symbol tables (wave 2)
#
# A dynamic point's entries may carry EXPR STRINGS in shape/stride slots and
# ["I", hint, expr] entries for live symint inputs:
#
#     {"symbols": {"s16": {"hint": 16, "range": [2, null]}},
#      "guards": ["Eq(Mod(s0, 128), 0)"],
#      "points": [{"bindings": {"s16": 16},
#                  "inputs": [[[64, 64, "s16", "s82"], "f32",
#                              {"st": ["64*s16*s82", "s16*s82", "s82", 1]}],
#                             ["I", 256, "s16*s82"]]}]}
#
# Static entries are unchanged (ints stay ints); a fully static point has
# no symbols/bindings and no string dims. Exprs are sympy-printable strings
# referencing the symbol table; they are data, evaluated by
# evaluate_symbolic_entry under a binding — never eval()'d as Python.
# ---------------------------------------------------------------------------

_TORCH_SYMPY_LOCALS = None


def _sympy_locals():
    """torch._sympy.functions name -> class map, so str(SymInt) forms like
    `PythonMod(s0,128)`, `CeilToInt(s0/3)`, `ModularIndexing(...)`,
    `TruncToInt(ToFloat(s0)*0.5)` re-parse to the ACTUAL torch sympy classes
    (which fold to a concrete Integer/Bool on substitution) instead of opaque
    undefined Functions (which never reduce -> silent guard-accept / loud
    'not an int'). Empty dict if torch isn't importable (pure-int exprs still
    work). Computed once."""
    global _TORCH_SYMPY_LOCALS
    if _TORCH_SYMPY_LOCALS is None:
        loc: dict = {}
        try:
            import torch.utils._sympy.functions as _f
            for n in dir(_f):
                obj = getattr(_f, n)
                if isinstance(obj, type):
                    loc[n] = obj
        except Exception:
            pass
        _TORCH_SYMPY_LOCALS = loc
    return _TORCH_SYMPY_LOCALS


def _sympify_expr(text):
    """Parse a captured expr string to a sympy expr. Two things make this
    faithful vs a bare sympy.sympify: (1) the torch-functions locals map
    (above), so torch's shape functions fold; (2) every free symbol is
    declared integer & positive — a SymInt shape dim is a positive integer,
    and without this Max(1, s//k), floor(s)==s, etc. don't simplify and
    sympy.equals can even raise 'nan not comparable'."""
    import sympy
    expr = sympy.sympify(text, rational=False, locals=_sympy_locals())
    if getattr(expr, "free_symbols", None):
        expr = expr.subs({s: sympy.Symbol(s.name, integer=True, positive=True)
                          for s in expr.free_symbols})
    return expr


def is_symbolic_entry(entry) -> bool:
    """True if a compact entry contains symbolic dims/strides or is symint."""
    if not isinstance(entry, list) or not entry:
        return False
    if entry[0] == "I":
        return True
    if entry[0] == "S" and len(entry) > 1 and isinstance(entry[1], list):
        return any(isinstance(d, str) for d in entry[1])
    if isinstance(entry[0], list):
        if any(isinstance(d, str) for d in entry[0]):
            return True
        opts = entry[2] if len(entry) > 2 else {}
        if isinstance(opts, dict) and any(
                isinstance(s, str) for s in opts.get("st", [])):
            return True
    return False


def _eval_dim(dim, bindings: dict):
    """Evaluate one dim/stride slot: int passes through; expr string is
    sympified against the bindings. Raises on unbound symbols, and LOUD if
    the expr does not fold to a concrete int (a torch function we couldn't
    evaluate — never silently return a wrong value)."""
    if isinstance(dim, int):
        return dim

    expr = _sympify_expr(dim)
    free = expr.free_symbols
    missing = [str(s) for s in free if str(s) not in bindings]
    if missing:
        raise ValueError(f"unbound symbols {missing} in dim expr {dim!r}")
    val = expr.subs({s: bindings[s.name] for s in free})
    if not getattr(val, "is_Integer", False):
        raise ValueError(f"dim expr {dim!r} did not evaluate to an int "
                         f"under {bindings} (got {val!r}); an unrecognized "
                         "symbolic function may not have folded")
    return int(val)


def validate_bindings(symbols: dict, bindings: dict,
                      guards: list | None = None) -> None:
    """Check bindings against symbol ranges and residual guards. LOUD on
    violation — benchmarking an impossible shape answers no question. A guard
    that does not fold to a concrete True/False under the binding is itself
    LOUD (an unevaluated torch function must never be read as 'satisfied')."""
    import sympy

    for name, val in bindings.items():
        sym = symbols.get(name)
        if sym is None:
            raise ValueError(f"binding for unknown symbol {name!r} "
                             f"(table has {sorted(symbols)})")
        lo, hi = (sym.get("range") or [None, None])
        if lo is not None and val < lo:
            raise ValueError(f"{name}={val} below range min {lo}")
        if hi is not None and val > hi:
            raise ValueError(f"{name}={val} above range max {hi}")
    for g in guards or []:
        expr = _sympify_expr(g)
        # A guard that references symbols NOT in this binding is about a
        # DIFFERENT point (guards are a shared graph-level list); it is not
        # applicable here — skip it (the benign cross-point case). Only a
        # guard whose every symbol IS bound must be certified.
        if any(s.name not in bindings for s in expr.free_symbols):
            continue
        sub = expr.subs({s: bindings[s.name] for s in expr.free_symbols})
        if sub is sympy.false or sub == sympy.false:
            raise ValueError(f"binding {bindings} violates guard {g!r}")
        # Fully bound but did NOT collapse to concrete True -> an opaque
        # torch function never folded. Refuse rather than silently accept
        # (the interpolate TruncToInt class: a guard that never reduced to
        # sympy.false used to pass every binding).
        if not (sub is sympy.true or sub == sympy.true):
            raise ValueError(
                f"guard {g!r} did not reduce to a concrete bool under "
                f"{bindings} (got {sub!r}); an unrecognized symbolic function "
                "may not have folded — cannot certify the binding")


def evaluate_symbolic_entry(entry: list, bindings: dict) -> list:
    """Evaluate a (possibly symbolic) compact entry to a fully static one.

    ["I", hint, expr] -> ["sc", value] semantics are NOT applied here; symint
    entries evaluate to ["sym", value] (int input to forward). Tensor
    entries get every shape/stride expr evaluated. Static entries pass
    through unchanged.
    """
    if not is_symbolic_entry(entry):
        return entry
    if entry[0] == "I":
        hint = entry[1]
        expr = entry[2] if len(entry) > 2 else None
        return ["sym", _eval_dim(expr, bindings) if expr is not None else hint]
    if entry[0] == "S":
        # lifted shape param with symbolic dims: evaluate each slot
        return ["S", [_eval_dim(d, bindings) for d in entry[1]]]
    shape = [_eval_dim(d, bindings) for d in entry[0]]
    out = [shape, entry[1]]
    opts = dict(entry[2]) if len(entry) > 2 else {}
    if "st" in opts:
        opts["st"] = [_eval_dim(s, bindings) for s in opts["st"]]
    if opts:
        out.append(opts)
    return out


def evaluate_spec(spec: dict, bindings: dict) -> dict:
    """Evaluate a VERBOSE spec dict's symbolic shape/stride at a binding,
    returning a concrete spec (symbolic dims/strides -> ints, 'symbolic'
    block dropped). Static specs pass through. A symint spec with an 'expr'
    resolves to {'kind':'symint','value':...}. Reuses _eval_dim — one
    evaluator for compact entries AND verbose specs."""
    if spec.get("kind") == "symint" and spec.get("expr") is not None:
        out = dict(spec)
        out["value"] = _eval_dim(spec["expr"], bindings)
        out.pop("expr", None)
        return out
    symbolic = spec.get("symbolic")
    if not symbolic:
        return spec
    out = dict(spec)
    out.pop("symbolic", None)
    if symbolic.get("shape_exprs"):
        out["shape"] = [_eval_dim(d, bindings)
                        for d in _overlay_exprs(list(spec["shape"]),
                                                symbolic["shape_exprs"])]
    if symbolic.get("stride_exprs"):
        base = list(spec.get("stride") or [])
        out["stride"] = [_eval_dim(d, bindings)
                         for d in _overlay_exprs(base, symbolic["stride_exprs"])]
    return out


def instantiate_point(point: dict, symbols: dict,
                      bindings: dict | None = None,
                      guards: list | None = None) -> list:
    """Materialize a (possibly dynamic) point's entries at a binding.

    bindings=None -> the point's own recorded bindings, else the symbol
    hints (the static snapshot — default behavior matches static corpus).
    Returns fully static compact entries ready for spec_from_compact.
    """
    eff = dict(point.get("bindings") or
               {k: v.get("hint") for k, v in (symbols or {}).items()})
    if bindings:
        eff.update(bindings)
    if eff:
        validate_bindings(symbols or {}, eff, guards)
    return [evaluate_symbolic_entry(e, eff) for e in point.get("inputs", [])]
