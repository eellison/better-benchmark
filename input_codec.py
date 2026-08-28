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

import ast
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


def _symbol_point_value(definition: dict | None):
    """Observed concrete value for one family symbol, if any.

    Backed ShapeEnv symbols call this a hint. Unbacked symbols have no guarding
    hint, so their independently observed runtime value is stored explicitly.
    Optimization hints and range bounds are never point values.
    """
    definition = definition or {}
    key = "observed_value" if definition.get("unbacked") is True else "hint"
    return definition.get(key)


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
    # A symbolic storage_offset overrides the (hint) int: emit the expr so it
    # rebinds (review R4 Finding 2). offset_expr lives in the symbolic block.
    off_expr = (spec.get("symbolic") or {}).get("offset_expr")
    if off_expr is not None:
        opts["off"] = off_expr
    elif off:
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
    # A symbolic storage_offset is an expr string in 'off'; lift it into the
    # symbolic block (offset_expr) and keep storage_offset numeric (0 — the
    # real value comes from evaluate_spec at a binding).
    raw_off = opts.get("off", 0)
    if isinstance(raw_off, str):
        symbolic["offset_expr"] = raw_off
        storage_offset = 0
    else:
        storage_offset = raw_off
    spec: dict = {
        "kind": "tensor",
        "name": name,
        "shape": shape,
        "dtype": dtype,
        "stride": stride,
        "device": opts.get("dev", "cuda"),
        "storage_offset": storage_offset,
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


# AST node types permitted in a captured shape/guard expr string. This is the
# security boundary for _sympify_expr: sympy.sympify() eval()s its argument
# (documented-unsafe on untrusted input), and these expr strings arrive from
# DATA artifacts (shapes.json guards/dims/strides, full-graph annotations) that
# are never otherwise executed as Python. A poisoned artifact must not run code
# on load/merge/bench — so we validate the string against the pure
# numeric/relational-expression grammar BEFORE sympify sees it. Barred:
# attribute access, subscripting, string/bytes literals, dunder names, and
# calls to anything outside the sympy/torch shape-function set — the pieces
# every known escape needs (__import__('os').system(...) needs a str literal +
# a dunder; the ().__class__.__subclasses__() gadget needs attribute + subscript
# access). builtins-stripping alone is NOT enough (the subclass walk escapes an
# empty-__builtins__ eval), which is why this is a structural allowlist.
_SAFE_EXPR_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call,
    ast.Name, ast.Load, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not, ast.Invert,
    ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.BitAnd, ast.BitOr, ast.BitXor,
)


# Pure sympy helper FUNCTIONS (not classes) that construct expressions with no
# side effects and no code generation. Everything else legitimate in a shape
# expr is a Basic-subclass constructor (Eq, Mod, Max, Abs, floor, Add, ...) and
# is admitted structurally below; this tiny curated set is only for the handful
# of expr builders sympy exposes as plain functions.
_SAFE_SYMPY_FUNCS = frozenset({"sqrt", "cbrt", "root"})


def _call_name_is_allowed(name: str) -> bool:
    """A call target is legitimate only if it is a torch shape-function class
    (_sympy_locals), a sympy EXPRESSION-CONSTRUCTOR class (a subclass of
    sympy.Basic: Eq, Mod, Max, Abs, floor, And, Add, Mul, Pow, ...), or one of
    a tiny curated set of pure expression-building functions (sqrt/cbrt/root).

    NOT "any callable sympy exports": sympy.sympify() eval()s the expr, so a
    side-effecting or code-generating export reached as a call head IS code
    execution on a poisoned DATA artifact. The str-literal/attribute/subscript
    bans do NOT contain this — sympy.preview(s0) spawns an external viewer,
    plot(s0) starts a backend, init_session()/init_printing() mutate global
    state, and lambdify(s0, s0) compiles generated source, all with only a bare
    symbol argument. Restricting to Basic-subclass constructors (plus the
    curated pure functions) is the structural fix: those meta-functions are
    plain functions, not Basic subclasses, and are not in the curated set."""
    if name in _sympy_locals():
        return True
    import sympy
    obj = getattr(sympy, name, None)
    if obj is None:
        return False
    if getattr(obj, "__module__", "").split(".")[0] != "sympy":
        return False
    if isinstance(obj, type) and issubclass(obj, sympy.Basic):
        return True
    return name in _SAFE_SYMPY_FUNCS and callable(obj)


def _assert_safe_expr(text: str) -> None:
    """Raise ValueError unless `text` is a pure numeric/relational expression
    over symbols and sympy/torch shape functions. Fail-closed and LOUD — never
    silently strips (that would be lossy). See _SAFE_EXPR_NODES for the why."""
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"expr {text!r} is not a parseable expression: {e}")
    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_EXPR_NODES):
            raise ValueError(
                f"disallowed {type(node).__name__} in expr {text!r} (only "
                "arithmetic/relational expressions over symbols and sympy/torch "
                "shape functions are permitted)")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            # bool is an int subclass (True/False ok); str/bytes/complex/None barred.
            raise ValueError(
                f"disallowed {type(node.value).__name__} literal in expr {text!r}")
        if isinstance(node, ast.Name) and "__" in node.id:
            raise ValueError(f"dunder name {node.id!r} in expr {text!r}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError(f"non-name call target in expr {text!r}")
            if node.keywords:
                raise ValueError(f"keyword args not allowed in expr {text!r}")
            if not _call_name_is_allowed(node.func.id):
                raise ValueError(
                    f"call to non-whitelisted function {node.func.id!r} in "
                    f"expr {text!r}")


# The other data-channel eval boundary. Shape-config strings (the T()/S() forms
# in shapes.txt / _shapes_config / shapes.json compact entries) are parsed by
# raw eval() in the harnesses with a fixed namespace of data constructors
# (T/S/Index/Perm) and dtype tokens, under __builtins__: {}. Those strings are
# DATA — a poisoned corpus artifact must not run code when a repro loads its
# inputs. The empty-builtins guard is NOT sufficient on its own: the
# ().__class__.__base__.__subclasses__()[...] gadget walks from a bare literal
# back to os.system without ever naming a builtin. So we structurally forbid the
# pieces every escape needs — attribute access, subscripting, dunder names, and
# calls to anything but the four data constructors. This is the config-grammar
# analogue of _SAFE_EXPR_NODES; it is deliberately a DIFFERENT allowlist because
# the grammar differs (data constructors with keyword args + list/tuple/str
# literals, NOT a relational expression). Arithmetic nodes are permitted because
# they are provably code-exec-free on validated numeric operands and cost
# nothing — the boundary is the attribute/subscript/dunder/call-target bans.
_SAFE_CONFIG_NODES = (
    ast.Expression, ast.Tuple, ast.List, ast.Call, ast.keyword,
    ast.Name, ast.Load, ast.Constant, ast.Attribute,
    ast.UnaryOp, ast.BinOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
)

# The ONLY callables reachable in a shape-config eval namespace. Restricting call
# targets to these four means that even if a name somehow resolves, nothing but a
# data constructor can be invoked.
_SAFE_CONFIG_CALLS = frozenset({"T", "S", "Index", "Perm"})


def _assert_safe_shape_config(text: str) -> None:
    """Raise ValueError unless `text` is a shape-config expression — a tuple/list
    of T()/S()/Index()/Perm() constructor calls over int/float/str literals and
    dtype-token names. Fail-closed and LOUD, never silently strips. This is the
    load-time code-injection boundary for the raw eval() sites that instantiate
    repro inputs from shapes data. See _SAFE_CONFIG_NODES for the threat model."""
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"shape config {text!r} is not a parseable expression: {e}")
    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_CONFIG_NODES):
            raise ValueError(
                f"disallowed {type(node).__name__} in shape config {text!r} (only "
                "T()/S()/Index()/Perm() constructor calls over numeric/string "
                "literals and dtype names are permitted)")
        if isinstance(node, ast.Constant) and not isinstance(
                node.value, (int, float, str, type(None))):
            # bool is an int subclass (ok); bytes/complex barred.
            raise ValueError(
                f"disallowed {type(node.value).__name__} literal in shape config {text!r}")
        if isinstance(node, ast.Name) and "__" in node.id:
            raise ValueError(f"dunder name {node.id!r} in shape config {text!r}")
        if isinstance(node, ast.Attribute):
            # Only `torch.<dtype>` (e.g. torch.complex64) — the sole attribute
            # access real configs use, for dtypes without a short token. Require
            # the base to be exactly the Name `torch` and the attr to be a
            # non-dunder: this admits torch.float32/complex64/... and NOTHING
            # else. The subclass-walk gadget (().__class__.__base__...) chains
            # attributes off a literal/attribute, so its .value is never
            # Name('torch') — it stays barred.
            if not (isinstance(node.value, ast.Name) and node.value.id == "torch"):
                raise ValueError(
                    f"disallowed attribute access in shape config {text!r} "
                    "(only torch.<dtype> is permitted)")
            if "__" in node.attr:
                raise ValueError(f"dunder attribute {node.attr!r} in shape config {text!r}")
        if isinstance(node, ast.keyword) and node.arg is None:
            raise ValueError(f"**kwargs splat not allowed in shape config {text!r}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError(f"non-name call target in shape config {text!r}")
            if node.func.id not in _SAFE_CONFIG_CALLS:
                raise ValueError(
                    f"call to non-constructor {node.func.id!r} in shape config "
                    f"{text!r} (only {sorted(_SAFE_CONFIG_CALLS)} permitted)")


def _sympify_expr(text, symbols: dict | None = None):
    """Parse a captured expr string to a sympy expr. Two things make this
    faithful vs a bare sympy.sympify: (1) the torch-functions locals map
    (above), so torch's shape functions fold; (2) free symbols are declared
    integer with a sign assumption DRIVEN BY THE RANGE FLOOR — positive when
    the symbol's range lower bound >= 1, else nonnegative (a size is >= 0).
    Without an integer assumption Max(1, s//k)/floor(s)==s don't simplify and
    sympy.equals can raise 'nan not comparable'; but a blanket `positive`
    is WRONG for a zero-capable symbol (unbacked u0, range [0, ...]) — it
    folds Ne(u0,0)->True / Eq(u0,0)->False, dropping a guard / rejecting a
    valid 0 binding. `symbols` is the {name: {"range":[lo,hi]}} table; when
    absent we assume nonnegative (the safe floor for any shape dim).

    A STRING `text` is validated by _assert_safe_expr first: sympify() eval()s
    it and these strings come from data artifacts, so the safe-grammar gate is
    the load-time code-injection boundary (non-str input — a sympy expr/int —
    skips the gate; sympify does not eval those)."""
    import sympy
    if isinstance(text, str):
        _assert_safe_expr(text)
    expr = sympy.sympify(text, rational=False, locals=_sympy_locals())
    free = getattr(expr, "free_symbols", None)
    if free:
        subs = {}
        for s in free:
            lo = None
            if symbols and s.name in symbols:
                rng = symbols[s.name].get("range") or [None, None]
                lo = rng[0]
            if lo is not None and lo >= 1:
                subs[s] = sympy.Symbol(s.name, integer=True, positive=True)
            else:
                subs[s] = sympy.Symbol(s.name, integer=True, nonnegative=True)
        expr = expr.subs(subs)
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
        if isinstance(opts, dict):
            if any(isinstance(s, str) for s in opts.get("st", [])):
                return True
            if isinstance(opts.get("off"), str):   # symbolic storage_offset
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


def binding_violation(symbols: dict, bindings: dict,
                      guards: list | None = None) -> str | None:
    """Return a reason string for the first range/guard violation under
    `bindings`, or None if the binding is valid. Pure predicate — never
    raises — so callers can SEARCH for a valid binding (e.g. perturbing to a
    distinct warmup shape) without exceptions for control flow. The loud
    authority validate_bindings() wraps this and raises the same reason."""
    import sympy

    for name, val in bindings.items():
        sym = symbols.get(name)
        if sym is None:
            return (f"binding for unknown symbol {name!r} "
                    f"(table has {sorted(symbols)})")
        if not isinstance(val, int):
            # A hint-less / unresolved symbol can reach here as None (e.g.
            # instantiate_point on a symbols entry with no hint). Report it as a
            # violation rather than raising `None < lo` — this predicate is
            # documented never to raise so callers can search for a valid
            # binding without exceptions for control flow.
            return f"binding {name}={val!r} is not an int"
        lo, hi = (sym.get("range") or [None, None])
        if lo is not None and val < lo:
            return f"{name}={val} below range min {lo}"
        if hi is not None and val > hi:
            return f"{name}={val} above range max {hi}"
    for g in guards or []:
        # Pass the symbol table so sign assumptions come from each symbol's
        # range floor (a zero-capable unbacked symbol must NOT be assumed
        # positive — that drops Ne(u0,0) / rejects a valid u0=0).
        expr = _sympify_expr(g, symbols)
        # A guard that references symbols NOT in this binding is about a
        # DIFFERENT point (guards are a shared graph-level list); it is not
        # applicable here — skip it (the benign cross-point case). Only a
        # guard whose every symbol IS bound must be certified.
        if any(s.name not in bindings for s in expr.free_symbols):
            continue
        sub = expr.subs({s: bindings[s.name] for s in expr.free_symbols})
        if sub is sympy.false or sub == sympy.false:
            return f"binding {bindings} violates guard {g!r}"
        # Fully bound but did NOT collapse to concrete True -> an opaque
        # torch function never folded. Refuse rather than silently accept
        # (the interpolate TruncToInt class: a guard that never reduced to
        # sympy.false used to pass every binding).
        if not (sub is sympy.true or sub == sympy.true):
            return (f"guard {g!r} did not reduce to a concrete bool under "
                    f"{bindings} (got {sub!r}); an unrecognized symbolic "
                    "function may not have folded — cannot certify the binding")
    return None


def bindings_satisfy(symbols: dict, bindings: dict,
                     guards: list | None = None) -> bool:
    """True iff `bindings` satisfies every symbol range and applicable guard.
    Non-raising sibling of validate_bindings — for search/perturbation."""
    return binding_violation(symbols, bindings, guards) is None


def validate_bindings(symbols: dict, bindings: dict,
                      guards: list | None = None) -> None:
    """Check bindings against symbol ranges and residual guards. LOUD on
    violation — benchmarking an impossible shape answers no question. A guard
    that does not fold to a concrete True/False under the binding is itself
    LOUD (an unevaluated torch function must never be read as 'satisfied')."""
    reason = binding_violation(symbols, bindings, guards)
    if reason is not None:
        raise ValueError(reason)


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
    # storage_offset can also be symbolic (a view at a symbolic offset);
    # evaluate it too, else it stays an unevaluated expr string and the
    # tensor is as_strided at the wrong offset (review R4 Finding 2).
    if isinstance(opts.get("off"), str):
        opts["off"] = _eval_dim(opts["off"], bindings)
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
    if symbolic.get("offset_expr"):
        out["storage_offset"] = _eval_dim(symbolic["offset_expr"], bindings)
    return out


def instantiate_point(point: dict, symbols: dict,
                      bindings: dict | None = None,
                      guards: list | None = None) -> list:
    """Materialize a (possibly dynamic) point's entries at a binding.

    bindings=None -> the point's own recorded bindings, else the symbol
    hints (the static snapshot — default behavior matches static corpus).
    Returns fully static compact entries ready for spec_from_compact.
    """
    eff = dict(point.get("bindings") or {
        name: value
        for name, definition in (symbols or {}).items()
        if (value := _symbol_point_value(definition)) is not None
    })
    if bindings:
        eff.update(bindings)
    missing = sorted(set(symbols or {}) - set(eff))
    if missing:
        raise ValueError(
            f"point needs explicit bindings for unobserved symbols {missing}")
    if eff:
        validate_bindings(symbols or {}, eff, guards)
    return [evaluate_symbolic_entry(e, eff) for e in point.get("inputs", [])]
