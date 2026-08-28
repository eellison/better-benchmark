"""Dynamic-family contract decoding and ShapeEnv replay.

Live capture happens in capture_hook.py. That path records symbols, ranges,
guards, and symbolic input expressions from the graph's live ShapeEnv. This
module only consumes that serialized contract when a repro is materialized or
compiled; it must not invent, widen, or silently drop captured constraints.

Keeping this code separate makes the ownership boundary explicit:
  capture_hook -> capture
  merge_captures -> family/point persistence
  dynamic_shape_replay -> contract reconstruction
  repro_harness -> benchmark orchestration
"""

import json
import math
from pathlib import Path
from typing import Any

import torch

from input_codec import _symbol_point_value


def _parse_bind_args(bind_args: list | None, flag_name: str = "--bind") -> list:
    """Parse repeated --bind values into a list of binding dicts.

    Each element is one --bind occurrence, e.g. "s16=24,s82=24" ->
    {"s16": 24, "s82": 24}. Returns [] when bind_args is None/empty.
    Malformed entries raise ValueError (loud beats benchmarking a typo).

    ``flag_name`` names the flag in error messages: the identical
    symbol=int grammar also backs --prewarm, and attributing a --prewarm
    typo to --bind sends the reader to the wrong argument.
    """
    out = []
    for raw in bind_args or []:
        bindings = {}
        for part in str(raw).split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(
                    f"{flag_name} entry {part!r} must be symbol=int (e.g. s16=24)")
            name, _, val = part.partition("=")
            name = name.strip()
            if name in bindings:
                # A duplicated symbol in one binding is far more likely
                # a typo than intentional last-value-wins.
                raise ValueError(
                    f"{flag_name} binding {raw!r} assigns {name!r} twice")
            try:
                bindings[name] = int(val)
            except ValueError:
                raise ValueError(
                    f"{flag_name} value for {name!r} must be an int, "
                    f"got {val!r}") from None
        if not bindings:
            raise ValueError(f"{flag_name} {raw!r} parsed to no bindings")
        out.append(bindings)
    return out


def _resolve_bound_configs(repro_file: str, bindings_list: list,
                          shape: str | None = None) -> list:
    """Resolve (label, binding, config) rows for --bind/--dynamic benching.

    One row per (binding set x shape config): each binding set is threaded
    through load_shape_configs(symbol_bindings=...), which instantiates
    every symbolic point at that binding (range/guard violations raise).
    binding=None rows instantiate at each point's recorded bindings (the
    captured hint — what a plain run measures). `shape` filters to one
    named config; unknown names raise.
    """
    # Local import avoids a module cycle: repro_harness owns config decoding
    # and re-exports these replay helpers for existing callers.
    from repro_harness import load_shape_configs

    rows = []
    for binding in (bindings_list or [None]):
        configs = load_shape_configs(repro_file, symbol_bindings=binding)
        if shape is not None:
            if shape not in configs:
                raise ValueError(
                    f"--shape {shape!r} not in configs "
                    f"(have {sorted(configs)})")
            configs = {shape: configs[shape]}
        for label, cfg in configs.items():
            rows.append((label, binding, cfg))
    if not rows:
        raise ValueError(
            f"--bind/--dynamic found no shape configs for {repro_file} "
            "(needs a shapes.json next to the repro)")
    return rows


def format_binding(binding: dict | None) -> str:
    """Human/key form of a binding dict: 's16=24,s82=24' or 'hint'."""
    if not binding:
        return "hint"
    return ",".join(f"{k}={v}" for k, v in sorted(binding.items()))


def _parse_freeze_args(freeze_args: list, symbols: dict) -> dict:
    """Parse repeated --freeze values ('s0' or 's0=8', comma-separable) into
    {name: int}. A bare name freezes at the symbol's observed point value.
    Unknown symbols, non-int values, and value-less bare freezes error loudly —
    silently freezing the wrong thing benches the wrong artifact."""
    frozen: dict = {}
    for raw in freeze_args or []:
        for part in str(raw).split(","):
            part = part.strip()
            if not part:
                continue
            name, eq, val = part.partition("=")
            name = name.strip()
            if name not in symbols:
                raise ValueError(
                    f"--freeze names unknown symbol {name!r} "
                    f"(family has {sorted(symbols)})")
            if eq:
                try:
                    fval = int(val)
                except ValueError:
                    raise ValueError(
                        f"--freeze value for {name!r} must be an int, "
                        f"got {val!r}") from None
                if name in frozen and frozen[name] != fval:
                    # Identical repeats are harmless; conflicts are typos.
                    raise ValueError(
                        f"--freeze assigns {name!r} conflicting values "
                        f"{frozen[name]} and {fval}")
                frozen[name] = fval
            else:
                hint = _symbol_point_value(symbols[name])
                if not isinstance(hint, int) or isinstance(hint, bool):
                    raise ValueError(
                        f"--freeze {name} uses the observed point value, but "
                        f"the symbols table has none for {name!r}; "
                        f"freeze explicitly with --freeze {name}=N")
                if name in frozen and frozen[name] != hint:
                    raise ValueError(
                        f"--freeze assigns {name!r} conflicting values "
                        f"{frozen[name]} and {hint} (table hint)")
                frozen[name] = hint
    return frozen


def _recorded_point_bindings(repro_file: str, overlay: dict | None = None) -> list:
    """Every recorded dynamic point's bindings, in point order, with the
    optional `overlay` (frozen symbols) merged over each and duplicates
    (post-overlay) removed. Used when --freeze runs without explicit
    --run-at bindings: binding=None rows can't express the overlay."""
    shapes_json = Path(repro_file).parent / "shapes.json"
    if not shapes_json.exists():
        return []
    data = json.loads(shapes_json.read_text())
    symbols = set(data.get("symbols") or {})
    out, seen = [], set()
    for point in data.get("points", []):
        if not point.get("captured_dynamic"):
            continue
        b = dict(point.get("bindings") or {})
        b.update(overlay or {})
        if not b or not symbols.issubset(b):
            continue
        key = tuple(sorted(b.items()))
        if key not in seen:
            seen.add(key)
            out.append(b)
    return out


def _expr_free_names(dim) -> set:
    """Free symbol NAMES of a dim/stride slot: int -> set(), bare name ->
    {name}, composite expr -> its sympy free symbols (via the shared safe
    grammar). Used by the --freeze engine to decide whether a dim still
    depends on a non-frozen symbol."""
    if not isinstance(dim, str):
        return set()
    d = dim.strip()
    if d.isidentifier():
        return {d}
    from input_codec import _sympify_expr
    expr = _sympify_expr(d)
    return {s.name for s in getattr(expr, "free_symbols", ())}


def _eval_shape_env_expr(text: str, values: dict[str, Any]):
    """Interpret a captured expression over spec IntVars using PyTorch core.

    ``_sympify_expr`` is the existing fail-closed data boundary.  Core's
    ``sympy_interp(PythonReferenceAnalysis, ...)`` then performs the same
    SymPy-to-symbolic-operator playback used by FX/runtime-assert machinery;
    unsupported shape functions fail loudly.
    """
    from input_codec import _sympify_expr
    from torch.utils._sympy.interp import sympy_interp
    from torch.utils._sympy.reference import PythonReferenceAnalysis

    expr = _sympify_expr(text)
    env = {symbol: values[symbol.name] for symbol in expr.free_symbols}
    return sympy_interp(PythonReferenceAnalysis, env, expr)


def _shape_env_spec_for_repro(repro_file: str,
                              frozen: dict | None = None):
    """Build the native ``ShapesSpec`` for a captured dynamic family.

    Every family symbol is represented by one shared ``IntVar``.  The same
    value is used in tensor dims, lifted SymInt arguments, derived expressions,
    and residual guards.  Live root values are appended as private wrapper
    arguments so even a composite-only/stride-only symbol has a bare source in
    ShapeEnv; relations and guards therefore become runtime assertions rather
    than metadata checked only by the harness.

    Returns ``(spec, live_symbol_names, original_input_count,
    metadata_checks, frozen_symbols)``.
    """
    try:
        from torch.fx.experimental.dynamic_spec import (
            IntVar, SeqSpec, ShapesSpec, TensorSpec,
        )
    except ImportError as exc:
        raise RuntimeError(
            "faithful dynamic replay needs "
            "torch.fx.experimental.dynamic_spec.ShapesSpec so captured "
            "ranges/guards can be restored into ShapeEnv") from exc

    shapes_path = Path(repro_file).parent / "shapes.json"
    if not shapes_path.exists():
        raise ValueError(
            f"dynamic ShapeEnv replay needs {shapes_path}")
    data = json.loads(shapes_path.read_text())
    symbols = data.get("symbols") or {}
    frozen = dict(frozen or {})
    live_names = tuple(sorted(set(symbols) - set(frozen)))
    if not live_names:
        raise ValueError(
            "ShapeEnv dynamic replay has no live symbols; use --static when "
            "every family symbol is frozen")

    values: dict[str, Any] = dict(frozen)
    for name in live_names:
        meta = symbols[name] or {}
        lo, hi = meta.get("range") or [None, None]
        hint = _symbol_point_value(meta)
        optimization_hint = meta.get("optimization_hint")
        for field, val in (("range lower", lo), ("range upper", hi),
                           ("hint", hint),
                           ("optimization hint", optimization_hint)):
            if val is not None and (
                    not isinstance(val, int) or isinstance(val, bool)):
                raise ValueError(
                    f"symbol {name!r} has non-int {field} {val!r}")
        if not meta.get("unbacked"):
            optimization_hint = hint
        values[name] = IntVar(
            name, min=lo, max=hi,
            optimization_hint=optimization_hint)

    point = next(
        (p for p in data.get("points", [])
         if p.get("captured_dynamic") and p.get("inputs") is not None),
        None,
    )
    if point is None:
        raise ValueError(
            "dynamic ShapeEnv replay needs a captured_dynamic point with "
            "structured inputs in shapes.json")
    entries = point["inputs"]
    assumptions = []
    metadata_checks = []

    def _slot(value):
        """Map a serialized slot directly to its native spec expression.

        ShapesSpec supports derived expressions as leaf specs. Core creates
        and asserts the required runtime relationship, so replay does not
        need to mint a second IntVar plus an equivalent equality assumption.
        """
        if not isinstance(value, str):
            return value
        return _eval_shape_env_expr(value, values)

    specs = []
    from input_codec import _sympify_expr
    for pos, entry in enumerate(entries):
        if (isinstance(entry, list) and entry
                and isinstance(entry[0], list)):
            dims = []
            for dim, value in enumerate(entry[0]):
                dims.append(_slot(value))
            specs.append(TensorSpec(dims))
            options = entry[2] if len(entry) > 2 else {}
            symbolic_metadata = []
            for dim, value in enumerate((options or {}).get("st", [])):
                if isinstance(value, str):
                    symbolic_metadata.append(("stride", dim, value))
            if isinstance((options or {}).get("off"), str):
                symbolic_metadata.append(
                    ("storage_offset", None, options["off"]))
            live_shape_symbols = set().union(
                *(_expr_free_names(value) for value in entry[0])
            ) & set(live_names)
            for kind, dim, expression in symbolic_metadata:
                live_metadata_symbols = (
                    _expr_free_names(expression) & set(live_names))
                if live_metadata_symbols and not live_shape_symbols:
                    # ShapesSpec currently has no stride/offset fields, and
                    # Dynamo specializes tensor metadata when every size is
                    # static. Pretending otherwise would compile one artifact
                    # per stride/offset while reporting faithful reuse.
                    raise ValueError(
                        f"input {pos} has live symbolic {kind} "
                        f"{expression!r} but no live symbolic size; PyTorch "
                        "ShapesSpec cannot make stride/offset metadata dynamic "
                        "for a fully-static-size tensor")
                evaluator = _symint_expr_evaluator(
                    _sympify_expr(expression))
                if evaluator is None:
                    raise ValueError(
                        f"input {pos} {kind} expression {expression!r} uses "
                        "operations the traced metadata invariant cannot "
                        "represent")
                metadata_checks.append((pos, kind, dim, evaluator))
        elif (isinstance(entry, list) and entry and entry[0] == "I"):
            expr = entry[2] if len(entry) > 2 else None
            specs.append(
                _slot(expr)
                if isinstance(expr, str) else None)
        elif (isinstance(entry, list) and len(entry) > 1
              and entry[0] == "S" and isinstance(entry[1], list)):
            shape_specs = []
            for dim, value in enumerate(entry[1]):
                shape_specs.append(_slot(value))
            specs.append(SeqSpec(shape_specs))
        else:
            # Unspecified arguments remain static at their runtime value.
            specs.append(None)

    # Private root args bind every symbol even when it occurs only inside a
    # composite expression.  Repeated use of the same IntVar causes ShapeEnv
    # to assert equality between the root and every tensor/scalar occurrence.
    specs.extend(values[name] for name in live_names)

    for guard in data.get("guards") or []:
        assumption = _eval_shape_env_expr(guard, values)
        if isinstance(assumption, bool):
            if not assumption:
                raise ValueError(
                    f"frozen symbols make captured guard {guard!r} false")
            continue
        if not isinstance(assumption, torch.SymBool):
            raise ValueError(
                f"captured guard {guard!r} did not produce a SymBool "
                f"(got {assumption!r})")
        assumptions.append(assumption)

    return (
        ShapesSpec(
            params={"*args": specs},
            assumptions=assumptions,
        ),
        live_names,
        len(entries),
        metadata_checks,
        frozen,
    )


def _shape_env_repro(inner, original_input_count: int, spec,
                     live_symbol_names=(), metadata_checks=(), frozen=None):
    """Attach the native spec to a fresh wrapper class for this artifact."""
    from torch.fx.experimental.dynamic_spec import dynamic_spec

    class ShapeEnvRepro(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = inner

        @dynamic_spec(spec)
        def forward(self, *args):
            # Private symbol-root args bind composite-only symbols in ShapeEnv;
            # the captured forward still receives its exact original inputs.
            values = dict(frozen or {})
            values.update({
                name: args[original_input_count + i]
                for i, name in enumerate(live_symbol_names)
            })
            # Dynamo only wraps arguments that the Python body reads. Touch
            # every private root so a composite-only symbol is bound before
            # ShapesSpec finalizes assumptions such as dim == 4*s1.
            for name in live_symbol_names:
                torch._check(values[name] == values[name])
            for pos, kind, dim, evaluator in metadata_checks:
                tensor = args[pos]
                actual = (tensor.stride(dim) if kind == "stride"
                          else tensor.storage_offset())
                torch._check(actual == evaluator(values))
            return self.inner(*args[:original_input_count])

    return ShapeEnvRepro()


def _effective_binding_for_row(binding: dict | None, cfg: dict) -> dict:
    """Resolve ``binding=None`` (a recorded point) to its actual bindings."""
    effective = binding if binding is not None else cfg.get("bindings")
    if effective and all(
            isinstance(v, int) and not isinstance(v, bool)
            for v in effective.values()):
        return dict(effective)
    raise ValueError(
        f"cannot resolve concrete bindings for recorded point "
        f"{cfg.get('shape_hash')!r}")


def _symint_expr_evaluator(expr):
    """Compile a captured expression into direct Python/SymInt operations.

    Unlike ``sympy_interp``, the returned closure is safe to inline into a
    Dynamo trace: it performs no parsing or graph-side code generation. This
    matters for backed replay, where a normal Python branch over the resulting
    SymBool becomes a Dynamo guard without inserting a runtime-assert node
    that can change Inductor fusion.

    Returns ``None`` for nodes outside the supported closed grammar.
    """
    import sympy

    if isinstance(expr, sympy.Integer):
        v = int(expr)
        return lambda env: v
    if isinstance(expr, sympy.Float):
        v = float(expr)
        return lambda env: v
    if isinstance(expr, sympy.Rational):
        numerator, denominator = int(expr.p), int(expr.q)
        return lambda env: numerator / denominator
    if isinstance(expr, sympy.Symbol):
        name = expr.name
        return lambda env: env[name]
    parts = [_symint_expr_evaluator(a) for a in expr.args]
    if any(p is None for p in parts):
        return None
    if isinstance(expr, sympy.Add):
        def _add(env):
            r = parts[0](env)
            for p in parts[1:]:
                r = r + p(env)
            return r
        return _add
    if isinstance(expr, sympy.Mul):
        def _mul(env):
            r = parts[0](env)
            for p in parts[1:]:
                r = r * p(env)
            return r
        return _mul
    if isinstance(expr, sympy.Pow) and isinstance(expr.exp, sympy.Integer) \
            and int(expr.exp) >= 0:
        base, e = parts[0], int(expr.exp)
        return lambda env: base(env) ** e
    if isinstance(expr, sympy.Mod):
        return lambda env: parts[0](env) % parts[1](env)
    if isinstance(expr, sympy.Equality):
        return lambda env: parts[0](env) == parts[1](env)
    if isinstance(expr, sympy.Unequality):
        return lambda env: parts[0](env) != parts[1](env)
    if isinstance(expr, sympy.StrictLessThan):
        return lambda env: parts[0](env) < parts[1](env)
    if isinstance(expr, sympy.LessThan):
        return lambda env: parts[0](env) <= parts[1](env)
    if isinstance(expr, sympy.StrictGreaterThan):
        return lambda env: parts[0](env) > parts[1](env)
    if isinstance(expr, sympy.GreaterThan):
        return lambda env: parts[0](env) >= parts[1](env)
    if isinstance(expr, sympy.And):
        def _and(env):
            result = parts[0](env)
            for part in parts[1:]:
                result = result & part(env)
            return result
        return _and
    if isinstance(expr, sympy.Or):
        def _or(env):
            result = parts[0](env)
            for part in parts[1:]:
                result = result | part(env)
            return result
        return _or
    if isinstance(expr, sympy.Not):
        return lambda env: ~parts[0](env)
    if isinstance(expr, sympy.Max):
        def _max(env):
            result = parts[0](env)
            for part in parts[1:]:
                result = torch.sym_max(result, part(env))
            return result
        return _max
    if isinstance(expr, sympy.Min):
        def _min(env):
            result = parts[0](env)
            for part in parts[1:]:
                result = torch.sym_min(result, part(env))
            return result
        return _min

    name = type(expr).__name__
    if name in {"FloorDiv", "CleanDiv"}:
        return lambda env: parts[0](env) // parts[1](env)
    if name in {"Mod", "PythonMod"}:
        return lambda env: parts[0](env) % parts[1](env)
    if name == "ModularIndexing":
        return lambda env: (
            parts[0](env) // parts[1](env)) % parts[2](env)
    if name == "ToFloat":
        return lambda env: torch.sym_float(parts[0](env))
    if name in {"floor", "FloorToInt"}:
        return lambda env: math.floor(parts[0](env))
    if name in {"ceiling", "CeilToInt"}:
        return lambda env: math.ceil(parts[0](env))
    if name == "TruncToInt":
        return lambda env: math.trunc(parts[0](env))
    if name in {"Abs", "AbsMax"}:
        return lambda env: abs(parts[0](env))
    return None


def _backed_replay_plan_for_repro(repro_file: str,
                                   frozen: dict | None = None):
    """Build a kernel-faithful plan when every live symbol is tensor-backed.

    ``ShapesSpec`` is required for genuinely unbacked symbols, but it models
    them with deferred runtime assertions. Those assertion nodes can change an
    otherwise identical Inductor fusion. When capture metadata says every
    live family symbol is backed *and* each has a bare tensor-dimension source,
    mark that exact dimension dynamic, re-derive lifted SymInt/shape arguments
    from it, and express residual relations as ordinary Python branches.
    Dynamo turns those branches into guards, matching the original graph
    boundary.

    Returns ``None`` when a symbol/range/expression cannot be represented by
    this backed path; the caller then uses native ShapesSpec replay.
    """
    from input_codec import _sympify_expr

    shapes_path = Path(repro_file).parent / "shapes.json"
    if not shapes_path.exists():
        return None
    data = json.loads(shapes_path.read_text())
    symbols = data.get("symbols") or {}
    frozen = dict(frozen or {})
    live_names = tuple(sorted(set(symbols) - set(frozen)))
    point = next(
        (p for p in data.get("points", [])
         if p.get("captured_dynamic") and p.get("inputs") is not None),
        None,
    )
    if point is None or not live_names:
        return None
    entries = point["inputs"]

    # Provenance is authoritative. A genuinely unbacked symbol can happen to
    # appear in a tensor shape after a data-dependent operation; that does not
    # make it backed. Older artifacts omit the key for backed symbols, which is
    # the schema convention used by _harvest_shape_env.
    if any((symbols[name] or {}).get("unbacked") is True
           for name in live_names):
        return None

    sources = {}
    for pos, entry in enumerate(entries):
        if not (isinstance(entry, list) and entry
                and isinstance(entry[0], list)):
            continue
        for dim, value in enumerate(entry[0]):
            if value in live_names:
                sources.setdefault(value, (pos, dim))
    if set(sources) != set(live_names):
        return None

    marks = []
    for name in live_names:
        meta = symbols[name] or {}
        lo, hi = meta.get("range") or [None, None]
        if lo == 2 and hi is None:
            bounds = None  # plain mark_dynamic is exactly [2, +inf)
        elif (isinstance(lo, int) and not isinstance(lo, bool)
              and isinstance(hi, int) and not isinstance(hi, bool)
              and lo < hi):
            bounds = (lo, hi)
        else:
            # Half-open 0/1-capable ranges and hint-less ranges cannot be
            # represented safely by mark_dynamic on this torch version.
            return None
        marks.append((*sources[name], bounds))

    def evaluator(text):
        expr = _sympify_expr(text, symbols)
        if any(s.name not in symbols for s in expr.free_symbols):
            return None
        return _symint_expr_evaluator(expr)

    derived = {}
    for pos, entry in enumerate(entries):
        if (isinstance(entry, list) and entry and entry[0] == "I"
                and len(entry) > 2 and isinstance(entry[2], str)):
            fn = evaluator(entry[2])
            if fn is None:
                return None
            derived[pos] = fn
        elif (isinstance(entry, list) and len(entry) > 1
              and entry[0] == "S" and isinstance(entry[1], list)
              and any(isinstance(value, str) for value in entry[1])):
            fns = []
            for value in entry[1]:
                fn = evaluator(value) if isinstance(value, str) else (
                    lambda _env, value=value: value)
                if fn is None:
                    return None
                fns.append(fn)
            derived[pos] = (
                lambda env, fns=tuple(fns): [fn(env) for fn in fns])

    checks = []
    for pos, entry in enumerate(entries):
        if not (isinstance(entry, list) and entry
                and isinstance(entry[0], list)):
            continue
        for dim, value in enumerate(entry[0]):
            if not isinstance(value, str):
                continue
            if sources.get(value) == (pos, dim):
                continue
            fn = evaluator(value)
            if fn is None:
                return None
            checks.append((pos, "size", dim, value, fn))
        options = entry[2] if len(entry) > 2 else {}
        for dim, value in enumerate((options or {}).get("st", [])):
            if isinstance(value, str):
                fn = evaluator(value)
                if fn is None:
                    return None
                checks.append((pos, "stride", dim, value, fn))
        offset = (options or {}).get("off")
        if isinstance(offset, str):
            fn = evaluator(offset)
            if fn is None:
                return None
            checks.append((pos, "storage_offset", None, offset, fn))

    guards = []
    for text in data.get("guards") or []:
        fn = evaluator(text)
        if fn is None:
            return None
        guards.append((text, fn))

    kept = tuple(pos for pos in range(len(entries)) if pos not in derived)
    kept_index = {original: current for current, original in enumerate(kept)}
    if any(pos not in kept_index for pos, _dim, _bounds in marks):
        return None
    return {
        "input_count": len(entries),
        "kept": kept,
        "markings": tuple(
            (kept_index[pos], dim, bounds) for pos, dim, bounds in marks),
        "sources": tuple(
            (name, pos, dim) for name, (pos, dim) in sorted(sources.items())),
        "derived": tuple(sorted(derived.items())),
        "checks": tuple(checks),
        "guards": tuple(guards),
        "frozen": frozen,
    }


def _backed_repro(inner, plan):
    """Reconstruct the captured call from backed tensor dimensions."""
    class BackedRepro(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = inner

        def forward(self, *args):
            full = [None] * plan["input_count"]
            for pos, value in zip(plan["kept"], args):
                full[pos] = value
            values = dict(plan["frozen"])
            for name, pos, dim in plan["sources"]:
                values[name] = full[pos].size(dim)
            for pos, fn in plan["derived"]:
                full[pos] = fn(values)
            for pos, kind, dim, expression, fn in plan["checks"]:
                tensor = full[pos]
                actual = (
                    tensor.size(dim) if kind == "size"
                    else tensor.stride(dim) if kind == "stride"
                    else tensor.storage_offset()
                )
                if actual != fn(values):
                    raise RuntimeError(
                        f"captured {kind} relation {expression!r} failed")
            for text, fn in plan["guards"]:
                # A normal SymBool branch becomes a Dynamo guard. Do not use
                # torch._check/expect_true here: their runtime-assert nodes
                # can split an otherwise fused Inductor kernel.
                if not fn(values):
                    raise RuntimeError(f"captured guard {text!r} failed")
            return self.inner(*full)

    return BackedRepro()


def _backed_args(inputs, plan):
    """Select the compiled call arguments and mark exact backed dimensions."""
    args = [inputs[pos] for pos in plan["kept"]]
    for pos, dim, bounds in plan["markings"]:
        tensor = args[pos]
        if bounds is None:
            torch._dynamo.mark_dynamic(tensor, dim)
        else:
            torch._dynamo.mark_dynamic(
                tensor, dim, min=bounds[0], max=bounds[1])
    return args


def _symbols_and_guards_for_repro(repro_file: str) -> tuple[dict, list]:
    """(symbols_table, guards) from a dynamic repro's shapes.json, or ({}, [])
    for a static repro / shapes.txt. Used to validate generated warmup
    bindings against ranges and guards: a blind name->val+i perturbation can
    break Eq(s0,s1) couplings or range-max pins."""
    shapes_json = Path(repro_file).parent / "shapes.json"
    if not shapes_json.exists():
        return {}, []
    data = json.loads(shapes_json.read_text())
    return data.get("symbols") or {}, data.get("guards") or []


def _distinct_dynamic_bindings(repro_file, rows, n=2,
                               frozen: dict | None = None):
    """Pick up to `n` GUARD-VALID warmup bindings that force inductor past
    0/1/many specialization into the GENERAL dynamic kernel — the one the
    model runs. These are WARMUP shapes (the artifact is timed later at the
    --bind rows), so they are generated fresh from the symbol table, NOT
    seeded from the timed rows (a timed row may have equal dims, e.g. 16x16,
    which makes dynamo specialize on the equality and defeats generalization).

    Two requirements:
      (1) INTERNALLY DISTINCT — within each warmup binding every symbol takes
          a DIFFERENT value, so Dynamo never unifies two dims into one
          square symbol (square specialization == wrong kernel).
      (2) MUTUALLY DISTINCT — across the warmup set every symbol takes >=2
          different values, so the 0/1/many rule generalizes EACH symbol.

    A coupling guard (e.g. Eq(s0,s1)) genuinely forbids internal distinctness:
    the model really runs the square kernel. We never fabricate a binding that
    breaks a guard — when the distinct candidates all violate guards we fall
    back to scaled COUPLED (equal-magnitude) shapes, whose square kernel IS the
    faithful one. Returns >=1 binding (length 1 only if the family is so
    constrained a single shape is the only reachable one)."""
    from input_codec import bindings_satisfy

    symbols, guards = _symbols_and_guards_for_repro(repro_file)
    hint = {
        name: value
        for name, definition in (symbols or {}).items()
        if isinstance(
            (value := _symbol_point_value(definition)), int)
        and not isinstance(value, bool)
    }
    row_bindings = [
        (binding if binding is not None else config.get("bindings")) or {}
        for _label, binding, config in rows
    ]
    for name in symbols:
        if name not in hint:
            value = next(
                (binding[name] for binding in row_bindings
                 if isinstance(binding.get(name), int)
                 and not isinstance(binding.get(name), bool)),
                None,
            )
            if value is not None:
                hint[name] = value
    frozen = dict(frozen or {})
    if frozen:
        # Frozen symbols are compile-time constants: candidates vary
        # only the NON-frozen symbols; the frozen values ride along in every
        # emitted binding and guards over them validate on the merged
        # binding. Distinctness requirements likewise apply to the varying
        # symbols only (a frozen dim is concrete — no symbol to unify).
        hint = {k: v for k, v in hint.items() if k not in frozen}
        if not hint:
            from input_codec import bindings_satisfy as _bs
            return ([dict(frozen)]
                    if _bs(symbols, dict(frozen), guards) else [None])
    missing_seed = sorted(set(symbols) - set(hint) - set(frozen))
    if missing_seed:
        raise ValueError(
            "dynamic warmup needs an explicit binding for symbols without "
            f"an observed point: {missing_seed}")
    if not hint:
        # No symbol table (hand-written dynamic shapes.json) — fall back to
        # the --bind rows' bindings, distinct ones first.
        bs = [b for _l, b, _c in rows if b]
        return bs[:n] if bs else [None]

    names = sorted(hint)

    def valid(b):
        return bindings_satisfy(symbols, b, guards)

    def internally_distinct(b):
        vals = list(b.values())
        return len(set(vals)) == len(vals)

    seen, out = set(), []

    def add(b):
        # Requirement (1): reject a binding where two VARYING symbols share a
        # value — dynamo would unify them into ONE square symbol. A plain
        # multiplicative scale collides for adversarial hint ratios (e.g.
        # h0 == 2*h1), so the candidate generator can still emit squares;
        # drop them here rather than letting a square short-circuit the
        # coupled fallback below. Frozen values are merged AFTER the
        # distinctness check (a frozen dim is concrete, not a symbol).
        if len(names) > 1 and not internally_distinct(b):
            return
        b = {**b, **frozen}
        key = tuple(sorted(b.items()))
        if key not in seen and valid(b):
            seen.add(key)
            out.append(b)

    def _distinct_binding(shape_idx):
        # Scale every symbol by (shape_idx + 1); on a within-shape value
        # collision, bump ONLY the colliding symbol's multiplier by +1 until
        # distinct. Each value stays an integer multiple of the symbol's hint,
        # so divisibility guards (Mod(s, k) == 0) survive; the per-symbol base
        # multiplier grows with shape_idx, so each symbol also takes >=2 values
        # across the set (requirement (2)). Eq(s0, s1) couplings are the one
        # case distinct values legitimately violate -> valid() rejects them and
        # we fall through to the coupled path.
        b, used = {}, set()
        for name in names:
            mult = shape_idx + 1
            val = hint[name] * mult
            while val in used:
                mult += 1
                val = hint[name] * mult
            used.add(val)
            b[name] = val
        return b

    # INTERNALLY-DISTINCT candidates across `n` shapes with growing scale.
    for shape_idx in range(n + 4):
        if len(out) >= n:
            break
        add(_distinct_binding(shape_idx))
    if len(out) >= min(n, 2) and len(out) >= 1:
        # Got enough internally-distinct guard-valid shapes (or the only ones
        # reachable). Good — these force the general non-square kernel.
        if len(out) >= 2 or len(names) == 1:
            return out[:n]
    # Coupled / heavily-guarded family: distinct shapes violate guards. Warm
    # at scaled EQUAL-magnitude shapes (preserves Eq couplings); the square
    # kernel is then the model's real kernel.
    out2, seen2 = [], set()

    def add2(b):
        b = {**b, **frozen}
        key = tuple(sorted(b.items()))
        if key not in seen2 and valid(b):
            seen2.add(key)
            out2.append(b)

    for factor in (1, 2, 3, 4):
        if len(out2) >= n:
            break
        add2({name: val * factor for name, val in hint.items()})
    if out2:
        return out2[:n]
    return out[:n] if out else [{**hint, **frozen}]
