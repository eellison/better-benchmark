"""Mask rules for jagged / HSTU / side-effect / approximate-index nodes.

Each rule takes one FX node and returns a mask reason or None. The rules
read only the graph (op targets, schema argument names, producer chains),
never tensor values, so they work on a live post-grad graph, a fake-traced
runnable, or a saved full graph.

- "hstu": attention over jagged sequences. Its cost depends on the length
  distribution, so a synthetic timing is junk; held at its prod time.
- "jagged": other ops whose work or validity depends on offsets/lengths
  values (jagged/cumsum/segment ops, repeat_interleave with a repeats
  tensor). Held at prod time.
- "side_effect": ops that register fine but cannot be replayed from a saved
  graph (live handles, logging). Stripped at ingest when nothing reads
  their results (strip_side_effect_source); otherwise held at prod time.
- "hw_unsupported": ops that need a newer GPU than the benchmark target
  (e.g. Blackwell-only TLX kernels on H100). Held at prod time.
- "approx_index": an aten op whose index operand is derived from repro
  inputs, which a repro regenerates as uniform random in-range values.
  Timed and reported, but flagged (locality differs from prod).

capture_hook registers side_effect_rule, jagged_rule and hw_unsupported_rule
as node rules and
region_approx_index_rule as a region rule. Only generic rules live here. Operator lists for non-public libraries are
registered at runtime with register_mask_ops().
"""

import ast
import json
import os
import re
from pathlib import Path

# Namespaces whose ops BB treats as built in.
BUILTIN_NAMESPACES = frozenset({
    "aten", "prims", "higher_order", "inductor", "profiler", "quantized",
    "_c10d_functional", "c10d_functional",
})

# HOPs that wrap an op; the wrapped op is the one the rules classify.
#   auto_functionalized(_v2)(op, **kwargs), with_effects(token, op, *args)
_WRAPPED_OP_POSITION = {
    "auto_functionalized": 0,
    "auto_functionalized_v2": 0,
    "with_effects": 1,
}

# (namespace, op-name regex) lists per reason. register_mask_ops() extends
# them; the defaults cover public libraries only.
HSTU_OPS: list[tuple[str, re.Pattern]] = []
JAGGED_OPS: list[tuple[str, re.Pattern]] = [
    ("fbgemm", re.compile(
        r"jagged|cumsum|segment|pack_segments|offsets_range|lengths_range")),
    ("aten", re.compile(r"^(_segment_reduce|segment_reduce)$")),
]
# (namespace, op name) -> positions of the arguments the op mutates, used
# when the op's schema is not available.
SIDE_EFFECT_OPS: dict[tuple[str, str], tuple[int, ...]] = {}

# (namespace, op-name regex, minimum (major, minor) compute capability).
HW_UNSUPPORTED_OPS: list[tuple[str, re.Pattern, tuple[int, int]]] = []

# Overrides the benchmark target's compute capability, e.g. "9.0", when the
# ingesting process is not on the target GPU.
TARGET_CAPABILITY_ENV = "BETTER_BENCHMARK_TARGET_CAPABILITY"

# Schema argument names that carry jagged structure. A custom op with a
# Tensor argument of one of these names is jagged even when its name is not
# in the lists above.
JAGGED_ARG_NAMES = frozenset({
    "offsets", "lengths", "seq_offsets", "x_offsets", "y_offsets",
    "q_offsets", "k_offsets", "kv_offsets", "cu_seqlens", "cu_seqlens_q",
    "cu_seqlens_k", "num_targets", "repeats", "seq_lengths",
})
_ATTENTION_NAME = re.compile(r"attn|attention")

# Index operand position for aten ops, used when the schema is unavailable
# (the schema names the operand "index" or "indices").
INDEX_OPERAND = {
    "index.Tensor": 1,
    "index_put.default": 1,
    "index_put_.default": 1,
    "_unsafe_index.Tensor": 1,
    "_unsafe_index_put.default": 1,
    "_unsafe_masked_index.default": 2,
    "_unsafe_masked_index_put_accumulate.default": 2,
    "_index_put_impl_.default": 1,
    "gather.default": 2,
    "scatter.src": 2,
    "scatter.value": 2,
    "scatter.reduce": 2,
    "scatter.value_reduce": 2,
    "scatter_add.default": 2,
    "scatter_reduce.two": 2,
    "index_select.default": 2,
    "index_add.default": 2,
    "index_copy.default": 2,
    "index_fill.int_Scalar": 2,
    "index_fill.int_Tensor": 2,
    "embedding.default": 1,
    "embedding_dense_backward.default": 1,
    "take.default": 1,
}
_INDEX_ARG_NAMES = frozenset({"index", "indices"})

# Producers whose output does not depend on input values: an index chain
# that bottoms out only in these is exact in a repro.
_VALUE_FREE_PRODUCERS = frozenset({
    "iota", "arange", "full", "zeros", "ones", "empty", "scalar_tensor",
    "full_like", "zeros_like", "ones_like", "empty_like", "empty_strided",
    "new_zeros", "new_ones", "new_full", "new_empty",
    "sym_size", "sym_numel", "sym_stride",
})


def register_mask_ops(spec: dict) -> None:
    """Add operator lists, e.g. from a private config file.

    ``spec`` = {"hstu": [{"namespace", "pattern"}], "jagged": [...],
    "side_effect": [{"namespace", "op", "mutates": [arg positions]}]}.
    """
    for key, table in (("hstu", HSTU_OPS), ("jagged", JAGGED_OPS)):
        for entry in spec.get(key, []):
            item = (entry["namespace"], re.compile(entry["pattern"]))
            if all(item[0] != ns or item[1].pattern != p.pattern for ns, p in table):
                table.append(item)
    for entry in spec.get("hw_unsupported", []):
        HW_UNSUPPORTED_OPS.append((
            entry["namespace"],
            re.compile(entry["pattern"]),
            tuple(entry["min_capability"]),
        ))
    for entry in spec.get("side_effect", []):
        SIDE_EFFECT_OPS[(entry["namespace"], entry["op"])] = tuple(
            entry.get("mutates", ()))


def load_mask_ops(path: str | Path) -> None:
    register_mask_ops(json.loads(Path(path).read_text()))


# Full graphs held at production time whatever their own masks say (e.g. a
# numerics check that cannot run on the benchmark GPU), keyed by the sha256
# of the graph runnable they were ingested from, with the reason.
HELD_GRAPHS: dict[str, str] = {}
SOURCE_SHA_KEY = "source_runnable_sha256"


def register_held_graphs(entries: list[dict]) -> None:
    """``entries`` = [{"runnable_sha256", "reason"}, ...]."""
    for entry in entries:
        HELD_GRAPHS[entry["runnable_sha256"]] = entry["reason"]


def load_held_graphs(path: str | Path) -> None:
    register_held_graphs(json.loads(Path(path).read_text())["held"])


def held_mask(sidecar: dict) -> dict | None:
    """The graph mask of a held full graph's sidecar, or None."""
    reason = HELD_GRAPHS.get(sidecar.get(SOURCE_SHA_KEY))
    if reason is None:
        return None
    masked = sidecar.get("masked") or {}
    reasons = sorted({*masked.get("reasons", ()), "held"})
    return {"reasons": reasons, "scope": "graph", "held_reason": reason}


def op_parts(target):
    """(namespace, op name, overload name) of an OpOverload-like target."""
    namespace = getattr(target, "namespace", None)
    if namespace is None:
        return None
    packet = getattr(target, "overloadpacket", None)
    name = getattr(packet, "__name__", None) or getattr(target, "_opname", "")
    name = name.split(".")[-1]
    return namespace, name, getattr(target, "_overloadname", "default")


def _hop_name(target):
    name = getattr(target, "name", None)
    if callable(name):
        try:
            return name()
        except TypeError:
            return None
    return None


def classified_op(node):
    """The op a node runs, unwrapping auto_functionalized / with_effects.

    Returns (target, positional args, kwargs) or None for non-op nodes.
    """
    if node.op != "call_function":
        return None
    target = node.target
    hop = _hop_name(target)
    if hop in _WRAPPED_OP_POSITION:
        pos = _WRAPPED_OP_POSITION[hop]
        if len(node.args) > pos:
            return node.args[pos], tuple(node.args[pos + 1:]), dict(node.kwargs)
        return None
    return target, tuple(node.args), dict(node.kwargs)


def _bound_arguments(target, args, kwargs):
    """{schema arg name: value} for the args a node passes, or None."""
    schema = getattr(target, "_schema", None)
    if schema is None:
        return None
    bound = dict(kwargs)
    for arg_schema, value in zip(schema.arguments, args):
        bound[arg_schema.name] = value
    return bound


def _is_tensor_like(value):
    return hasattr(value, "op") or (
        isinstance(value, (list, tuple)) and any(hasattr(v, "op") for v in value))


def _subgraph_nodes(node):
    """Nodes of the subgraphs a control-flow HOP (cond, while_loop,
    invoke_subgraph, ...) runs, recursively."""
    owner = node.graph.owning_module
    if node.op != "call_function" or owner is None:
        return
    stack = [a for a in (*node.args, *node.kwargs.values()) if hasattr(a, "op")]
    for arg in stack:
        if arg.op != "get_attr":
            continue
        sub = getattr(owner, arg.target, None)
        graph = getattr(sub, "graph", None)
        if graph is None:
            continue
        for inner in graph.nodes:
            yield inner
            yield from _subgraph_nodes(inner)


def _first_in_subgraphs(rule, node):
    for inner in _subgraph_nodes(node):
        reason = rule(inner)
        if reason is not None:
            return reason
    return None


def side_effect_rule(node):
    """"side_effect" for ops in SIDE_EFFECT_OPS (wrappers unwrapped), or
    for a control-flow HOP whose subgraphs run one."""
    return _side_effect_op(node) or _first_in_subgraphs(side_effect_rule, node)


def _side_effect_op(node):
    op = classified_op(node)
    if op is None:
        return None
    parts = op_parts(op[0])
    if parts is not None and (parts[0], parts[1]) in SIDE_EFFECT_OPS:
        return "side_effect"
    return None


def target_capability() -> tuple[int, int] | None:
    """The benchmark target's compute capability: the env override, else
    the current CUDA device, else None (unknown)."""
    override = os.environ.get(TARGET_CAPABILITY_ENV)
    if override:
        major, _, minor = override.partition(".")
        return int(major), int(minor or 0)
    import torch

    if torch.cuda.is_available():
        return tuple(torch.cuda.get_device_capability())
    return None


def hw_unsupported_rule(node):
    """"hw_unsupported" for ops in HW_UNSUPPORTED_OPS whose minimum compute
    capability the target lacks (or when the target is unknown), or for a
    control-flow HOP whose subgraphs run one.

    Unlike the other rules this depends on the target hardware, so a mask
    recorded at capture holds for that target only.
    """
    return _hw_unsupported_op(node) or _first_in_subgraphs(hw_unsupported_rule, node)


def _hw_unsupported_op(node):
    if not HW_UNSUPPORTED_OPS:
        return None
    op = classified_op(node)
    if op is None:
        return None
    parts = op_parts(op[0])
    if parts is None:
        return None
    for namespace, pattern, min_capability in HW_UNSUPPORTED_OPS:
        if parts[0] == namespace and pattern.search(parts[1]):
            capability = target_capability()
            if capability is None or capability < min_capability:
                return "hw_unsupported"
    return None


def jagged_rule(node):
    """"hstu" or "jagged" for ops whose cost/validity depends on offsets,
    or for a control-flow HOP whose subgraphs run one."""
    return _jagged_op(node) or _first_in_subgraphs(jagged_rule, node)


def _jagged_op(node):
    op = classified_op(node)
    if op is None:
        return None
    target, args, kwargs = op
    parts = op_parts(target)
    if parts is None:
        return None
    namespace, name, overload = parts
    for ns, pattern in HSTU_OPS:
        if namespace == ns and pattern.search(name):
            return "hstu"
    for ns, pattern in JAGGED_OPS:
        if namespace == ns and pattern.search(name):
            return "jagged"
    # repeat_interleave(repeats_tensor, output_size=...) device-asserts under
    # synthetic repeats; repeat_interleave.self_int (a scalar repeat) is dense.
    if namespace == "aten" and name == "repeat_interleave":
        return "jagged" if overload in ("Tensor", "self_Tensor") else None
    if namespace in BUILTIN_NAMESPACES:
        return None
    bound = _bound_arguments(target, args, kwargs) or kwargs
    if any(arg_name in JAGGED_ARG_NAMES and _is_tensor_like(value)
           for arg_name, value in bound.items()):
        return "hstu" if _ATTENTION_NAME.search(name) else "jagged"
    return None


def index_operands(node):
    """The index operand nodes of an index-consuming aten op (may be [])."""
    op = classified_op(node)
    if op is None:
        return []
    target, args, kwargs = op
    parts = op_parts(target)
    if parts is None or parts[0] != "aten":
        return []
    bound = _bound_arguments(target, args, kwargs)
    if bound is not None:
        values = [v for k, v in bound.items() if k in _INDEX_ARG_NAMES]
    else:
        pos = INDEX_OPERAND.get(f"{parts[1]}.{parts[2]}")
        values = [args[pos]] if pos is not None and pos < len(args) else []
    flat = []
    for value in values:
        if isinstance(value, (list, tuple)):
            flat.extend(v for v in value if hasattr(v, "op"))
        elif hasattr(value, "op"):
            flat.append(value)
    return flat


def _is_value_free(node):
    if node.op != "call_function":
        return False
    parts = op_parts(node.target)
    return parts is not None and parts[1] in _VALUE_FREE_PRODUCERS


def index_depends_on_inputs(node, region=None):
    """True if an index operand of ``node`` is derived from input values.

    ``region`` is the set of nodes regenerated together (a fusion
    partition); a value entering from outside it, or a placeholder, is a
    repro input and so becomes synthetic. With ``region=None`` the whole
    graph is the region and only placeholders count.
    """
    stack = list(index_operands(node))
    seen = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur.op == "placeholder":
            return True
        if region is not None and cur not in region:
            return True
        if _is_value_free(cur):
            continue
        stack.extend(cur.all_input_nodes)
    return False


def approx_index_rule(node, region=None):
    """"approx_index" for index ops whose indices come from input values."""
    if index_operands(node) and index_depends_on_inputs(node, region):
        return "approx_index"
    return None


def region_approx_index_rule(nodes):
    """Repro-level rule for one fusion partition: "approx_index" or None."""
    region = set(nodes)
    for node in nodes:
        if approx_index_rule(node, region) is not None:
            return "approx_index"
    return None


# ---------------------------------------------------------------------------
# Strip side-effect ops from graph source at ingest.
# ---------------------------------------------------------------------------

def _dotted(expr):
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        base = _dotted(expr.value)
        return f"{base}.{expr.attr}" if base else None
    return None


def _op_key(dotted):
    """(namespace, op) of a torch.ops.<ns>.<op>[.<overload>] name."""
    parts = (dotted or "").split(".")
    if len(parts) in (4, 5) and parts[:2] == ["torch", "ops"]:
        return parts[2], parts[3]
    return None


def _assigned_name(stmt):
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        return target.id if isinstance(target, ast.Name) else None
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return stmt.target.id
    return None


def _call_of(stmt):
    value = getattr(stmt, "value", None)
    return value if isinstance(value, ast.Call) else None


def _loads(stmt):
    return {n.id for n in ast.walk(stmt)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _mutated_positions(key):
    """Mutated argument positions of a side-effect op, from its schema."""
    import torch

    try:
        packet = getattr(getattr(torch.ops, key[0]), key[1])
        schema = packet.default._schema
    except (AttributeError, RuntimeError):
        return SIDE_EFFECT_OPS.get(key)
    return tuple(i for i, a in enumerate(schema.arguments)
                 if a.alias_info is not None and a.alias_info.is_write)


def _mutated_names(key, call, schema_args):
    """Names of the graph values a side-effect call mutates, or None."""
    positions = _mutated_positions(key)
    if positions is None:
        return None
    names = set()
    for pos in positions:
        value = call.args[pos] if pos < len(call.args) else None
        if value is None and schema_args is not None and pos < len(schema_args):
            value = next((k.value for k in call.keywords
                          if k.arg == schema_args[pos]), None)
        if value is None:
            continue
        if not isinstance(value, ast.Name):
            return None
        names.add(value.id)
    return names


def _schema_arg_names(key):
    import torch

    try:
        packet = getattr(getattr(torch.ops, key[0]), key[1])
        return [a.name for a in packet.default._schema.arguments]
    except (AttributeError, RuntimeError):
        return None


def _is_copy_back(stmt, params, value_names):
    """``copy_(param, v)`` with v in value_names: a mutation epilogue."""
    call = _call_of(stmt)
    if call is None or _dotted(call.func) != "torch.ops.aten.copy_.default":
        return False
    if len(call.args) != 2 or not all(isinstance(a, ast.Name) for a in call.args):
        return False
    return call.args[0].id in params and call.args[1].id in value_names


def _plan_strips(forward):
    """Statements to blank for each strippable side-effect call.

    Returns (strips, blocked): strips is a list of (op label, [stmt]) and
    blocked a list of (op label, reason).
    """
    params = {a.arg for a in forward.args.args}
    body = forward.body
    strips, blocked = [], []
    for i, stmt in enumerate(body):
        call = _call_of(stmt)
        if call is None:
            continue
        func = _dotted(call.func)
        wrapped = None
        key = _op_key(func)
        if func is not None and func.startswith("torch.ops.higher_order."):
            hop = func.rsplit(".", 1)[1]
            if hop in _WRAPPED_OP_POSITION and len(call.args) > _WRAPPED_OP_POSITION[hop]:
                wrapped = hop
                key = _op_key(_dotted(call.args[_WRAPPED_OP_POSITION[hop]]))
        if key is None or key not in SIDE_EFFECT_OPS:
            continue
        label = f"{key[0]}.{key[1]}"
        if wrapped == "with_effects":
            blocked.append((label, "with_effects token threading"))
            continue
        result = _assigned_name(stmt)
        removed = [stmt]
        later = body[i + 1:]
        if wrapped is not None:
            # auto_functionalized: results are (op outputs..., new bases).
            # They may only feed getitems whose sole readers copy the new
            # base back into a graph input.
            mutated = set()
            bases = next((k.value for k in call.keywords if k.arg == "_all_bases"), None)
            if isinstance(bases, (ast.List, ast.Tuple)):
                mutated = {e.id for e in bases.elts if isinstance(e, ast.Name)}
            items = {}
            ok = True
            for s in later:
                if result in _loads(s):
                    value = getattr(s, "value", None)
                    name = _assigned_name(s)
                    if (isinstance(value, ast.Subscript)
                            and isinstance(value.value, ast.Name)
                            and value.value.id == result and name):
                        items[name] = s
                    else:
                        ok = False
            for s in later:
                used = _loads(s) & set(items)
                if not used or s in items.values():
                    continue
                if _is_copy_back(s, params, set(items)):
                    removed.append(s)
                else:
                    ok = False
            removed.extend(items.values())
            if not ok:
                blocked.append((label, "result read by compute"))
                continue
        else:
            if result is not None and any(result in _loads(s) for s in later):
                blocked.append((label, "result read by compute"))
                continue
            mutated = _mutated_names(key, call, _schema_arg_names(key))
            if mutated is None:
                blocked.append((label, "unknown mutated arguments"))
                continue
        # A mutated value may only be read later by a copy-back epilogue.
        if any(_loads(s) & mutated and s not in removed
               and not _is_copy_back(s, params, mutated)
               for s in later):
            blocked.append((label, "mutated argument read by compute"))
            continue
        strips.append((label, removed))
    return strips, blocked


def _graph_forward(tree):
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == "forward":
                    return member
    return None


def strip_side_effect_source(source: str):
    """Blank strippable side-effect calls in a printed graph's forward().

    A call is stripped when nothing but a copy-back epilogue reads its
    results or the arguments it mutates. Each removed statement becomes
    ``pass`` in place, so line numbers and the other statements on its line
    are unchanged. Producers that only fed a stripped call stay in the
    source (they are dead code to a compiler).

    Returns (new source, stripped op labels, [(op label, reason)] blocked).
    """
    if not SIDE_EFFECT_OPS:
        return source, [], []
    parse_source = source.replace("class <lambda>", "class Repro")
    try:
        tree = ast.parse(parse_source)
    except SyntaxError:
        return source, [], []
    forward = _graph_forward(tree)
    if forward is None:
        return source, [], []
    strips, blocked = _plan_strips(forward)
    if not strips:
        return source, [], blocked
    lines = source.splitlines(keepends=True)
    # Lines are edited as UTF-8 bytes (ast offsets are byte offsets), right
    # to left so earlier offsets stay valid.
    edits = sorted(
        ((s.lineno, s.col_offset, s.end_lineno, s.end_col_offset)
         for _label, stmts in strips for s in stmts),
        reverse=True,
    )
    for lineno, col, end_lineno, end_col in edits:
        first = lines[lineno - 1].encode()
        last = lines[end_lineno - 1].encode()
        merged = first[:col] + b"pass" + last[end_col:]
        lines[lineno - 1:end_lineno] = [merged.decode()]
    return "".join(lines), [label for label, _ in strips], blocked
