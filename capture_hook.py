"""
Standalone post-grad capture hook.

Register this hook before running ANY model with torch.compile. It captures
all fusible regions from the post-grad graph and writes standalone repro files
to a configurable output directory.

Usage:
    # In your script or PYTHONSTARTUP:
    from capture_hook import install_capture_hook
    install_capture_hook(output_dir="/tmp/captures/my_model", label="my_model")

    # Then just run your model normally:
    model = ...
    compiled = torch.compile(model)
    compiled(inputs)

    # Or as an environment variable (auto-installs on import):
    REPRO_CAPTURE_DIR=/tmp/captures/my_model python my_training_script.py

The captured repros are standalone .py files with class Repro + make_inputs().
After capture, merge them into the canonical set with:
    python merge_captures.py /tmp/captures/my_model --canonical-dir repros/
"""
import collections
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.fx as fx
import torch._inductor.config as inductor_config

from full_graph_harness import (
    _is_fake_or_meta_tensor as _is_fake_or_meta,
    infer_index_bounds_from_gm,
    infer_permutation_indices_from_gm,
    placeholder_info_from_gm,
)
# Single source of truth for the emitted version marker: the generated repro
# template stamps CURRENT_REPRO_VERSION rather than a hardcoded literal, so a
# future format bump (changing only the constant) can never leave this writer
# stamping a stale version -- the exact bug the v3 migration hit when it
# bumped the template but not repro_harness.CURRENT_REPRO_VERSION.
from repro_harness import CURRENT_REPRO_VERSION


# ============================================================================
# Shared partitioning + pattern hashing (single source of truth)
#
# These module-level helpers define EXACTLY how the capture pipeline cuts a
# post-grad graph into fusible partitions and how each partition is
# content-addressed (pattern_hash / shape_hash). They are used both by the
# capture path (_CaptureState.process_graph) and by offline accounting tools
# (scripts/model_graph_accounting.py), guaranteeing that accounting partitions
# are identical to how canonical repros were originally cut.
# ============================================================================

import operator as _operator

# Pure view/metadata ops the partitioner treats as transparent: they may be
# absorbed into a fusible partition but never constitute compute on their own.
TRANSPARENT_OPS = {
    _operator.getitem,
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.squeeze.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.expand.default,
    torch.ops.aten.t.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.select.int,
    torch.ops.aten.as_strided.default,
}


def partition_node_is_supported(node: fx.Node) -> bool:
    """Capture-pipeline fusibility test for a single node.

    A node belongs in a fusible partition if it is a transparent view op or
    Inductor's is_fusible_node says it has a fusible lowering (no BLAS/cuDNN
    flop-counter ops, no fallbacks, no collectives).
    """
    from torch._inductor.fx_passes.fusion_regions import is_fusible_node

    if node.op == "call_function" and node.target in TRANSPARENT_OPS:
        return True
    return is_fusible_node(node)


def graph_node_accounting(gm: fx.GraphModule, components=None) -> dict:
    """Exhaustive fusibility classification of every call_function node.

    Serialized into each full_graph_*.meta.json so the capture-time
    fusible/non-fusible decision is an auditable artifact, not a transient:
    if Inductor's is_fusible_node changes between pytorch versions, the
    manifest diff shows exactly which ops moved buckets. Every node lands
    in exactly ONE bucket and the counts must sum to total_call_functions
    (enforced by tests/test_canonical_invariants.py).

      fusible_in_partition: in a compute partition -> covered by a
          canonical repro's pattern hash
      fusible_unpartitioned: fusible/transparent but placed in no
          partition (dangling views etc.) -> not a kernel, claimed by
          neither side of the attribution identity
      non_fusible: extern/fallback (BLAS, cuDNN, sdpa, ...) -> the
          extern side of model_attribution

    All three buckets log the aggregate op-target -> count set (graph
    level, never per-partition — partition contents live in the canonical
    repros). A misclassification in EITHER direction is then visible in
    the artifact: an extern op wrongly marked fusible shows up in the
    fusible set, and vice versa.
    """
    if components is None:
        components = get_fusion_partitions(gm)
    partitioned = set()
    for comp in components:
        partitioned.update(comp)
    buckets = {
        "fusible_in_partition": collections.Counter(),
        "fusible_unpartitioned": collections.Counter(),
        "non_fusible": collections.Counter(),
    }
    total = 0
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        total += 1
        # builtins str() as "<built-in function getitem>" — spell them
        # module.name (operator.getitem) like every other op.
        if isinstance(node.target, torch._ops.OpOverload):
            name = str(node.target)
        else:
            name = (f"{getattr(node.target, '__module__', '')}."
                    f"{getattr(node.target, '__name__', node.target)}").lstrip(".")
        if node in partitioned:
            buckets["fusible_in_partition"][name] += 1
        elif partition_node_is_supported(node):
            buckets["fusible_unpartitioned"][name] += 1
        else:
            buckets["non_fusible"][name] += 1
    return {
        "total_call_functions": total,
        "counts": {k: sum(v.values()) for k, v in buckets.items()},
        "ops": {k: dict(sorted(v.items())) for k, v in buckets.items()},
    }


def partition_has_reduction(nodes) -> bool:
    """True if any node in the partition is a reduction op."""
    for n in nodes:
        if n.op == "call_function" and isinstance(n.target, torch._ops.OpOverload):
            if torch.Tag.reduction in n.target.tags:
                return True
    return False


def partition_has_real_compute(nodes) -> bool:
    """Check if a partition has at least one non-transparent compute op."""
    for n in nodes:
        if n.op != "call_function":
            continue
        if n.target in TRANSPARENT_OPS:
            continue
        # It's a real compute op (pointwise, reduction, etc.)
        return True
    return False


def _split_connected_components(nodes):
    """Split a list of nodes into connected components by data flow."""
    from collections import deque

    node_set = set(nodes)
    # Build adjacency: two nodes in the partition are connected if one
    # feeds directly into another (producer -> consumer).
    adjacency = {n: set() for n in nodes}
    for n in nodes:
        # Check all args: if any arg is a node in our partition, link them
        def _link_arg(x):
            if isinstance(x, fx.Node) and x in node_set and x is not n:
                adjacency[n].add(x)
                adjacency[x].add(n)
        fx.map_arg(n.args, _link_arg)
        fx.map_arg(n.kwargs, _link_arg)

    # BFS to find connected components
    visited = set()
    result_components = []
    for start in nodes:
        if start in visited:
            continue
        component = []
        queue = deque([start])
        visited.add(start)
        while queue:
            cur = queue.popleft()
            component.append(cur)
            for neighbor in adjacency[cur]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        result_components.append(component)
    return result_components


def get_fusion_partitions(gm: fx.GraphModule) -> list:
    """Partition a post-grad GraphModule into fusible regions, EXACTLY as the
    capture pipeline does.

    Uses CapabilityBasedPartitioner + is_fusible_node (from
    torch._inductor.fx_passes.fusion_regions) + create_op_support, with
    transparent view ops allowed inside partitions and horizontal fusion
    disabled (via skip_horizontal_fusion when available, otherwise a
    connected-component split). Partitions without real compute are dropped —
    the returned list corresponds 1:1 to the canonical repros process_graph()
    would capture.

    Returns: list of partitions, each a list of fx.Node from gm.graph.
    """
    import inspect

    from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
    from torch.fx.passes.operator_support import create_op_support

    def _is_supported(_submodules, node):
        return partition_node_is_supported(node)

    support = create_op_support(_is_supported)
    _part_kwargs = dict(allows_single_node_partition=True)
    has_skip_horizontal_fusion = (
        'skip_horizontal_fusion'
        in inspect.signature(CapabilityBasedPartitioner.__init__).parameters
    )
    if has_skip_horizontal_fusion:
        _part_kwargs['skip_horizontal_fusion'] = True
    partitioner = CapabilityBasedPartitioner(
        gm, support, **_part_kwargs,
    )
    partitions = partitioner.propose_partitions()
    components = [list(p.nodes.keys()) for p in partitions]

    if not has_skip_horizontal_fusion:
        split_components = []
        for comp in components:
            split_components.extend(_split_connected_components(comp))
        components = split_components

    return [comp for comp in components if partition_has_real_compute(comp)]


def extract_partition_subgraph(origin_nodes: list, gm: fx.GraphModule):
    """Extract a standalone sub-GraphModule for one fusible partition.

    Returns (sub_gm, placeholder_info, shape_params) or None.
    """
    seen = set()
    unique_origins = []
    for n in origin_nodes:
        if id(n) not in seen:
            seen.add(id(n))
            unique_origins.append(n)
    origin_nodes = unique_origins

    needed_nodes: set[fx.Node] = set(origin_nodes)

    # A node is a partition output iff it has ANY use outside the partition
    # (including being a graph output). Internal users are irrelevant: a node
    # consumed both inside (e.g. by a sum) and outside (e.g. by a
    # convolution_backward the partitioner cut away) MUST remain an output,
    # or the repro under-constrains the computation — elimination passes get
    # freedom the real model never grants and measured gaps don't compose
    # (see investigation_results/squeezenet_scatter_e2e_validation.md).
    # The escaping set: any use outside the partition (including the graph
    # output node, which is never in needed_nodes) makes a node an output.
    # Nodes with only internal users are internal; nodes with NO users are
    # dead (upstream DCE makes this near-impossible) and are simply dropped —
    # EXCEPT mutating ops (copy_, index_put_, ...): their effect is the
    # in-place mutation of an input buffer, not their return value, so they
    # can legitimately have zero users and must never be dropped.
    def _is_mutating(n):
        target = getattr(n, "target", None)
        schema = getattr(target, "_schema", None)
        if schema is not None:
            return schema.is_mutable
        name = getattr(target, "__name__", "") or str(target)
        return name.rstrip(".default").endswith("_")

    output_nodes = [
        n for n in origin_nodes
        if any(user not in needed_nodes for user in n.users) or _is_mutating(n)
    ]
    if not output_nodes:
        output_nodes = origin_nodes

    # Output order must never carry information: emit outputs in DEFINITION
    # order (the node's position in the source graph). origin_nodes arrives
    # in whatever order the partitioner enumerated; two captures of the same
    # partition could otherwise serialize different output-tuple orders and
    # mint different DAG-signature hashes for identical computations.
    graph_pos = {n: i for i, n in enumerate(gm.graph.nodes)}
    output_nodes = sorted(output_nodes, key=lambda n: graph_pos.get(n, 0))

    new_graph = fx.Graph()
    env: dict[fx.Node, fx.Node] = {}
    placeholder_info: dict[str, dict] = {}

    def _resolve_sym(x):
        """Resolve SymInt/SymFloat to concrete int/float."""
        if isinstance(x, (torch.SymInt, torch.SymFloat)):
            return x.node.hint if hasattr(x, 'node') and hasattr(x.node, 'hint') else int(x)
        return int(x)

    def _storage_key(val):
        try:
            return id(val.untyped_storage())
        except Exception:
            return None

    def _record_placeholder(name: str, meta: dict) -> None:
        val = meta.get("val", None)
        if val is not None and isinstance(val, torch.Tensor):
            placeholder_info[name] = {
                "shape": [_resolve_sym(s) for s in val.shape],
                "stride": [_resolve_sym(s) for s in val.stride()] if not val.is_contiguous() else [],
                "dtype": str(val.dtype),
                "device": str(val.device),
            }
            off = _resolve_sym(val.storage_offset()) if val.storage_offset() else 0
            if off:
                placeholder_info[name]["storage_offset"] = off
            # Alias tag: inputs whose fake vals share one untyped storage
            # (packed-qkv saved views). Live-capture-only signal — any
            # retrace re-fabricates inputs and the identity is gone. Tag
            # is the storage key; serialization rewrites it to a small
            # group index ("alias_group") so replay can allocate ONE
            # buffer per group and as_strided the members. Storage SIZE
            # captured here too (the true allocation) so consumers never
            # re-derive it by scanning members.
            sk = _storage_key(val)
            if sk is not None:
                placeholder_info[name]["_storage_key"] = sk
                try:
                    placeholder_info[name]["_storage_nbytes"] = int(
                        val.untyped_storage().size())
                except Exception:
                    pass
        elif val is not None and isinstance(val, (torch.SymInt, torch.SymFloat)):
            hint = val.node.hint if hasattr(val, 'node') and hasattr(val.node, 'hint') else int(val)
            placeholder_info[name] = {
                "shape": [],
                "stride": [],
                "dtype": "symint",
                "device": "cpu",
                "hint": hint,
            }

    _ph_names_used: set[str] = set()

    def _unique_ph_name(base: str) -> str:
        """Ensure placeholder names don't collide with internal node names."""
        # Internal call_function nodes get names like full_default, clone_default, etc.
        # Prefix external inputs to avoid collision.
        name = base
        if name in _ph_names_used:
            i = 1
            while f"{name}_{i}" in _ph_names_used:
                i += 1
            name = f"{name}_{i}"
        _ph_names_used.add(name)
        return name

    def _ensure_in_env(x: Any) -> Any:
        if isinstance(x, fx.Node):
            if x in env:
                return env[x]
            name = _unique_ph_name(x.name)
            ph = new_graph.placeholder(name)
            ph.meta = copy.copy(x.meta) if x.meta else {}
            env[x] = ph
            _record_placeholder(name, x.meta or {})
            return ph
        return x

    _VIEW_LIKE_OPS = {
        torch.ops.aten.reshape.default,
        torch.ops.aten.view.default,
        torch.ops.aten.expand.default,
    }

    shape_params: dict[str, list[int]] = {}
    _shape_counter = [0]

    def _should_lift_shape(shape_list):
        """Don't lift trivial shapes that are always the same."""
        if not shape_list:
            return False
        # All dims are 0, 1, or -1 — not shape-dependent
        if all(d in (0, 1, -1) for d in shape_list):
            return False
        return True

    def _lift_shape_arg(node, args):
        """For reshape/view ops, lift the shape literal to a parameter."""
        if node.target not in _VIEW_LIKE_OPS:
            return args
        if len(args) < 2 or not isinstance(args[1], (list, tuple)):
            return args
        shape_list = list(args[1])
        if not _should_lift_shape(shape_list):
            return args
        param_name = f"_shape_param_{_shape_counter[0]}"
        _shape_counter[0] += 1
        ph = new_graph.placeholder(param_name)
        ph.meta = {"val": shape_list}
        shape_params[param_name] = shape_list
        return (args[0], ph) + tuple(args[2:]) if len(args) > 2 else (args[0], ph)

    all_graph_nodes = list(gm.graph.nodes)
    node_order = {n: i for i, n in enumerate(all_graph_nodes)}
    sorted_needed = sorted(needed_nodes, key=lambda n: node_order.get(n, 0))

    # Create placeholders for ALL external dependencies first,
    # so they claim clean names before internal nodes are added.
    for node in sorted_needed:
        if node.op in ("call_function", "call_method"):
            def _pre_register(x):
                if isinstance(x, fx.Node) and x not in needed_nodes and x not in env:
                    _ensure_in_env(x)
            fx.map_arg(node.args, _pre_register)
            fx.map_arg(node.kwargs, _pre_register)

    for node in sorted_needed:
        if node.op == "placeholder":
            new_node = new_graph.placeholder(node.name)
            new_node.meta = copy.copy(node.meta) if node.meta else {}
            env[node] = new_node
            _record_placeholder(node.name, node.meta or {})
        elif node.op == "get_attr":
            new_node = new_graph.get_attr(node.target)
            new_node.meta = copy.copy(node.meta) if node.meta else {}
            env[node] = new_node
        elif node.op in ("call_function", "call_method"):
            new_args = fx.map_arg(node.args, _ensure_in_env)
            new_kwargs = fx.map_arg(node.kwargs, _ensure_in_env)
            if node.op == "call_function":
                new_args = _lift_shape_arg(node, new_args)
                new_node = new_graph.call_function(
                    node.target, args=new_args, kwargs=new_kwargs
                )
            else:
                new_node = new_graph.call_method(
                    node.target, args=new_args, kwargs=new_kwargs
                )
            new_node.meta = copy.copy(node.meta) if node.meta else {}
            env[node] = new_node

    mapped_outputs = [env[n] for n in output_nodes if n in env]
    if not mapped_outputs:
        return None
    if len(mapped_outputs) == 1:
        new_graph.output(mapped_outputs[0])
    else:
        new_graph.output(tuple(mapped_outputs))

    new_graph.lint()
    new_gm = fx.GraphModule(gm, new_graph)
    return new_gm, placeholder_info, shape_params


# ----------------------------------------------------------------------------
# Hash-time canonicalization (closes the reshape/view canonical-hash fork)
# ----------------------------------------------------------------------------
#
# The same partition used to hash two different ways depending on which
# pipeline produced the graph being hashed:
#
#   * Capture time: Inductor's compile_fx runs view_to_reshape(gm) BEFORE
#     post_grad_passes (see torch/_inductor/compile_fx.py), so the
#     post_grad_custom_pre_pass capture hook sees every view spelled as
#     aten.reshape.default.
#   * Retrace time: any offline path that re-traces a saved full_graph_*.py
#     through make_fx (scripts/model_graph_accounting.py,
#     scripts/repartition_from_graphs.py, recapture runs) sees the reshape
#     decompose back to aten.view.default.
#
# Result: 167 reshape-only vs 887 view-only canonical repro dirs, with
# confirmed duplicate pairs (e.g. convnextv2 sum_sum_sum_26d1711c064d ==
# sum_sum_sum_f68c9f1fa09b — origin_ops differ ONLY by 3x reshape vs 3x view).
#
# Fix: LIGHT spelling normalization at hash time. NO retracing — retracing is
# slow, fails on device-literal mismatches for CPU-loaded graphs, and couples
# the canonical hash to the pytorch version's decomposition behavior.
_HASH_OP_SPELLING_ALIASES = {
    # WHY: every aten.reshape.default in a saved post-grad graph was created
    # by Inductor's view_to_reshape pass FROM a valid aten.view, so it is
    # always view-able on its input and make_fx provably retraces it back to
    # aten.view.default. (reshape's clone+_unsafe_view decomposition only
    # fires for non-view-able inputs, which cannot occur for reshapes that
    # view_to_reshape produced.) Verified on convnextv2 saved full graphs:
    # make_fx retrace op-count diff is exactly -44 reshape / +44 view
    # (train) and -29/+29 (infer), nothing else.
    "aten.reshape.default": "aten.view.default",
}

# Ops elided at hash time (op name -> arg index whose value flows through).
# CURRENTLY EMPTY, deliberately:
#
#   WHY no clone/detach elision: empirically, make_fx fake-mode retracing
#   (the exact path model_graph_accounting.trace_full_graph uses) PRESERVES
#   aten.clone.default and aten.detach.default. Measured on the convnextv2
#   train full graph: all 14 clones survive the retrace, and both members of
#   every confirmed duplicate pair still contain clone_default. Eliding
#   clones here would therefore CREATE a new fork (capture-hash without
#   clone vs retrace-hash with clone) instead of closing one. If a provably
#   elided pattern is found later (the Milestone-2 fixed-point test reports
#   residual divergences), add it here with the same level of evidence.
_HASH_ELIDED_OPS: dict[str, int] = {}


def canonicalize_for_hash(gm):
    """LIGHT graph-level normalization for pattern hashing (NO retracing).

    Returns (nodes, resolve, op_name):
      * nodes: node list to encode, in original order, with hash-elided ops
        removed (none today; see _HASH_ELIDED_OPS).
      * resolve(node): maps a node through any elided ops to the node the
        signature should reference instead.
      * op_name(node): canonical op spelling for the signature
        (reshape -> view; see _HASH_OP_SPELLING_ALIASES).
    """
    elided: dict = {}
    for n in gm.graph.nodes:
        if n.op == "call_function" and str(n.target) in _HASH_ELIDED_OPS:
            src = n.args[_HASH_ELIDED_OPS[str(n.target)]]
            if isinstance(src, fx.Node):
                elided[n] = src
    nodes = [n for n in gm.graph.nodes if n not in elided]

    def resolve(n):
        while n in elided:
            n = elided[n]
        return n

    def op_name(n):
        name = str(n.target)
        return _HASH_OP_SPELLING_ALIASES.get(name, name)

    return nodes, resolve, op_name


def compute_dag_signature(gm) -> list:
    """Compute a DAG-structure signature for the graph.

    Encodes: for each node in topological order, its op name, which
    predecessor nodes feed each argument (by index), and structural
    literal args (list/tuple values like reduction dims, permute orders).

    Does NOT encode scalar int/float constants (e.g., mul by 3.0,
    eps=1e-6) — those don't affect kernel structure.

    Op spellings are canonicalized via canonicalize_for_hash so that
    trace-equivalent graphs (saved post-grad reshape-spelling vs make_fx
    retrace view-spelling) hash identically.

    Node ORDER is canonicalized too: nodes are renumbered by a
    deterministic topological order (Kahn's algorithm, ready set
    tie-broken by each node's structural encoding), so the textual
    interleaving of INDEPENDENT ops (relu defined before vs after a
    sibling sigmoid) never forks the hash. Placeholders keep their
    original order — input position is semantics, not spelling.
    """
    nodes, _resolve, _op_name = canonicalize_for_hash(gm)

    # --- canonical node order: deterministic Kahn topological sort -------
    node_set = set(nodes)

    def _preds(n):
        out = []

        def visit(a):
            if isinstance(a, fx.Node):
                r = _resolve(a)
                if r in node_set:
                    out.append(r)
        fx.map_arg((n.args, n.kwargs), visit)
        return out

    def _static_key(n):
        """Order-independent structural key for tie-breaking: op spelling +
        literal args (Node refs masked)."""
        def mask(a):
            if isinstance(a, fx.Node):
                return "·"
            if isinstance(a, (list, tuple)):
                return [mask(x) for x in a]
            return repr(a)
        name = _op_name(n) if n.op == "call_function" else str(n.target)
        return (n.op, name, repr(mask(list(n.args))), repr(mask(sorted(
            n.kwargs.items(), key=lambda kv: kv[0]) if n.kwargs else [])))

    node_to_idx: dict = {}
    order: list = []
    pending = []
    for n in nodes:
        if n.op == "placeholder":
            # Input position is semantics, not spelling — keep original order.
            node_to_idx[n] = len(order)
            order.append(n)
        else:
            pending.append(n)
    remaining = list(pending)
    while remaining:
        ready = [n for n in remaining
                 if all(p in node_to_idx for p in _preds(n))]
        if not ready:  # cycle cannot happen in fx; defensive fallback
            ready = remaining[:]
        ready.sort(key=lambda n: (
            sorted(node_to_idx.get(p, -1) for p in _preds(n)),
            _static_key(n),
        ))
        chosen = ready[0]
        node_to_idx[chosen] = len(order)
        order.append(chosen)
        remaining.remove(chosen)
    nodes = order
    # ---------------------------------------------------------------------

    def _int_slots(target):
        """(positional indices, kwarg names) whose schema type is bare
        'int'/'int?' (dims: cat, gather, cumsum, argmax...). Structural —
        changes the kernel — so hashed. 'Scalar'-typed slots (add.Scalar
        other, clamp bounds) are baked constants and stay unhashed. Same
        schema-typed discrimination as the SymInt[] shape lift: the op's
        own schema declares which ints are structure, no op list to
        maintain. Both invocation forms covered (the same arg may arrive
        positionally or as a kwarg depending on how the model called it)."""
        schema = getattr(target, "_schema", None)
        if schema is None:
            return (), frozenset()
        positions = []
        names = []
        for i, a in enumerate(schema.arguments):
            # arg.type str-spells optional as Optional[int] (schema text
            # says int?) — accept both spellings.
            if str(a.type) in ("int", "Optional[int]"):
                names.append(a.name)
                if not a.kwarg_only:
                    positions.append(i)
        return tuple(positions), frozenset(names)

    def _encode_arg(arg, arg_idx, int_slots=()):
        """Encode an argument for the signature."""
        if isinstance(arg, torch.fx.Node):
            arg = _resolve(arg)
        if isinstance(arg, torch.fx.Node) and arg in node_to_idx:
            return ("node", node_to_idx[arg], arg_idx)
        elif isinstance(arg, (list, tuple)):
            # Structural args: reduction dims, permute orders, reshape targets
            # Encode the structure (length + which are ints) but not concrete values
            # EXCEPT for small int lists (dims) where the values matter
            if all(isinstance(x, int) for x in arg):
                # This is a dim list like [0, 2, 1, 3] or [-1] — encode values
                return ("dims", list(arg), arg_idx)
            return ("list", len(arg), arg_idx)
        elif isinstance(arg, bool):
            # Booleans like keepdim=True matter for output shape
            return ("bool", arg, arg_idx)
        elif isinstance(arg, torch.dtype):
            # convert_element_type/to.dtype targets: different output dtype
            # = different bytes written = different kernel. Collision found
            # by adversarial review 2026-06-11 (f32 vs f16 convert hashed
            # identically); pinned by tests/test_canonical_invariants.py.
            return ("dtype", str(arg), arg_idx)
        elif isinstance(arg, str):
            # String modes (div rounding_mode, gelu approximate): select
            # genuinely different computations.
            return ("str", arg, arg_idx)
        elif isinstance(arg, torch.memory_format):
            return ("memfmt", str(arg), arg_idx)
        elif isinstance(arg, int) and arg_idx in int_slots:
            # Bare 'int'-typed schema slot: a dim (cat/gather/cumsum) —
            # structural. 'Scalar'-typed int constants fall through below.
            return ("int", arg, arg_idx)
        # Scalar int/float constants — don't encode (same kernel regardless)
        return None

    signature = []
    for node in nodes:
        if node.op == "placeholder":
            signature.append(("input", node_to_idx[node]))
        elif node.op == "call_function":
            encoded_args = []
            int_slots, int_kwarg_names = _int_slots(node.target)
            for arg_idx, arg in enumerate(node.args):
                enc = _encode_arg(arg, arg_idx, int_slots)
                if enc is not None:
                    encoded_args.append(enc)
            # Also encode relevant kwargs (like correction, keepdim)
            for kw, val in (node.kwargs or {}).items():
                if isinstance(val, bool):
                    encoded_args.append(("kw_bool", kw, val))
                elif isinstance(val, (list, tuple)) and all(isinstance(x, int) for x in val):
                    encoded_args.append(("kw_dims", kw, list(val)))
                elif isinstance(val, torch.dtype):
                    encoded_args.append(("kw_dtype", kw, str(val)))
                elif isinstance(val, str):
                    # rounding_mode="floor"/"trunc", approximate="tanh", ...
                    encoded_args.append(("kw_str", kw, val))
                elif isinstance(val, torch.memory_format):
                    encoded_args.append(("kw_memfmt", kw, str(val)))
                elif isinstance(val, int) and kw in int_kwarg_names:
                    # dim passed as kwarg (argmax(x, dim=1)) — same
                    # structural slot as the positional form.
                    encoded_args.append(("kw_int", kw, val))
            signature.append((_op_name(node), encoded_args))
        elif node.op == "output":
            def _collect_output_indices(x):
                if isinstance(x, torch.fx.Node):
                    x = _resolve(x)
                if isinstance(x, torch.fx.Node) and x in node_to_idx:
                    return node_to_idx[x]
                elif isinstance(x, (tuple, list)):
                    return [_collect_output_indices(item) for item in x]
                return None
            out_indices = _collect_output_indices(node.args[0])
            # Output tuple order is consumption spelling, not kernel
            # structure (extraction orders outputs by the ORIGINAL graph's
            # def positions, which the canonical renumbering replaces) —
            # sort top-level indices so output permutations of the same
            # DAG hash identically.
            if isinstance(out_indices, list):
                out_indices = sorted(
                    out_indices,
                    key=lambda x: (x is None, x if isinstance(x, int) else repr(x)),
                )
            signature.append(("output", out_indices))

    return signature


def pattern_hash_for_subgraph(sub_gm) -> str:
    """Content-addressed pattern hash: ops + wiring, ignoring shapes.

    This is the 12-hex hash in repros/canonical/<family>_<hash> dir names.
    """
    dag_signature = compute_dag_signature(sub_gm)
    return hashlib.md5(json.dumps(dag_signature).encode()).hexdigest()[:12]


def shape_hash_for_placeholders(placeholder_info: dict) -> str:
    """8-hex hash of the partition's input shapes+strides+dtypes (shape config id).

    Stride is part of the identity: occurrences of the same pattern at the
    same shapes but different layouts (contiguous vs channels-last) are
    distinct benchmark points and must not collapse to one. Contiguous
    inputs record stride=[] (placeholder_info construction), which is the
    canonical spelling for "contiguous", so the hash is stable across
    captures of the same layout.
    """
    input_shapes = sorted(
        f"{info.get('shape', '?')}:{info.get('stride', [])}:{info.get('dtype', '?')}"
        for info in placeholder_info.values()
    )
    return hashlib.md5(json.dumps(input_shapes).encode()).hexdigest()[:8]



def lift_shape_params(gm):
    """Lift concrete shape literals into _shape_param_N placeholders.

    THE single lifting implementation, run on the canonical (retraced)
    graph. Discriminator is SCHEMA-TYPED, not an op allowlist: any argument
    occupying a SymInt[]-typed parameter slot (per the op's own schema) is a
    shape; int[]-typed slots (reduction dims, permute orders) are structural
    and never lifted. The op's schema declares which of its lists are
    shapes — no list of ops to maintain.

    Returns (gm, shape_params) — gm mutated in place and recompiled.
    """

    def _symint_list_positions(target):
        schema = getattr(target, "_schema", None)
        if schema is None:
            return ()
        out = []
        for i, a in enumerate(schema.arguments):
            if a.kwarg_only:
                continue
            if "SymInt[]" in str(a.type_with_alias()) if hasattr(a, "type_with_alias") else False:
                out.append(i)
        if not out:
            # str(schema) spells SymInt[] even where arg.type collapses to
            # List[int]; parse positions from the schema string.
            sig = str(schema)
            args_str = sig[sig.index("(") + 1: sig.rindex("->")].rsplit(")", 1)[0]
            for i, part in enumerate(args_str.split(",")):
                if "SymInt[]" in part:
                    out.append(i)
        return tuple(out)

    def _should_lift(shape_list):
        if not shape_list:
            return False
        if all(d in (0, 1, -1) for d in shape_list):
            return False
        return True

    g = gm.graph
    shape_params: dict[str, list[int]] = {}
    counter = 0
    last_ph = None
    for n in g.nodes:
        if n.op == "placeholder":
            last_ph = n
    for node in list(g.nodes):
        if node.op != "call_function":
            continue
        # SymInt[] slots on NON-fusible ops (conv stride/padding/dilation,
        # convolution_backward) are op configuration, not output shapes —
        # lifting them would conflate different convs into one pattern.
        # Partition subgraphs never contain such ops (the partitioner
        # excludes them), so this guard is defense-in-depth for any caller
        # handing us a full graph. Pinned by test_canonical_invariants.
        if not partition_node_is_supported(node):
            continue
        for pos in _symint_list_positions(node.target):
            if pos >= len(node.args):
                continue
            arg = node.args[pos]
            if not isinstance(arg, (list, tuple)):
                continue
            if any(isinstance(x, torch.fx.Node) for x in arg):
                continue  # already parametric/dynamic
            shape_list = list(arg)
            if not _should_lift(shape_list):
                continue
            name = f"_shape_param_{counter}"
            counter += 1
            if last_ph is None:
                with g.inserting_before(next(iter(g.nodes))):
                    ph = g.placeholder(name)
            else:
                with g.inserting_after(last_ph):
                    ph = g.placeholder(name)
            ph.meta["val"] = list(shape_list)
            shape_params[name] = list(shape_list)
            last_ph = ph
            args = list(node.args)
            args[pos] = ph
            node.args = tuple(args)
    if shape_params:
        gm.recompile()
    return gm, shape_params


def canonicalize_subgraph(sub_gm, placeholder_info, shape_params=None):
    """ONE make_fx retrace of an extracted partition: the canonical form.

    The settled identity invariant (CORPUS_MIGRATION_PLAN §1): one retrace is
    allowed, BEFORE serialization; thereafter every trace and hash must be
    identical. make_fx is deterministic given identical inputs, so the
    retraced graph is a fixed point: retrace(canonical) == canonical. The
    serialized artifact (repro.py) is generated FROM the canonical form, and
    the pattern hash is computed FROM it — so live-cut hash, artifact hash,
    and any later re-derivation cannot diverge (no spelling alias tables
    needed: reshape->view, dead multi-output getitems, clone placement all
    normalize in the retrace itself).

    Input fidelity is the one care point: fake inputs are built EXACTLY from
    placeholder_info (shape, stride, dtype, device) — wrong strides (e.g.
    assuming contiguous for a channels-last tensor) would change which view
    ops the retrace emits. SymInt inputs (dynamic shapes) are not yet
    handled here: partitions with symint placeholders return the original
    sub_gm unchanged (documented limitation; revisit with the dynamic-shapes
    serialization work).

    Mutation preservation: zero-user mutating ops (copy_) survive make_fx
    because they are outputs of the extracted sub_gm (the 77a691d80 rule) —
    make_fx keeps everything reachable from outputs. Verified by
    scripts/test_partition_outputs.py.

    Returns (canonical sub_gm, remapped placeholder_info), or the originals
    on any retrace failure
    (loudly, on stderr — a non-canonical artifact is better than no artifact,
    and the roundtrip gate will flag it downstream).
    """
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx

    _DTYPES = {
        "torch.float32": torch.float32, "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16, "torch.float64": torch.float64,
        "torch.int64": torch.int64, "torch.int32": torch.int32,
        "torch.int16": torch.int16, "torch.int8": torch.int8,
        "torch.uint8": torch.uint8, "torch.bool": torch.bool,
    }

    # The sub_gm's node metas carry FakeTensors created under the ORIGINAL
    # trace's FakeTensorMode. make_fx requires our fabricated inputs to share
    # that mode (mixing modes is rejected) — so fish the mode out of any
    # placeholder's meta val and fabricate under it.
    _graph_mode = None
    for _n in sub_gm.graph.nodes:
        _v = _n.meta.get("val") if hasattr(_n, "meta") else None
        _m = getattr(_v, "fake_mode", None)
        if _m is not None:
            _graph_mode = _m
            break

    _ph_order = [n for n in sub_gm.graph.nodes if n.op == "placeholder"]

    fake_inputs = []

    for name, info in placeholder_info.items():
        if info.get("dtype") == "symint":
            print(
                f"[canonicalize] symint placeholder {name!r}: retrace skipped "
                f"(dynamic-shape partitions not yet canonicalized)",
                file=sys.stderr,
            )
            return sub_gm, placeholder_info, shape_params
        dtype = _DTYPES.get(info.get("dtype"), torch.float32)
        shape = info.get("shape", [])
        stride = info.get("stride") or None
        try:
            if _graph_mode is not None:
                mode = _graph_mode
            else:
                from torch._guards import detect_fake_mode
                mode = detect_fake_mode() or FakeTensorMode(
                    allow_non_fake_inputs=True)
            dev = info.get("device") or "meta"
            with mode:
                if stride:
                    t = torch.empty_strided(shape, stride, dtype=dtype,
                                            device=dev)
                else:
                    t = torch.empty(shape, dtype=dtype, device=dev)
            fake_inputs.append(t)
        except Exception as exc:
            print(f"[canonicalize] could not fabricate input {name!r}: {exc}",
                  file=sys.stderr)
            return sub_gm, placeholder_info, shape_params

    # Lifted shape params (_shape_param_N, list-valued placeholders) must
    # NOT be traced as inputs — make_fx explodes a list into one scalar
    # placeholder per element, breaking the serialized signature. Instead,
    # PARTIALLY APPLY them: a closure bakes the concrete values in during
    # the retrace (tensor-only trace), and they are RE-LIFTED into
    # placeholders afterward so the canonical graph keeps the exact original
    # signature (and _shapes_config keeps its S(...) entries).
    _shape_param_pos = {}   # position in _ph_order -> (name, concrete value)
    for _i, _ph in enumerate(_ph_order):
        _v = _ph.meta.get("val") if hasattr(_ph, "meta") else None
        if not (hasattr(_v, "shape") and hasattr(_v, "dtype")):
            _val = list(_v) if isinstance(_v, (list, tuple)) else _v
            if _val is None and shape_params and _ph.name in shape_params:
                _val = shape_params[_ph.name]
            if _val is None:
                print(f"[canonicalize] no concrete value for shape param "
                      f"{_ph.name!r} — skipping retrace", file=sys.stderr)
                return sub_gm, placeholder_info, shape_params
            _shape_param_pos[_i] = (_ph.name, _val)

    try:
        with torch.no_grad():
            if _shape_param_pos:
                _names_in_order = [n.name for n in _ph_order]

                def _bound(*tensor_args):
                    full, ti = [], iter(tensor_args)
                    for _i, _nm in enumerate(_names_in_order):
                        if _i in _shape_param_pos:
                            full.append(_shape_param_pos[_i][1])
                        else:
                            full.append(next(ti))
                    return sub_gm(*full)

                canonical = make_fx(_bound, tracing_mode="fake")(*fake_inputs)
            else:
                canonical = make_fx(sub_gm, tracing_mode="fake")(*fake_inputs)
        # make_fx wraps the module callable with pytree flatten/unflatten
        # codegen and names the class after the callable ("<lambda>") — both
        # break standalone serialization (class <lambda> is a SyntaxError;
        # tree_flatten_spec needs the in/out specs). Reset to plain codegen:
        # the graph body is identical, the wrapper disappears.
        canonical.graph.set_codegen(fx.graph.CodeGen())
        canonical.recompile()
        # Re-lift shape params with THE single lifting implementation: the
        # bake+retrace concretized all shape literals; lifting the canonical
        # graph assigns _shape_param_N names/positions by one set of rules
        # for every partition (no bespoke re-insertion).
        canonical, lifted_shape_params = lift_shape_params(canonical)
    except Exception as exc:
        print(
            f"[canonicalize] retrace failed ({type(exc).__name__}: "
            f"{str(exc)[:120]}) — using un-canonicalized subgraph; the "
            f"roundtrip gate will flag any resulting drift",
            file=sys.stderr,
        )
        return sub_gm, placeholder_info, shape_params

    # make_fx renames placeholders (arg0_1-style). placeholder_info is keyed
    # by name and feeds _shapes_config / index inference — remap it to the
    # canonical graph's names POSITIONALLY (make_fx preserves input order),
    # or stride/dtype fidelity silently vanishes from the serialized repro.
    canonical_phs = [
        n for n in canonical.graph.nodes
        if n.op == "placeholder" and not n.name.startswith("_shape_param_")
    ]
    original_items = list(placeholder_info.items())
    if len(canonical_phs) == len(original_items):
        remapped = {
            ph.name: info
            for ph, (_old, info) in zip(canonical_phs, original_items)
        }
        return canonical, remapped, lifted_shape_params
    # Count differs: make_fx materialized lifted shape params as their own
    # placeholders. Remap the TENSOR subset positionally (tensor placeholders
    # keep their relative order); shape-param placeholders need no info entry
    # (serialization emits them as S(...) from their meta vals).
    tensor_phs = [ph for ph in canonical_phs
                  if hasattr(ph.meta.get("val"), "shape")]
    if len(tensor_phs) == len(original_items):
        remapped = {
            ph.name: info
            for ph, (_old, info) in zip(tensor_phs, original_items)
        }
        return canonical, remapped, lifted_shape_params
    print(
        f"[canonicalize] placeholder structure changed in retrace "
        f"({len(original_items)} tensor infos vs {len(tensor_phs)} tensor "
        f"placeholders) — using un-canonicalized subgraph",
        file=sys.stderr,
    )
    return sub_gm, placeholder_info, shape_params


def compute_partition_pattern(comp: list, gm: fx.GraphModule) -> dict | None:
    """Extract one partition's subgraph, canonicalize it (one make_fx
    retrace — see canonicalize_subgraph), and compute the capture-pipeline
    hashes FROM THE CANONICAL FORM.

    Returns dict with pattern_hash, shape_hash, sub_gm (canonical),
    placeholder_info, shape_params — or None if extraction fails.
    """
    result = extract_partition_subgraph(comp, gm)
    if result is None:
        return None
    sub_gm, placeholder_info, shape_params = result
    sub_gm, placeholder_info, shape_params = canonicalize_subgraph(
        sub_gm, placeholder_info, shape_params)
    return {
        "pattern_hash": pattern_hash_for_subgraph(sub_gm),
        "shape_hash": shape_hash_for_placeholders(placeholder_info),
        "sub_gm": sub_gm,
        "placeholder_info": placeholder_info,
        "shape_params": shape_params,
    }


def compute_observed_stats(tensor: torch.Tensor) -> dict | None:
    """Compute observed-value stats for an integer/bool tensor.

    Returns {"min": int, "max": int, "n_unique": int} or None if the tensor
    is not integer/bool dtype or is empty. Keeps it cheap: single pass with
    .min()/.max()/.unique().numel() on the real tensor.
    """
    if tensor.numel() == 0:
        return None
    dtype_name = str(tensor.dtype)
    if not ("int" in dtype_name or "bool" in dtype_name):
        return None
    try:
        # Move to CPU for stats if needed (avoids GPU sync issues with .unique())
        t = tensor.detach()
        if t.is_cuda:
            t = t.cpu()
        t_flat = t.reshape(-1)
        return {
            "min": int(t_flat.min().item()),
            "max": int(t_flat.max().item()),
            "n_unique": int(t_flat.unique().numel()),
        }
    except Exception:
        return None


class _CaptureState:
    def __init__(self, output_dir: str, label: str = "capture", graph_dir: str | None = None,
                 validate: bool = True, capture_only: bool = False):
        self.output_dir = output_dir
        self.label = label
        self.graph_dir = graph_dir
        self.validate = validate
        self.capture_only = capture_only
        self.seen_hashes: set[str] = set()
        # Exact occurrence count per (pattern_hash, shape_hash) across ALL
        # graphs of this capture session — counted BEFORE dedup, so the
        # accounting join (sum standalone_us x occurrences) is auditable
        # from artifacts without a GPU retrace.
        self.occurrence_counts: dict[str, int] = {}
        self.counter = 0
        self.graph_counter = 0
        self.captured: list[dict] = []
        self.dropped: list[dict] = []
        # Stash for real example inputs from compile_fx (keyed by graph counter)
        self._real_inputs_stash: list[torch.Tensor] | None = None
        # Observed stats for the current graph being processed
        self._current_observed_stats: dict[str, dict] = {}
        os.makedirs(output_dir, exist_ok=True)
        if graph_dir:
            os.makedirs(graph_dir, exist_ok=True)

    def stash_real_inputs(self, example_inputs):
        """Store real example inputs from compile_fx for observed-stats collection."""
        real_tensors = []
        for inp in example_inputs:
            if isinstance(inp, torch.Tensor) and not _is_fake_or_meta(inp):
                real_tensors.append(inp)
            else:
                real_tensors.append(None)
        self._real_inputs_stash = real_tensors

    def get_observed_stats_for_placeholders(
        self, gm: fx.GraphModule, placeholder_info: dict[str, dict]
    ) -> dict[str, dict]:
        """Compute observed stats for integer/bool placeholders using stashed real inputs.

        Returns {placeholder_name: {"min": int, "max": int, "n_unique": int}} for
        each integer/bool placeholder where stats could be computed.
        """
        if self._real_inputs_stash is None:
            return {}
        stats = {}
        ph_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
        for idx, node in enumerate(ph_nodes):
            if idx >= len(self._real_inputs_stash):
                break
            real_tensor = self._real_inputs_stash[idx]
            if real_tensor is None:
                continue
            info = placeholder_info.get(node.name, {})
            dtype = info.get("dtype", "")
            if not ("int" in dtype or "bool" in dtype):
                continue
            observed = compute_observed_stats(real_tensor)
            if observed is not None:
                stats[node.name] = observed
        return stats

    def _extract_subgraph(self, origin_nodes: list[fx.Node], gm: fx.GraphModule):
        return extract_partition_subgraph(origin_nodes, gm)

    def _compute_dag_signature(self, gm) -> list:
        return compute_dag_signature(gm)

    @staticmethod
    def _infer_index_bounds(gm, placeholder_info) -> dict[str, int]:
        return infer_index_bounds_from_gm(gm, placeholder_info)

    @staticmethod
    def _infer_permutation_indices(gm, placeholder_info) -> dict[str, int]:
        return infer_permutation_indices_from_gm(gm, placeholder_info)

    def _generate_repro_file(self, gm, placeholder_info, meta, filename, shape_params=None):
        shape_params = shape_params or {}
        code = gm.print_readable(print_output=False)
        import re as _re
        code = _re.sub(r"^class \S+\(torch\.nn\.Module\):",
                       "class Repro(torch.nn.Module):", code, count=1,
                       flags=_re.M)

        # Infer index bounds for int64 placeholders (graph inference = tier 1).
        # constant_values: placeholders where only a CONSTANT is safe
        # (maxpool window-center offsets — random offsets OOB at padded
        # edges -> device-side assert poisons the CUDA context).
        constant_values: dict[str, int] = {}
        index_bounds = infer_index_bounds_from_gm(
            gm, placeholder_info, constants_out=constant_values)
        permutation_indices = self._infer_permutation_indices(gm, placeholder_info)

        # Apply bound hierarchy: observed fallback for integer inputs without
        # graph-inferred bounds. This eliminates the high=512 guess class.
        for name, info in placeholder_info.items():
            dtype = info.get("dtype", "")
            if "int" not in dtype and "bool" not in dtype:
                continue
            if name in index_bounds or name in permutation_indices:
                continue  # Graph inference won — keep it
            observed = info.get("observed")
            if observed is not None:
                # Observed fallback: bound = observed.max + 1
                index_bounds[name] = int(observed["max"]) + 1

        # Replace symbolic shape refs (s0, s1, ...) in annotations with concrete values
        import re
        for name, info in placeholder_info.items():
            if not info.get("shape"):
                continue
            ann_match = re.search(rf'{re.escape(name)}: "(\w+)\[([^\]]+)\]"', code)
            if ann_match:
                sym_dims = ann_match.group(2).split(", ")
                concrete_dims = info["shape"]
                if len(sym_dims) == len(concrete_dims):
                    for sym, conc in zip(sym_dims, concrete_dims):
                        if re.match(r's\d+', sym):
                            code = re.sub(rf'\b{re.escape(sym)}\b', str(conc), code)

        input_lines = []
        ph_names = [n.name for n in gm.graph.nodes if n.op == "placeholder"]
        for name in ph_names:
            if name in shape_params:
                input_lines.append(f"    {shape_params[name]},  # {name}")
                continue
            info = placeholder_info.get(name)
            if info and info["dtype"] != "symint":
                shape = info["shape"]
                stride = info.get("stride", [])
                dtype = info["dtype"]
                device = info.get("device", "cuda")
                if "cuda" in device:
                    device = "cuda"
                if stride and shape:
                    storage_size = sum(
                        s * (d - 1) for s, d in zip(stride, shape) if d > 1
                    ) + 1
                    if "int" in dtype:
                        # int8 = pool offsets (0..8), int64 = indices (inferred or shape[0])
                        default_hi = 9 if "int8" in dtype else max(shape)
                        hi = index_bounds.get(name, default_hi)
                        if name in permutation_indices:
                            input_lines.append(
                                f"    torch.randperm({permutation_indices[name]}, dtype={dtype}, device='{device}')"
                                f".as_strided({shape}, {stride}),  # {name}"
                            )
                        else:
                            input_lines.append(
                                f"    torch.randint(0, {hi}, ({storage_size},), dtype={dtype}, device='{device}')"
                                f".as_strided({shape}, {stride}),  # {name}"
                            )
                    elif "bool" in dtype:
                        input_lines.append(
                            f"    torch.randint(0, 2, ({storage_size},), dtype=torch.bool, device='{device}')"
                            f".as_strided({shape}, {stride}),  # {name}"
                        )
                    else:
                        input_lines.append(
                            f"    torch.randn({storage_size}, dtype={dtype}, device='{device}')"
                            f".as_strided({shape}, {stride}),  # {name}"
                        )
                else:
                    if "int" in dtype:
                        default_hi = 9 if "int8" in dtype else max(shape) if shape else 2
                        hi = index_bounds.get(name, default_hi)
                        if name in permutation_indices:
                            input_lines.append(
                                f"    torch.randperm({permutation_indices[name]}, dtype={dtype}, device='{device}')"
                                f".reshape({shape}),"
                            )
                        else:
                            input_lines.append(
                                f"    torch.randint(0, {hi}, {shape}, dtype={dtype}, device='{device}'),"
                            )
                    elif "bool" in dtype:
                        input_lines.append(
                            f"    torch.randint(0, 2, {shape}, dtype=torch.bool, device='{device}'),"
                        )
                    else:
                        input_lines.append(
                            f"    torch.randn({shape}, dtype={dtype}, device='{device}'),"
                        )
            elif info and info.get("dtype") == "symint":
                # SymInt placeholder — emit the concrete hint value as a plain int
                # This is a scalar dimension (like seq_len) passed to ops like iota/full
                val = info.get("hint", 1)
                input_lines.append(f"    {val},  # {name} (symbolic dim)")
            else:
                input_lines.append(f"    torch.tensor(1),  # {name} (unknown shape)")

        inputs_code = "\n".join(input_lines)

        # Build compact shapes config line (same format as shapes.txt)
        _DTYPE_SHORT = {
            "torch.float32": "f32", "torch.float16": "f16",
            "torch.bfloat16": "bf16", "torch.float64": "f64",
            "torch.int64": "i64", "torch.int32": "i32",
            "torch.int16": "i16", "torch.int8": "i8",
            "torch.bool": "b8", "torch.uint8": "u8",
        }
        # Build the inputs as DATA (compact shared encoding); the T()/S()
        # string is a RENDERING of that data for repro.py documentation —
        # the same value flows to shapes.json structurally, never re-parsed.
        from input_codec import compact_from_spec, render_signature

        # Rewrite live storage keys to small group indices: members of one
        # alias group share an integer "alias" tag so replay allocates ONE
        # buffer per group and as_strided's each member (footprint and
        # locality fidelity for packed-qkv style saved views). Group sizes
        # (true allocation nbytes, from the live storage) are emitted ONCE
        # as alias_group_nbytes — consumers never scan members to size the
        # buffer.
        _group_of: dict[int, int] = {}
        _group_counts: dict[int, int] = {}
        _group_nbytes_by_key: dict[int, int] = {}
        for name in ph_names:
            info_a = placeholder_info.get(name) or {}
            sk = info_a.get("_storage_key")
            if sk is not None:
                _group_counts[sk] = _group_counts.get(sk, 0) + 1
                if "_storage_nbytes" in info_a:
                    _group_nbytes_by_key[sk] = max(
                        _group_nbytes_by_key.get(sk, 0),
                        info_a["_storage_nbytes"])
        for sk, count in _group_counts.items():
            if count > 1:
                _group_of[sk] = len(_group_of)
        alias_group_nbytes = [
            _group_nbytes_by_key.get(sk, 0)
            for sk, _idx in sorted(_group_of.items(), key=lambda kv: kv[1])
        ]

        compact_inputs = []
        for name in ph_names:
            if name in shape_params:
                compact_inputs.append(["S", list(shape_params[name])])
            else:
                info = placeholder_info.get(name)
                if info and info["dtype"] != "symint":
                    spec = {
                        "kind": "tensor",
                        "shape": info["shape"],
                        "dtype": info["dtype"],
                        "stride": info.get("stride", []),
                    }
                    if info.get("storage_offset"):
                        spec["storage_offset"] = info["storage_offset"]
                    sk = info.get("_storage_key")
                    if sk in _group_of:
                        spec["alias_group"] = _group_of[sk]
                    bound = index_bounds.get(name)
                    if name in constant_values:
                        spec["gen"] = {"kind": "constant",
                                       "value": constant_values[name]}
                    elif name in permutation_indices:
                        spec["gen"] = {"kind": "permutation",
                                       "size": permutation_indices[name]}
                    elif "int" in info["dtype"] and bound:
                        spec["gen"] = {"kind": "index", "low": 0,
                                       "high": bound}
                    elif info.get("gen"):
                        # Inference attached a gen directly (float->index
                        # chains: OPT position ids derive from a float
                        # mask; randn would index negative/OOB).
                        spec["gen"] = info["gen"]
                    compact_inputs.append(compact_from_spec(spec))
                elif info and info.get("dtype") == "symint":
                    compact_inputs.append(["sym", info.get("hint", 1)])
        shapes_config_line = render_signature(compact_inputs)

        script = f'''"""
Standalone repro captured via capture_hook.
Label: {self.label}
Pattern hash: {meta.get("pattern_hash", "?")}
Shape hash: {meta.get("shape_hash", "?")}
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = {CURRENT_REPRO_VERSION}
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

{code}


def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
'''
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            f.write(script)

        # Validate: the repro must run in eager without error
        if self.validate:
            try:
                import importlib.util
                import math as _math
                import types
                # Ensure project root is on sys.path so repro_harness is importable
                # (the generated repro uses parents[3] which only works at final install path)
                project_root = str(Path(__file__).resolve().parent)
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                # Use a fake filepath at depth 4+ so the generated code's
                # `Path(__file__).resolve().parents[3]` resolves correctly
                fake_path = os.path.join(project_root, "repros", "canonical", "_validation_", filename)
                code = open(filepath).read()
                compiled_code = compile(code, fake_path, "exec")
                mod = types.ModuleType("_validate")
                mod.__file__ = fake_path
                mod.device = torch.device
                mod.inf = _math.inf
                mod.nan = float("nan")
                exec(compiled_code, mod.__dict__)
                repro_instance = mod.Repro()
                # Explicit config: shapes.json doesn't exist yet (merge
                # creates it later in the canonical dir), so the default
                # path can't be used here — and validating with the same
                # compact entries that BECOME the shapes.json point means
                # we validate exactly what consumers will load.
                from input_codec import spec_from_compact
                from repro_harness import make_inputs_from_config as _mific
                _vcfg = {
                    "inputs": [spec_from_compact(e) for e in compact_inputs]}
                if alias_group_nbytes:
                    _vcfg["alias_group_nbytes"] = alias_group_nbytes
                inputs = mod.make_inputs(shape_config=_vcfg)
                with torch.no_grad():
                    out = repro_instance(*inputs)
                if out is None:
                    print(f"  [capture_hook] WARNING: {filename} forward() returned None")
            except Exception as e:
                raise RuntimeError(
                    f"Captured repro {filename} failed eager validation: {e}\n"
                    f"This indicates a bug in the capture hook (index bounds, shape params, etc.)"
                ) from e

        # Signature returned as DATA alongside the file: consumers (merge ->
        # shapes.json) must never re-derive it by regexing the generated
        # source — that round-trip through rendered Python is exactly the
        # lossy text-parsing the project bans for graphs.
        return filepath, shapes_config_line, compact_inputs, alias_group_nbytes

    def process_graph(self, gm: fx.GraphModule):
        """Called by the hook for each post-grad graph. Partitions and captures."""
        # Compute observed stats for the full graph's integer/bool placeholders
        # using the stashed real inputs from compile_fx.
        full_graph_placeholder_info = placeholder_info_from_gm(gm)
        self._current_observed_stats = self.get_observed_stats_for_placeholders(
            gm, full_graph_placeholder_info
        )

        # Partition ONCE; both the metadata sidecar (node_accounting) and the
        # capture loop below consume the same components.
        components = get_fusion_partitions(gm)

        # Save the FULL post-grad graph (before partitioning) for recovery
        if self.graph_dir:
            full_graph_path = os.path.join(self.graph_dir, f"full_graph_{self.graph_counter:03d}.py")
            self.graph_counter += 1
            try:
                full_code = gm.print_readable(
                    print_output=False,
                    include_stride=True,
                    include_device=True,
                )
                with open(full_graph_path, "w") as f:
                    f.write(full_code)
                try:
                    from full_graph_harness import (
                        infer_full_graph_source,
                        write_full_graph_metadata,
                    )

                    placeholder_info = placeholder_info_from_gm(gm)
                    index_bounds = self._infer_index_bounds(gm, placeholder_info)
                    permutation_indices = self._infer_permutation_indices(gm, placeholder_info)
                    write_full_graph_metadata(
                        full_graph_path,
                        gm,
                        extra={
                            "source": infer_full_graph_source(full_graph_path),
                            "node_accounting": graph_node_accounting(
                                gm, components),
                        },
                        index_bounds=index_bounds,
                        permutation_indices=permutation_indices,
                        observed_stats=self._current_observed_stats or {},
                    )
                except Exception as exc:
                    print(
                        f"[capture_hook] WARNING: could not write full graph metadata "
                        f"for {full_graph_path}: {exc}",
                        file=sys.stderr,
                    )
                # Validate-before-write gate: reload the just-written artifact,
                # run input round-trip (A) + partition round-trip (C) against
                # the live gm, stamp the sidecar "roundtrip": "ok"|"failed: ..".
                # Never deletes the artifact; failures WARN loudly on stderr.
                try:
                    from roundtrip_validation import run_write_gate

                    run_write_gate(full_graph_path, original_gm=gm)
                except Exception as exc:
                    print(
                        f"[capture_hook] WARNING: round-trip gate unavailable "
                        f"for {full_graph_path}: {exc}",
                        file=sys.stderr,
                    )
            except Exception as exc:
                print(
                    f"[capture_hook] WARNING: could not write full graph "
                    f"{full_graph_path}: {exc}",
                    file=sys.stderr,
                )
        # Shared partitioning + hashing (see module-level helpers above):
        # this is the single source of truth for how graphs are cut into
        # canonical repros and how each partition is content-addressed.
        for comp in components:
            is_reduction = partition_has_reduction(comp)

            # Route through compute_partition_pattern — the SINGLE path that
            # extracts, CANONICALIZES (one make_fx retrace; see
            # canonicalize_subgraph), and hashes. The live capture previously
            # extracted+hashed directly here, bypassing canonicalization,
            # which is how live-cut hashes diverged from artifact/retrace
            # hashes (the resnet18 live-vs-artifact hash drift, fixed 2026-06).
            pattern = compute_partition_pattern(comp, gm)
            if pattern is None:
                continue
            sub_gm = pattern["sub_gm"]
            placeholder_info = pattern["placeholder_info"]
            shape_params = pattern["shape_params"]

            # Attach observed stats to placeholder_info for integer/bool inputs.
            # The subgraph's placeholders inherit names from the full graph's nodes,
            # so we can match by name to the full-graph observed stats.
            if self._current_observed_stats:
                for ph_name, info in placeholder_info.items():
                    dtype = info.get("dtype", "")
                    if "int" in dtype or "bool" in dtype:
                        if ph_name in self._current_observed_stats:
                            info["observed"] = self._current_observed_stats[ph_name]

            origin_ops = sorted(str(n.target) for n in comp if n.op == "call_function")

            # Hashes come from the CANONICAL sub_gm (computed inside
            # compute_partition_pattern) — identical to what any later
            # retrace of the serialized artifact will produce.
            pattern_key = pattern["pattern_hash"]
            shape_key = pattern["shape_hash"]

            full_key = f"{pattern_key}_{shape_key}"
            # Count EVERY occurrence (pre-dedup): the repro file is written
            # once per point, but the count is what the accounting joins on.
            self.occurrence_counts[full_key] = (
                self.occurrence_counts.get(full_key, 0) + 1)
            if full_key in self.seen_hashes:
                continue
            self.seen_hashes.add(full_key)

            reduction_infos = []
            for n in comp:
                if n.op == "call_function" and isinstance(n.target, torch._ops.OpOverload):
                    if torch.Tag.reduction in n.target.tags:
                        val = n.meta.get("val", None)
                        shape = list(val.shape) if isinstance(val, torch.Tensor) else []
                        dtype = str(val.dtype) if isinstance(val, torch.Tensor) else "float32"
                        reduction_infos.append({
                            "reduction_type": n.target.overloadpacket.__name__,
                            "ranges": [str(s) for s in shape],
                            "reduction_ranges": [],
                            "dtype": dtype,
                        })

            kind = "reduction" if is_reduction else "pointwise"
            meta = {
                "kind": kind,
                "pattern_hash": pattern_key,
                "shape_hash": shape_key,
                "hash": full_key,
                "n_ops": len(comp),
                "origin_ops": origin_ops,
                "reduction_types": [r["reduction_type"] for r in reduction_infos],
            }

            if is_reduction:
                red_types = "_".join(sorted(set(r["reduction_type"] for r in reduction_infos)))
                filename = f"region_{self.counter:03d}_{red_types}_{pattern_key}_{shape_key}.py"
            else:
                filename = f"region_{self.counter:03d}_pointwise_{pattern_key}_{shape_key}.py"
            self.counter += 1

            try:
                (filepath, signature, compact_inputs,
                 alias_group_nbytes) = self._generate_repro_file(
                    sub_gm, placeholder_info, meta, filename, shape_params)
                entry = {
                    "file": filepath,
                    "kind": kind,
                    "pattern_hash": pattern_key,
                    "shape_hash": shape_key,
                    "hash": full_key,
                    "signature": signature,
                    "inputs": compact_inputs,
                    "reduction_types": meta["reduction_types"],
                    "n_ops": len(comp),
                    "origin_ops": origin_ops,
                }
                if alias_group_nbytes:
                    entry["alias_group_nbytes"] = alias_group_nbytes
                self.captured.append(entry)
            except Exception as e:
                # NEVER silently drop a region: record it so the run report
                # and manifests can show exactly what's missing and why
                # (silent drops are how the dynamic-regions hole and the
                # resnet18 mean-head hole happened).
                self.dropped.append({
                    "filename": filename,
                    "pattern_hash": pattern_key,
                    "shape_hash": shape_key,
                    "reason": f"{type(e).__name__}: {str(e)[:300]}",
                })
                print(
                    f"  [capture_hook] REGION DROPPED ({pattern_key}_"
                    f"{shape_key}): {type(e).__name__}: {str(e)[:160]}",
                    file=sys.stderr,
                )

    def finalize(self):
        """Write index.json for the captured session.

        Drops are FIRST-CLASS in the index: a region that failed
        validation/serialization must be visible to downstream consumers
        (run_recapture status, corpus validation), not just stderr.
        """
        index_path = os.path.join(self.output_dir, "index.json")
        # Stamp each captured entry with its exact occurrence count so the
        # merge can write it into shapes.json (occurrences were null before
        # — the accounting join needed a GPU retrace to recover them).
        for entry in self.captured:
            entry["occurrences"] = self.occurrence_counts.get(entry["hash"], 1)
        payload = {
            "captured": self.captured,
            "dropped": self.dropped,
            "n_captured": len(self.captured),
            "n_dropped": len(self.dropped),
        }
        tmp_path = index_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, index_path)
        n_red = sum(1 for c in self.captured if c.get("kind") == "reduction")
        n_pw = sum(1 for c in self.captured if c.get("kind") == "pointwise")
        msg = (f"[capture_hook] Captured {len(self.captured)} regions "
               f"({n_red} reduction, {n_pw} pointwise) -> {self.output_dir}")
        if self.dropped:
            msg += f" | DROPPED {len(self.dropped)} regions (see index.json)"
        print(msg)


_active_state: _CaptureState | None = None


def install_capture_hook(output_dir: str, label: str = "capture", graph_dir: str | None = None,
                         validate: bool = True, capture_only: bool = False):
    """Install the post-grad capture hook.

    Call this before torch.compile(). All subsequent compilations will have
    their post-grad graphs captured as standalone repros.

    Args:
        output_dir: Directory to write captured repro files.
        label: Human-readable label for the capture session.
        capture_only: If True, raise _CaptureComplete after capture to skip
            expensive inductor compilation. Caller should catch the error.
        validate: If True, run eager validation on captured repros (can trigger CUDA errors).
    """
    global _active_state
    _active_state = _CaptureState(output_dir, label, graph_dir=graph_dir, validate=validate, capture_only=capture_only)

    _old_pass = inductor_config.post_grad_custom_pre_pass

    class _CaptureComplete(Exception):
        """Raised after successful capture to skip expensive compilation."""
        pass

    def _capture_pass(graph_or_gm):
        if isinstance(graph_or_gm, fx.GraphModule):
            gm = graph_or_gm
        elif isinstance(graph_or_gm, fx.Graph):
            if hasattr(graph_or_gm, 'owning_module') and graph_or_gm.owning_module is not None:
                gm = graph_or_gm.owning_module
            else:
                return graph_or_gm
        else:
            return graph_or_gm

        try:
            _active_state.process_graph(copy.deepcopy(gm))
        except Exception as e:
            # A whole-graph processing failure loses EVERY region in the
            # graph — record it as a drop so the fail-hard gate (run_recapture
            # raises pre-merge; merge refuses dropped captures) sees it.
            # Swallowing it here silently merged partial corpora (opus
            # verifier gap #2, 2026-06-11).
            _active_state.dropped.append({
                "filename": None,
                "pattern_hash": None,
                "shape_hash": None,
                "reason": (f"WHOLE-GRAPH process_graph failure "
                           f"(graph {_active_state.graph_counter}): "
                           f"{type(e).__name__}: {str(e)[:300]}"),
            })
            print(f"[capture_hook] Error processing graph (recorded as drop): {e}",
                  file=sys.stderr)

        if _active_state.capture_only:
            _active_state.finalize()
            raise _CaptureComplete()

        if callable(_old_pass):
            return _old_pass(graph_or_gm)
        return graph_or_gm

    inductor_config.post_grad_custom_pre_pass = _capture_pass
    inductor_config.force_disable_caches = True

    # Install compile_fx wrapper to stash real example inputs for observed-stats.
    # The real inputs exist at the compile_fx entry point (before AOTAutograd fakes them).
    import torch._inductor.compile_fx as _compile_fx_module
    _original_compile_fx = _compile_fx_module.compile_fx

    def _capture_compile_fx(model_, example_inputs_, **kwargs):
        """Wrapper that stashes real inputs before calling the real compile_fx."""
        if _active_state is not None:
            _active_state.stash_real_inputs(example_inputs_)
        return _original_compile_fx(model_, example_inputs_, **kwargs)

    _compile_fx_module.compile_fx = _capture_compile_fx
    _active_state._original_compile_fx = _original_compile_fx
    _active_state._compile_fx_module = _compile_fx_module

    import atexit
    atexit.register(_finalize_on_exit)

    print(f"[capture_hook] Installed. Captures will be written to {output_dir}")
    return _active_state


def _finalize_on_exit():
    global _active_state
    if _active_state is not None:
        _active_state.finalize()
        _active_state = None


def uninstall_capture_hook():
    """Remove the capture hook and finalize."""
    global _active_state
    if _active_state is not None:
        # Restore original compile_fx
        if hasattr(_active_state, '_original_compile_fx') and hasattr(_active_state, '_compile_fx_module'):
            _active_state._compile_fx_module.compile_fx = _active_state._original_compile_fx
        _active_state.finalize()
        _active_state = None
    inductor_config.post_grad_custom_pre_pass = None
    inductor_config.force_disable_caches = False


# Auto-install if REPRO_CAPTURE_DIR is set
_auto_dir = os.environ.get("REPRO_CAPTURE_DIR")
if _auto_dir:
    _auto_label = os.environ.get("REPRO_CAPTURE_LABEL", "auto")
    install_capture_hook(_auto_dir, _auto_label)
