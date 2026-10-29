"""Normalize proven metadata operations and match scalar-preserving MM DAGs."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import networkx as nx
import torch
from torch import fx
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import tree_leaves

from capture_hook import lift_shape_params
from repro_harness import make_inputs_from_config


def tensor_metadata(value):
    if not isinstance(value, torch.Tensor):
        return None
    return (
        tuple(value.shape),
        tuple(value.stride()),
        str(value.dtype),
        value.storage_offset(),
    )


def trace_repro(path: Path, config: dict):
    spec = importlib.util.spec_from_file_location("curate_" + path.parent.name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = make_inputs_from_config(config)
    tensor_positions = [i for i, v in enumerate(args) if isinstance(v, torch.Tensor)]

    def bound(*tensors):
        values = list(args)
        for i, value in zip(tensor_positions, tensors):
            values[i] = value
        return module.Repro()(*values)

    gm = make_fx(bound, tracing_mode="fake")(*(args[i] for i in tensor_positions))
    gm.graph.set_codegen(fx.graph.CodeGen())
    gm.recompile()
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    infos = {
        n.name: copy.deepcopy(config["inputs"][i])
        for n, i in zip(placeholders, tensor_positions)
    }
    positions = {n.name: i for n, i in zip(placeholders, tensor_positions)}
    return gm, infos, positions


_PERMUTATIONS = {"aten.permute.default", "aten.t.default", "aten.transpose.int"}
_ALIAS_VIEWS = _PERMUTATIONS | {
    "aten.view.default",
    "aten.reshape.default",
    "aten._unsafe_view.default",
    "aten.expand.default",
    "aten.squeeze.default",
    "aten.squeeze.dim",
    "aten.unsqueeze.default",
    "aten.slice.Tensor",
    "aten.alias.default",
    "aten.detach.default",
}
_RESHAPES = {
    "aten.view.default",
    "aten._unsafe_view.default",
    "aten.squeeze.default",
    "aten.squeeze.dim",
    "aten.unsqueeze.default",
}
_FLOAT_CASTS = {"prims.convert_element_type.default", "aten._to_copy.default"}


def _refresh_info(node, info):
    value = node.meta["val"]
    info.update(
        shape=list(value.shape),
        stride=list(value.stride()),
        storage_offset=value.storage_offset(),
        dtype=str(value.dtype),
        device=str(value.device),
    )


def _freeze(value):
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_freeze(x) for x in value))
    if isinstance(value, dict):
        return tuple((k, _freeze(v)) for k, v in sorted(value.items()))
    return value


def normalize_graph(gm, infos):
    """Only rewrites validated by tensor metadata or internal pure CSE.

    Input permutations become strided input specs; explicit generators whose
    logical ordering matters are kept. Views are removed only if their entire
    metadata matches, or collapsed when a direct view has identical metadata.
    """
    changes = []
    # Reduction axes are a set; singleton reduced axes can be represented
    # consistently by keepdim=True and an explicit metadata view.
    for node in list(gm.graph.nodes):
        if str(node.target) not in {
            "aten.mean.dim",
            "aten.sum.dim_IntList",
            "aten.amax.default",
            "aten.amin.default",
        }:
            continue
        source = node.args[0]
        value = node.meta.get("val")
        if (
            not isinstance(source, fx.Node)
            or not isinstance(source.meta.get("val"), torch.Tensor)
            or not isinstance(value, torch.Tensor)
        ):
            continue
        rank = source.meta["val"].ndim
        if rank == 0:
            continue
        dims = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        if dims is None or dims == []:
            axes = list(range(rank))
        elif isinstance(dims, (list, tuple)) and all(isinstance(d, int) for d in dims):
            axes = sorted(d % rank for d in dims)
        else:
            continue
        keep = node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        args = list(node.args)
        while len(args) < 3:
            args.append(None)
        args[1], args[2] = axes, True
        kwargs = {k: v for k, v in node.kwargs.items() if k not in {"dim", "keepdim"}}
        probe = node.target(source.meta["val"], *args[1:], **kwargs)
        restored = probe.view(list(value.shape))
        if tensor_metadata(restored) != tensor_metadata(value):
            continue
        node.args, node.kwargs = tuple(args), kwargs
        if not keep:
            users = list(node.users)
            with gm.graph.inserting_after(node):
                view = gm.graph.call_function(
                    torch.ops.aten.view.default, (node, list(value.shape))
                )
            view.meta = copy.copy(node.meta)
            node.meta["val"] = probe
            for user in users:
                user.replace_input_with(node, view)
            changes.append("canonical_reduction_keepdim")
        if axes != dims:
            changes.append("canonical_reduction_axes")

    initial_output = next(n for n in gm.graph.nodes if n.op == "output")
    exposed_storages = {
        n.meta["val"].untyped_storage()._cdata
        for n in tree_leaves(initial_output.args[0])
        if isinstance(n, fx.Node) and isinstance(n.meta.get("val"), torch.Tensor)
    }
    for node in list(gm.graph.nodes):
        if str(node.target) not in {
            "aten._adaptive_avg_pool2d.default",
            "aten.adaptive_avg_pool2d.default",
        }:
            continue
        if not node.args or not isinstance(node.args[0], fx.Node):
            continue
        source = node.args[0]
        value = node.meta.get("val")
        if (
            tensor_metadata(value) == tensor_metadata(source.meta.get("val"))
            and isinstance(value, torch.Tensor)
            and value.untyped_storage()._cdata not in exposed_storages
        ):
            node.replace_all_uses_with(source)
            gm.graph.erase_node(node)
            changes.append("internal_identity_adaptive_pool")
    input_transforms = {n.name: [] for n in gm.graph.nodes if n.op == "placeholder"}
    for ph in [n for n in gm.graph.nodes if n.op == "placeholder"]:
        info = infos[ph.name]
        value = ph.meta.get("val")
        gen = info.get("gen") or info.get("generator") or {}
        if (
            not isinstance(value, torch.Tensor)
            or not value.is_floating_point()
            or info.get("exact")
            or info.get("data")
            or gen.get("kind") not in (None, "randn", "positive")
        ):
            continue
        while len(ph.users) == 1:
            node = next(iter(ph.users))
            if str(node.target) in _FLOAT_CASTS and len(node.users) == 1:
                permutation = next(iter(node.users))
                if str(permutation.target) in _PERMUTATIONS:
                    ph_value = ph.meta["val"]
                    transformed = permutation.target(
                        ph_value, *permutation.args[1:], **permutation.kwargs
                    )
                    cast_result = node.target(
                        transformed, *node.args[1:], **node.kwargs
                    )
                    if tensor_metadata(cast_result) == tensor_metadata(
                        permutation.meta.get("val")
                    ):
                        input_transforms[ph.name].append(
                            {
                                "op": str(permutation.target),
                                "args": list(permutation.args[1:]),
                            }
                        )
                        ph.meta["val"] = transformed
                        node.meta["val"] = cast_result
                        _refresh_info(ph, info)
                        permutation.replace_all_uses_with(node)
                        gm.graph.erase_node(permutation)
                        changes.append("input_permutation_through_float_cast")
                        continue
            if str(node.target) not in _PERMUTATIONS:
                break
            input_transforms[ph.name].append(
                {"op": str(node.target), "args": list(node.args[1:])}
            )
            ph.meta["val"] = node.meta["val"]
            _refresh_info(ph, info)
            node.replace_all_uses_with(ph)
            gm.graph.erase_node(node)
            changes.append("input_permutation")

    changed = True
    while changed:
        changed = False
        for node in list(gm.graph.nodes):
            name = str(node.target)
            if (
                name not in _ALIAS_VIEWS
                or not node.args
                or not isinstance(node.args[0], fx.Node)
            ):
                continue
            parent = node.args[0]
            value, before = node.meta.get("val"), parent.meta.get("val")
            if tensor_metadata(value) is None or tensor_metadata(before) is None:
                continue
            if tensor_metadata(value) == tensor_metadata(before):
                node.replace_all_uses_with(parent)
                gm.graph.erase_node(node)
                changes.append("identity_view")
                changed = True
                continue
            if (
                name not in _RESHAPES
                or not before.is_contiguous()
                or not value.is_contiguous()
            ):
                continue
            candidate = before.view(list(value.shape))
            if tensor_metadata(candidate) != tensor_metadata(value):
                continue
            if str(parent.target) in {"aten.view.default", "aten._unsafe_view.default"}:
                base = parent.args[0]
                if not base.meta["val"].is_contiguous():
                    continue
                probe = base.meta["val"].view(list(value.shape))
                if tensor_metadata(probe) == tensor_metadata(value):
                    node.args = (base, list(value.shape))
                    node.kwargs = {}
                    node.target = torch.ops.aten.view.default
                    changes.append("reshape_chain")
                    if not parent.users:
                        gm.graph.erase_node(parent)
                    changed = True
                    continue
            if name != "aten.view.default":
                node.target = torch.ops.aten.view.default
                node.args = (parent, list(value.shape))
                node.kwargs = {}
                changes.append("contiguous_shape_view")
                changed = True
    gm.graph.eliminate_dead_code()

    output = next(n for n in gm.graph.nodes if n.op == "output")
    output_storages = {
        v.untyped_storage()._cdata
        for n in tree_leaves(output.args[0])
        if isinstance(n, fx.Node)
        for v in [n.meta.get("val")]
        if isinstance(v, torch.Tensor)
    }
    expressions = {}
    for node in list(gm.graph.nodes):
        tags = getattr(node.target, "tags", ())
        value = node.meta.get("val")
        if (
            node.op != "call_function"
            or torch.Tag.pointwise not in tags
            or any("nondeterministic" in str(t) for t in tags)
            or not isinstance(value, torch.Tensor)
            or value.untyped_storage()._cdata in output_storages
        ):
            continue
        schema = getattr(node.target, "_schema", None)
        if schema and any(
            a.alias_info and a.alias_info.is_write for a in schema.arguments
        ):
            continue
        key = (node.target, _freeze(node.args), _freeze(node.kwargs))
        previous = expressions.get(key)
        if previous is not None and tensor_metadata(
            previous.meta["val"]
        ) == tensor_metadata(value):
            node.replace_all_uses_with(previous)
            gm.graph.erase_node(node)
            changes.append("internal_pointwise_cse")
        else:
            expressions[key] = node
    gm.graph.eliminate_dead_code()
    gm, shape_params = lift_shape_params(gm)
    gm.graph.set_codegen(fx.graph.CodeGen())
    gm.recompile()
    live_names = {n.name for n in gm.graph.nodes if n.op == "placeholder"}
    infos = {k: v for k, v in infos.items() if k in live_names}
    return gm, infos, shape_params, input_transforms, changes


def semantic_graph(gm):
    graph = nx.DiGraph()
    serial = 0

    def add(label, **attrs):
        nonlocal serial
        key = ("aux", serial)
        serial += 1
        graph.add_node(key, label=label, **attrs)
        return key

    def link(child, dest, slot):
        edge = add("edge:" + str(slot))
        graph.add_edge(child, edge)
        graph.add_edge(edge, dest)

    def value(arg):
        if isinstance(arg, fx.Node):
            return arg
        if isinstance(arg, (list, tuple)):
            node = add("sequence:" + type(arg).__name__)
            for i, item in enumerate(arg):
                link(value(item), node, i)
            return node
        return add("literal:" + type(arg).__name__ + ":" + repr(arg))

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            kind = (
                "tensor" if isinstance(node.meta.get("val"), torch.Tensor) else "shape"
            )
            graph.add_node(node, label="input:" + kind, role="input")
            continue
        if node.op == "output":
            graph.add_node(node, label="output", role="output")
            outputs = node.args[0]
            if isinstance(outputs, (tuple, list)):
                for i, out in enumerate(outputs):
                    edge = add("edge:output", output_index=i)
                    graph.add_edge(value(out), edge)
                    graph.add_edge(edge, node)
            else:
                link(value(outputs), node, "output")
            continue
        graph.add_node(node, label="op:" + str(node.target), role="op")
        symmetric = str(node.target) in {
            "aten.mul.Tensor",
            "prims.mul.default",
            "prims.add.default",
        }
        if str(node.target) == "aten.add.Tensor":
            alpha = node.kwargs.get("alpha", node.args[2] if len(node.args) > 2 else 1)
            symmetric = not isinstance(alpha, fx.Node) and alpha == 1
        for i, arg in enumerate(node.args):
            link(
                value(arg), node, "commuted" if symmetric and i < 2 else "pos:" + str(i)
            )
        for key, arg in sorted(node.kwargs.items()):
            link(value(arg), node, "kw:" + key)
    return graph


def match_graphs(reference, candidate):
    """An isomorphism reference -> candidate (node -> node), or None.

    Nodes are relabeled to integers in insertion order first. networkx walks
    node *sets*, and fx.Node hashes are memory addresses, so for a symmetric
    graph (two identical branches) the first isomorphism found, and thus the
    recorded input mapping, would otherwise change from run to run.
    """
    ref = nx.convert_node_labels_to_integers(reference, label_attribute="node")
    cand = nx.convert_node_labels_to_integers(candidate, label_attribute="node")
    matcher = nx.algorithms.isomorphism.DiGraphMatcher(
        ref,
        cand,
        node_match=nx.algorithms.isomorphism.categorical_node_match("label", ""),
    )
    mapping = next(matcher.isomorphisms_iter(), None)
    if mapping is None:
        return None
    return {ref.nodes[a]["node"]: cand.nodes[b]["node"] for a, b in mapping.items()}


def has_rng(gm):
    return any(
        any("nondeterministic" in str(t) for t in getattr(n.target, "tags", ()))
        for n in gm.graph.nodes
    )


def graph_digest(graph):
    return nx.weisfeiler_lehman_graph_hash(graph, node_attr="label", iterations=6)
