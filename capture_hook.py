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
    infer_index_bounds_from_gm,
    infer_permutation_indices_from_gm,
    placeholder_info_from_gm,
)


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

    output_nodes = []
    for n in origin_nodes:
        has_internal_user = any(
            user in needed_nodes and user.op != "output"
            for user in n.users
        )
        if not has_internal_user:
            output_nodes.append(n)
    if not output_nodes:
        output_nodes = origin_nodes

    new_graph = fx.Graph()
    env: dict[fx.Node, fx.Node] = {}
    placeholder_info: dict[str, dict] = {}

    def _resolve_sym(x):
        """Resolve SymInt/SymFloat to concrete int/float."""
        if isinstance(x, (torch.SymInt, torch.SymFloat)):
            return x.node.hint if hasattr(x, 'node') and hasattr(x.node, 'hint') else int(x)
        return int(x)

    def _record_placeholder(name: str, meta: dict) -> None:
        val = meta.get("val", None)
        if val is not None and isinstance(val, torch.Tensor):
            placeholder_info[name] = {
                "shape": [_resolve_sym(s) for s in val.shape],
                "stride": [_resolve_sym(s) for s in val.stride()] if not val.is_contiguous() else [],
                "dtype": str(val.dtype),
                "device": str(val.device),
            }
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


def compute_dag_signature(gm) -> list:
    """Compute a DAG-structure signature for the graph.

    Encodes: for each node in topological order, its op name, which
    predecessor nodes feed each argument (by index), and structural
    literal args (list/tuple values like reduction dims, permute orders).

    Does NOT encode scalar int/float constants (e.g., mul by 3.0,
    eps=1e-6) — those don't affect kernel structure.
    """
    nodes = list(gm.graph.nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    def _encode_arg(arg, arg_idx):
        """Encode an argument for the signature."""
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
        # Scalar int/float constants — don't encode (same kernel regardless)
        return None

    signature = []
    for node in nodes:
        if node.op == "placeholder":
            signature.append(("input", node_to_idx[node]))
        elif node.op == "call_function":
            encoded_args = []
            for arg_idx, arg in enumerate(node.args):
                enc = _encode_arg(arg, arg_idx)
                if enc is not None:
                    encoded_args.append(enc)
            # Also encode relevant kwargs (like correction, keepdim)
            for kw, val in (node.kwargs or {}).items():
                if isinstance(val, bool):
                    encoded_args.append(("kw_bool", kw, val))
                elif isinstance(val, (list, tuple)) and all(isinstance(x, int) for x in val):
                    encoded_args.append(("kw_dims", kw, list(val)))
            signature.append((str(node.target), encoded_args))
        elif node.op == "output":
            def _collect_output_indices(x):
                if isinstance(x, torch.fx.Node) and x in node_to_idx:
                    return node_to_idx[x]
                elif isinstance(x, (tuple, list)):
                    return [_collect_output_indices(item) for item in x]
                return None
            signature.append(("output", _collect_output_indices(node.args[0])))

    return signature


def pattern_hash_for_subgraph(sub_gm) -> str:
    """Content-addressed pattern hash: ops + wiring, ignoring shapes.

    This is the 12-hex hash in repros/canonical/<family>_<hash> dir names.
    """
    dag_signature = compute_dag_signature(sub_gm)
    return hashlib.md5(json.dumps(dag_signature).encode()).hexdigest()[:12]


def shape_hash_for_placeholders(placeholder_info: dict) -> str:
    """8-hex hash of the partition's input shapes+dtypes (shape config id)."""
    input_shapes = sorted(
        f"{info.get('shape', '?')}:{info.get('dtype', '?')}"
        for info in placeholder_info.values()
    )
    return hashlib.md5(json.dumps(input_shapes).encode()).hexdigest()[:8]


def compute_partition_pattern(comp: list, gm: fx.GraphModule) -> dict | None:
    """Extract one partition's subgraph and compute its capture-pipeline hashes.

    Returns dict with pattern_hash, shape_hash, sub_gm, placeholder_info,
    shape_params — or None if extraction fails.
    """
    result = extract_partition_subgraph(comp, gm)
    if result is None:
        return None
    sub_gm, placeholder_info, shape_params = result
    return {
        "pattern_hash": pattern_hash_for_subgraph(sub_gm),
        "shape_hash": shape_hash_for_placeholders(placeholder_info),
        "sub_gm": sub_gm,
        "placeholder_info": placeholder_info,
        "shape_params": shape_params,
    }


class _CaptureState:
    def __init__(self, output_dir: str, label: str = "capture", graph_dir: str | None = None,
                 validate: bool = True, capture_only: bool = False):
        self.output_dir = output_dir
        self.label = label
        self.graph_dir = graph_dir
        self.validate = validate
        self.capture_only = capture_only
        self.seen_hashes: set[str] = set()
        self.counter = 0
        self.graph_counter = 0
        self.captured: list[dict] = []
        os.makedirs(output_dir, exist_ok=True)
        if graph_dir:
            os.makedirs(graph_dir, exist_ok=True)

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
        code = code.replace("class GraphModule(", "class Repro(", 1)

        # Infer index bounds for int64 placeholders
        index_bounds = self._infer_index_bounds(gm, placeholder_info)
        permutation_indices = self._infer_permutation_indices(gm, placeholder_info)

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
        config_parts = []
        for name in ph_names:
            if name in shape_params:
                config_parts.append(f"S({shape_params[name]})")
            else:
                info = placeholder_info.get(name)
                if info and info["dtype"] != "symint":
                    dt = _DTYPE_SHORT.get(info["dtype"], info["dtype"])
                    stride = info.get("stride", [])
                    bound = index_bounds.get(name)
                    gen_kwarg = ""
                    if name in permutation_indices:
                        gen_kwarg = f", gen=Perm({permutation_indices[name]})"
                    elif "int" in info["dtype"] and bound:
                        gen_kwarg = f", gen=Index({bound})"
                    elif "int8" in info["dtype"]:
                        gen_kwarg = ", gen=Index(9)"
                    if stride and info["shape"]:
                        config_parts.append(f"T({info['shape']}, {dt}, stride={tuple(stride)}{gen_kwarg})")
                    else:
                        config_parts.append(f"T({info['shape']}, {dt}{gen_kwarg})")
                elif info and info.get("dtype") == "symint":
                    config_parts.append(f"S([{info.get('hint', 1)}])")
        shapes_config_line = f"({', '.join(config_parts)})"

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

_repro_version = 2
_shapes_config = "{shapes_config_line}"

{code}


def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


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
                inputs = mod._default_make_inputs()
                with torch.no_grad():
                    out = repro_instance(*inputs)
                if out is None:
                    print(f"  [capture_hook] WARNING: {filename} forward() returned None")
            except Exception as e:
                raise RuntimeError(
                    f"Captured repro {filename} failed eager validation: {e}\n"
                    f"This indicates a bug in the capture hook (index bounds, shape params, etc.)"
                ) from e

        return filepath

    def process_graph(self, gm: fx.GraphModule):
        """Called by the hook for each post-grad graph. Partitions and captures."""
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
                        extra={"source": infer_full_graph_source(full_graph_path)},
                        index_bounds=index_bounds,
                        permutation_indices=permutation_indices,
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
        components = get_fusion_partitions(gm)

        for comp in components:
            is_reduction = partition_has_reduction(comp)

            result = self._extract_subgraph(comp, gm)
            if result is None:
                continue
            sub_gm, placeholder_info, shape_params = result

            origin_ops = sorted(str(n.target) for n in comp if n.op == "call_function")

            # DAG-structure hash: encodes op names + wiring (which input feeds which arg)
            # This ensures two graphs with same ops but different connectivity get different hashes
            pattern_key = pattern_hash_for_subgraph(sub_gm)
            shape_key = shape_hash_for_placeholders(placeholder_info)

            full_key = f"{pattern_key}_{shape_key}"
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
                filepath = self._generate_repro_file(sub_gm, placeholder_info, meta, filename, shape_params)
                self.captured.append({
                    "file": filepath,
                    "kind": kind,
                    "pattern_hash": pattern_key,
                    "shape_hash": shape_key,
                    "hash": full_key,
                    "reduction_types": meta["reduction_types"],
                    "n_ops": len(comp),
                    "origin_ops": origin_ops,
                })
            except Exception as e:
                print(f"  [capture_hook] Failed to extract region: {e}")

    def finalize(self):
        """Write index.json for the captured session."""
        index_path = os.path.join(self.output_dir, "index.json")
        with open(index_path, "w") as f:
            json.dump(self.captured, f, indent=2)
        n_red = sum(1 for c in self.captured if c.get("kind") == "reduction")
        n_pw = sum(1 for c in self.captured if c.get("kind") == "pointwise")
        print(f"[capture_hook] Captured {len(self.captured)} regions ({n_red} reduction, {n_pw} pointwise) -> {self.output_dir}")


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
            print(f"[capture_hook] Error processing graph: {e}")

        if _active_state.capture_only:
            _active_state.finalize()
            raise _CaptureComplete()

        if callable(_old_pass):
            return _old_pass(graph_or_gm)
        return graph_or_gm

    inductor_config.post_grad_custom_pre_pass = _capture_pass
    inductor_config.force_disable_caches = True

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
        _active_state.finalize()
        _active_state = None
    inductor_config.post_grad_custom_pre_pass = None
    inductor_config.force_disable_caches = False


# Auto-install if REPRO_CAPTURE_DIR is set
_auto_dir = os.environ.get("REPRO_CAPTURE_DIR")
if _auto_dir:
    _auto_label = os.environ.get("REPRO_CAPTURE_LABEL", "auto")
    install_capture_hook(_auto_dir, _auto_label)
