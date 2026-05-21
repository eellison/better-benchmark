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


class _CaptureState:
    def __init__(self, output_dir: str, label: str = "capture", graph_dir: str | None = None,
                 validate: bool = True):
        self.output_dir = output_dir
        self.label = label
        self.graph_dir = graph_dir
        self.validate = validate
        self.seen_hashes: set[str] = set()
        self.counter = 0
        self.graph_counter = 0
        self.captured: list[dict] = []
        os.makedirs(output_dir, exist_ok=True)
        if graph_dir:
            os.makedirs(graph_dir, exist_ok=True)

    def _extract_subgraph(self, origin_nodes: list[fx.Node], gm: fx.GraphModule):
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

    def _compute_dag_signature(self, gm) -> list:
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

    def _infer_index_bounds(self, gm, placeholder_info) -> dict[str, int]:
        """Infer valid index bounds for integer placeholders by inspecting consumers.

        Traces up to 3 hops downstream to find scatter/gather/embedding consumers.
        Also handles:
        - int8 pool offsets (_low_memory_max_pool_offsets_to_indices)
        - Multi-hop: placeholder → clone/reshape → scatter_add
        """
        _INDEX_CONSUMERS = {
            torch.ops.aten.scatter.src: (0, 1),        # (target_arg, dim_arg)
            torch.ops.aten.scatter.value: (0, 1),
            torch.ops.aten.scatter_add.default: (0, 1),
            torch.ops.aten.gather.default: (0, 1),
            torch.ops.aten.index_put.default: (0, None),
            torch.ops.aten.index.Tensor: (0, None),
            torch.ops.aten.index_select.default: (0, 1),
            torch.ops.aten.embedding.default: (0, None),
        }

        # Ops that pass index values through unchanged
        _PASSTHROUGH_OPS = {
            torch.ops.aten.clone.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten.view.default,
            torch.ops.aten.expand.default,
            torch.ops.aten.slice.Tensor,
            torch.ops.aten.unsqueeze.default,
            torch.ops.aten.squeeze.default,
            torch.ops.aten.squeeze.dim,
            torch.ops.aten.select.int,
            torch.ops.aten.permute.default,
            torch.ops.aten.contiguous.default,
            torch.ops.aten.t.default,
            torch.ops.aten.transpose.int,
            torch.ops.aten.where.self,
        }

        bounds = {}
        ph_nodes = {n.name: n for n in gm.graph.nodes if n.op == "placeholder"}

        def _node_shape(n):
            if not hasattr(n, "meta"):
                return None
            val = n.meta.get("val")
            if isinstance(val, torch.Tensor):
                return list(val.shape)
            if n.op == "call_function" and n.target in {
                torch.ops.aten.empty.memory_format,
                torch.ops.aten.full.default,
            }:
                if n.args and isinstance(n.args[0], (list, tuple)):
                    return list(n.args[0])
            return None

        for name, node in ph_nodes.items():
            info = placeholder_info.get(name, {})
            dtype = info.get("dtype", "")
            if "int" not in dtype:
                continue

            # For int8: almost always pool offsets. Infer from pool kernel size.
            if "int8" in dtype:
                for user in node.users:
                    if user.op == "call_function":
                        target_name = str(user.target)
                        if "max_pool_offsets_to_indices" in target_name:
                            # Args: (offsets, kernel_size, ...) — bound = prod(kernel_size)
                            if len(user.args) >= 2 and isinstance(user.args[1], (list, tuple)):
                                ks = user.args[1]
                                bounds[name] = int(ks[0] * ks[1]) if len(ks) >= 2 else int(ks[0])
                                break
                if name not in bounds:
                    bounds[name] = 9  # default for 3x3 pool
                continue

            # For int64/int32: trace downstream up to 3 hops to find index consumers
            def _find_bound(start_node, max_hops=3):
                frontier = [start_node]
                for _hop in range(max_hops):
                    next_frontier = []
                    for n in frontier:
                        for user in n.users:
                            if user.op != "call_function":
                                continue
                            target = user.target

                            if target == torch.ops.aten.embedding.default:
                                weight_arg = user.args[0] if user.args else None
                                if weight_arg and hasattr(weight_arg, 'meta'):
                                    val = weight_arg.meta.get('val')
                                    if isinstance(val, torch.Tensor) and len(val.shape) > 0:
                                        return int(val.shape[0])

                            if target == torch.ops.aten.index.Tensor:
                                if len(user.args) >= 2:
                                    target_shape = _node_shape(user.args[0])
                                    indices = user.args[1]
                                    if target_shape and isinstance(indices, (list, tuple)):
                                        for dim, index_node in enumerate(indices):
                                            if index_node is n and dim < len(target_shape):
                                                return int(target_shape[dim])

                            if target == torch.ops.aten.gather.default:
                                # If `n` is the source tensor, gather passes
                                # those values through. Their valid bound comes
                                # from the gathered result's downstream index
                                # consumers, not from gather's index dimension.
                                if user.args and user.args[0] is n:
                                    next_frontier.append(user)
                                    continue

                            if target in _INDEX_CONSUMERS:
                                target_arg_idx, dim_arg_idx = _INDEX_CONSUMERS[target]
                                if len(user.args) > target_arg_idx:
                                    target_node = user.args[target_arg_idx]
                                    target_shape = _node_shape(target_node)
                                    if target_shape:
                                        if dim_arg_idx is not None and len(user.args) > dim_arg_idx:
                                            dim = user.args[dim_arg_idx]
                                            if isinstance(dim, int) and dim < len(target_shape):
                                                return int(target_shape[dim])
                                        else:
                                            return int(target_shape[0])

                            # Passthrough: trace further
                            if target in _PASSTHROUGH_OPS:
                                next_frontier.append(user)
                    frontier = next_frontier
                    if not frontier:
                        break
                return None

            bound = _find_bound(node)
            if bound is not None:
                bounds[name] = bound

        return bounds

    def _infer_permutation_indices(self, gm, placeholder_info) -> dict[str, int]:
        """Find integer placeholders that must be valid permutations.

        Pattern:
            inverse = empty([N], dtype=int64)
            inverse = index_put(inverse, [idx], iota(N))
            use(inverse)

        Random randint indices can leave holes in `inverse`; those holes contain
        uninitialized values and can trigger device asserts when used later as
        indices. A randperm preserves the intended inverse-permutation shape.
        """
        permutation_sizes = {}
        ph_nodes = {n.name: n for n in gm.graph.nodes if n.op == "placeholder"}

        def _shape_from_alloc(n):
            if n.op != "call_function":
                return None
            if n.target not in {
                torch.ops.aten.empty.memory_format,
                torch.ops.aten.full.default,
            }:
                return None
            if not n.args or not isinstance(n.args[0], (list, tuple)):
                return None
            return list(n.args[0])

        def _iota_size(n):
            if n.op != "call_function":
                return None
            if n.target != torch.ops.prims.iota.default:
                return None
            if n.args and isinstance(n.args[0], int):
                return int(n.args[0])
            return None

        for name, node in ph_nodes.items():
            info = placeholder_info.get(name, {})
            dtype = info.get("dtype", "")
            if "int" not in dtype:
                continue
            for user in node.users:
                if user.op != "call_function" or user.target != torch.ops.aten.index_put.default:
                    continue
                if len(user.args) < 3:
                    continue
                indices = user.args[1]
                if not isinstance(indices, (list, tuple)) or node not in indices:
                    continue
                target_shape = _shape_from_alloc(user.args[0])
                iota_size = _iota_size(user.args[2])
                if target_shape and iota_size and target_shape[0] == iota_size:
                    permutation_sizes[name] = iota_size
        return permutation_sizes

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
                    if "int" in info["dtype"] and bound:
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
                full_code = gm.print_readable(print_output=False)
                with open(full_graph_path, "w") as f:
                    f.write(full_code)
            except Exception:
                pass
        from torch._inductor.fx_passes.fusion_regions import is_fusible_node
        from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
        from torch.fx.passes.operator_support import create_op_support

        import operator
        _TRANSPARENT_OPS = {
            operator.getitem,
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

        def _is_supported(_submodules, node):
            if node.op == "call_function" and node.target in _TRANSPARENT_OPS:
                return True
            return is_fusible_node(node)

        def _has_reduction(nodes):
            for n in nodes:
                if n.op == "call_function" and isinstance(n.target, torch._ops.OpOverload):
                    if torch.Tag.reduction in n.target.tags:
                        return True
            return False

        support = create_op_support(_is_supported)
        partitioner = CapabilityBasedPartitioner(
            gm, support, allows_single_node_partition=True,
        )
        partitions = partitioner.propose_partitions()
        components = [list(p.nodes.keys()) for p in partitions]

        # Split each partition into connected components based on data flow.
        # The CapabilityBasedPartitioner groups ALL reachable supported nodes
        # into one partition even if they have no data dependency. Two independent
        # chains (e.g., reshape+transpose of tensor A, reshape+transpose of tensor B)
        # should be separate repros since inductor compiles them as separate kernels.
        def _split_connected_components(nodes):
            """Split a list of nodes into connected components by data flow."""
            node_set = set(nodes)
            # Build adjacency: two nodes in the partition are connected if one
            # feeds directly into another (producer → consumer).
            from collections import deque
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

        split_components = []
        for comp in components:
            sub_components = _split_connected_components(comp)
            split_components.extend(sub_components)
        components = split_components

        def _has_real_compute(nodes):
            """Check if a partition has at least one non-transparent compute op."""
            for n in nodes:
                if n.op != "call_function":
                    continue
                if n.target in _TRANSPARENT_OPS:
                    continue
                # It's a real compute op (pointwise, reduction, etc.)
                return True
            return False

        for comp in components:
            if not _has_real_compute(comp):
                continue

            is_reduction = _has_reduction(comp)

            result = self._extract_subgraph(comp, gm)
            if result is None:
                continue
            sub_gm, placeholder_info, shape_params = result

            origin_ops = sorted(str(n.target) for n in comp if n.op == "call_function")

            # DAG-structure hash: encodes op names + wiring (which input feeds which arg)
            # This ensures two graphs with same ops but different connectivity get different hashes
            dag_signature = self._compute_dag_signature(sub_gm)
            pattern_key = hashlib.md5(
                json.dumps(dag_signature).encode()
            ).hexdigest()[:12]

            input_shapes = sorted(
                f"{info.get('shape', '?')}:{info.get('dtype', '?')}"
                for info in placeholder_info.values()
            )
            shape_key = hashlib.md5(
                json.dumps(input_shapes).encode()
            ).hexdigest()[:8]

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
                         validate: bool = True):
    """Install the post-grad capture hook.

    Call this before torch.compile(). All subsequent compilations will have
    their post-grad graphs captured as standalone repros.

    Args:
        output_dir: Directory to write captured repro files.
        label: Human-readable label for the capture session.
        validate: If True, run eager validation on captured repros (can trigger CUDA errors).
    """
    global _active_state
    _active_state = _CaptureState(output_dir, label, graph_dir=graph_dir, validate=validate)

    _old_pass = inductor_config.post_grad_custom_pre_pass

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
