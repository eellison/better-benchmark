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
    def __init__(self, output_dir: str, label: str = "capture"):
        self.output_dir = output_dir
        self.label = label
        self.seen_hashes: set[str] = set()
        self.counter = 0
        self.captured: list[dict] = []
        os.makedirs(output_dir, exist_ok=True)

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
                placeholder_info[name] = {
                    "shape": [],
                    "stride": [],
                    "dtype": "symint",
                    "device": "cpu",
                }

        def _ensure_in_env(x: Any) -> Any:
            if isinstance(x, fx.Node):
                if x in env:
                    return env[x]
                ph = new_graph.placeholder(x.name)
                ph.meta = copy.copy(x.meta) if x.meta else {}
                env[x] = ph
                _record_placeholder(x.name, x.meta or {})
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

    def _generate_repro_file(self, gm, placeholder_info, meta, filename, shape_params=None):
        shape_params = shape_params or {}
        code = gm.print_readable(print_output=False)
        code = code.replace("class GraphModule(", "class Repro(", 1)

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
                        input_lines.append(
                            f"    torch.randint(0, 2, ({storage_size},), dtype={dtype}, device='{device}')"
                            f".as_strided({shape}, {stride}),  # {name}"
                        )
                    else:
                        input_lines.append(
                            f"    torch.randn({storage_size}, dtype={dtype}, device='{device}')"
                            f".as_strided({shape}, {stride}),  # {name}"
                        )
                else:
                    if "int" in dtype:
                        input_lines.append(
                            f"    torch.randint(0, 2, {shape}, dtype={dtype}, device='{device}'),"
                        )
                    elif "bool" in dtype:
                        input_lines.append(
                            f"    torch.randint(0, 2, {shape}, dtype=torch.bool, device='{device}'),"
                        )
                    else:
                        input_lines.append(
                            f"    torch.randn({shape}, dtype={dtype}, device='{device}'),"
                        )
            else:
                input_lines.append(f"    torch.tensor(1),  # {name} (unknown shape)")

        inputs_code = "\n".join(input_lines)

        script = f'''"""
Standalone repro captured via capture_hook.
Label: {self.label}
Pattern hash: {meta.get("pattern_hash", "?")}
Shape hash: {meta.get("shape_hash", "?")}
"""
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

{code}


def make_inputs():
    return [
{inputs_code}
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
'''
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            f.write(script)
        return filepath

    def process_graph(self, gm: fx.GraphModule):
        """Called by the hook for each post-grad graph. Partitions and captures."""
        from torch._inductor.fx_passes.fusion_regions import is_fusible_node
        from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
        from torch.fx.passes.operator_support import create_op_support

        def _is_supported(_submodules, node):
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

        from torch._inductor.fx_passes.fusion_regions import is_view_node

        for comp in components:
            if all(is_view_node(n) or n.op != "call_function" for n in comp):
                continue

            is_reduction = _has_reduction(comp)

            result = self._extract_subgraph(comp, gm)
            if result is None:
                continue
            sub_gm, placeholder_info, shape_params = result

            origin_ops = sorted(str(n.target) for n in comp if n.op == "call_function")

            pattern_key = hashlib.md5(
                json.dumps({"ops": origin_ops}, sort_keys=True).encode()
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


def install_capture_hook(output_dir: str, label: str = "capture"):
    """Install the post-grad capture hook.

    Call this before torch.compile(). All subsequent compilations will have
    their post-grad graphs captured as standalone repros.

    Args:
        output_dir: Directory to write captured repro files.
        label: Human-readable label for the capture session.
    """
    global _active_state
    _active_state = _CaptureState(output_dir, label)

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
