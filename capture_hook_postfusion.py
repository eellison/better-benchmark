"""
Post-fusion capture hook.

Unlike capture_hook.py (which fires pre-scheduling and guesses fusion boundaries),
this hook fires AFTER Inductor's scheduler decides kernel boundaries. It captures
the exact subgraph for each fused kernel, with real strides/layouts.

This is especially valuable for channels-last models (timm) where stride patterns
affect kernel performance.

Usage:
    from capture_hook_postfusion import install_postfusion_hook, uninstall_postfusion_hook
    install_postfusion_hook("/tmp/captures/my_model", label="my_model")

    compiled = torch.compile(model)
    compiled(inputs)

    uninstall_postfusion_hook()
"""
import atexit
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


class _PostFusionCaptureState:
    def __init__(self, output_dir: str, label: str = "capture"):
        self.output_dir = output_dir
        self.label = label
        self.seen_hashes: set[str] = set()
        self.counter = 0
        self.captured: list[dict] = []
        os.makedirs(output_dir, exist_ok=True)

    def process_fused_node(self, fused_node, graph_module):
        """Called for each FusedSchedulerNode during codegen."""
        from torch._inductor.scheduler import FusedSchedulerNode, SchedulerNode

        # Collect all origin FX nodes for this fused kernel
        origin_fx_nodes = []
        for snode in fused_node.get_nodes():
            if not isinstance(snode, SchedulerNode):
                continue
            if snode.node is None:
                continue
            for origin in snode.node.get_origins():
                if origin.op == "call_function" and origin not in origin_fx_nodes:
                    origin_fx_nodes.append(origin)

        if not origin_fx_nodes:
            return

        # Compute pattern hash from ops (same as capture_hook.py for dedup compatibility)
        origin_ops = sorted(str(n.target) for n in origin_fx_nodes)
        pattern_key = hashlib.md5(
            json.dumps({"ops": origin_ops}, sort_keys=True).encode()
        ).hexdigest()[:12]

        # Get input/output stride info from the IR nodes
        input_layouts = {}
        for snode in fused_node.get_nodes():
            if not isinstance(snode, SchedulerNode) or snode.node is None:
                continue
            node = snode.node
            if hasattr(node, 'layout') and hasattr(node.layout, 'stride'):
                try:
                    size = [int(s) for s in node.layout.size]
                    stride = [int(s) for s in node.layout.stride]
                    input_layouts[snode.get_name()] = {
                        "size": size,
                        "stride": stride,
                        "dtype": str(node.layout.dtype),
                    }
                except (TypeError, RuntimeError):
                    pass

        # Shape hash includes strides (so channels-last is distinct)
        shape_data = []
        for origin in origin_fx_nodes:
            val = origin.meta.get("val")
            if isinstance(val, torch.Tensor):
                shape_data.append(f"{list(val.shape)}:{list(val.stride())}:{val.dtype}")
        shape_key = hashlib.md5(
            json.dumps(shape_data).encode()
        ).hexdigest()[:8]

        full_key = f"{pattern_key}_{shape_key}"
        if full_key in self.seen_hashes:
            return
        self.seen_hashes.add(full_key)

        # Extract subgraph from the origin FX nodes
        result = self._extract_subgraph(origin_fx_nodes, graph_module)
        if result is None:
            return
        sub_gm, placeholder_info, shape_params = result

        # Determine kernel characteristics
        has_reduction = any(
            hasattr(n.target, 'tags') and torch.Tag.reduction in n.target.tags
            for n in origin_fx_nodes
            if n.op == "call_function" and isinstance(n.target, torch._ops.OpOverload)
        )
        kind = "reduction" if has_reduction else "pointwise"
        reduction_types = []
        if has_reduction:
            for n in origin_fx_nodes:
                if (n.op == "call_function" and isinstance(n.target, torch._ops.OpOverload)
                        and torch.Tag.reduction in n.target.tags):
                    reduction_types.append(n.target.overloadpacket.__name__)

        kind_label = "_".join(sorted(set(reduction_types[:3]))) if reduction_types else kind

        meta = {
            "kind": kind,
            "pattern_hash": pattern_key,
            "shape_hash": shape_key,
            "hash": full_key,
            "n_ops": len(origin_fx_nodes),
            "origin_ops": origin_ops,
            "reduction_types": sorted(set(reduction_types)),
            "input_layouts": input_layouts,
            "is_channels_last": any(
                self._is_channels_last(info) for info in input_layouts.values()
            ),
        }

        filename = f"region_{self.counter:03d}_{kind_label}_{full_key}.py"
        self.counter += 1

        filepath = self._generate_repro_file(sub_gm, placeholder_info, meta, filename, shape_params)

        entry = {
            "file": filepath,
            "pattern_hash": pattern_key,
            "shape_hash": shape_key,
            "kind": kind,
            "reduction_types": sorted(set(reduction_types)),
            "n_ops": len(origin_fx_nodes),
            "origin_ops": origin_ops,
            "is_channels_last": meta["is_channels_last"],
        }
        self.captured.append(entry)

    def _is_channels_last(self, layout_info: dict) -> bool:
        size = layout_info.get("size", [])
        stride = layout_info.get("stride", [])
        if len(size) != 4 or len(stride) != 4:
            return False
        # NHWC: stride order is [HWC, C*H*W doesn't matter, C, 1]
        # channels_last: stride[1] == 1 and stride[-1] != 1
        return stride[1] == 1 and len(size) == 4

    def _extract_subgraph(self, origin_nodes, gm):
        """Same logic as capture_hook.py but reused here."""
        # Import the shared implementation
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from capture_hook import _CaptureState
        dummy = _CaptureState.__new__(_CaptureState)
        dummy.output_dir = self.output_dir
        dummy.label = self.label
        dummy.seen_hashes = set()
        dummy.counter = 0
        dummy.captured = []
        return dummy._extract_subgraph(origin_nodes, gm)

    def _generate_repro_file(self, gm, placeholder_info, meta, filename, shape_params=None):
        """Same logic as capture_hook.py."""
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from capture_hook import _CaptureState
        dummy = _CaptureState.__new__(_CaptureState)
        dummy.output_dir = self.output_dir
        dummy.label = self.label
        return dummy._generate_repro_file(gm, placeholder_info, meta, filename, shape_params)

    def finalize(self):
        if not self.captured:
            return
        index_path = os.path.join(self.output_dir, "index.json")
        with open(index_path, "w") as f:
            json.dump(self.captured, f, indent=2)

        n_reduction = sum(1 for e in self.captured if e["kind"] == "reduction")
        n_pointwise = len(self.captured) - n_reduction
        n_cl = sum(1 for e in self.captured if e.get("is_channels_last"))
        print(f"[capture_hook_postfusion] Captured {len(self.captured)} regions "
              f"({n_reduction} reduction, {n_pointwise} pointwise, {n_cl} channels-last) "
              f"-> {self.output_dir}")


_state: _PostFusionCaptureState | None = None
_original_codegen = None


def install_postfusion_hook(output_dir: str, label: str = "capture"):
    """Install hook that captures fused kernel regions after scheduling."""
    global _state, _original_codegen

    _state = _PostFusionCaptureState(output_dir, label)

    from torch._inductor.scheduler import Scheduler

    _original_codegen = Scheduler._codegen

    def _patched_codegen(self, nodes):
        from torch._inductor.scheduler import FusedSchedulerNode, SchedulerNode
        from torch._inductor.virtualized import V

        # Capture each fused node before codegen
        gm = None
        try:
            gm = V.graph.module
        except Exception:
            pass

        if gm is not None and _state is not None:
            for node in nodes:
                try:
                    if isinstance(node, FusedSchedulerNode):
                        _state.process_fused_node(node, gm)
                    elif isinstance(node, SchedulerNode) and node.node is not None:
                        origins = list(node.node.get_origins())
                        if origins and any(o.op == "call_function" for o in origins):
                            _state.process_fused_node(node, gm)
                except Exception:
                    pass

        return _original_codegen(self, nodes)

    Scheduler._codegen = _patched_codegen

    atexit.register(lambda: _state.finalize() if _state else None)
    print(f"[capture_hook_postfusion] Installed. Captures will be written to {output_dir}")


def uninstall_postfusion_hook():
    global _state, _original_codegen

    if _state is not None:
        _state.finalize()

    if _original_codegen is not None:
        from torch._inductor.scheduler import Scheduler
        Scheduler._codegen = _original_codegen
        _original_codegen = None

    _state = None
