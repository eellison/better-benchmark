#!/usr/bin/env python3
"""Model-level graph accounting via programmatic FX graph traversal.

Loads saved full_graph_*.py artifacts, traces them with make_fx to get concrete
shape metadata on every node, then classifies each op as fusible or non-fusible
and builds a complete accounting of the model's compute structure.

For each model graph, produces:
  - List of non-fusible ops with output shapes and occurrence counts
  - List of fusible partitions (connected components of fusible ops)
  - Mapping of fusible partitions to canonical repro hashes (via manifest)
  - Summary statistics: total ops, partition counts, compute breakdown

Usage:
    # Single model (starts with convnextv2_train as the reference model)
    python scripts/model_graph_accounting.py --model convnextv2

    # All models
    python scripts/model_graph_accounting.py --all

    # JSON output for downstream tooling
    python scripts/model_graph_accounting.py --model convnextv2 --json
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "repros" / "models"
CANONICAL_DIR = ROOT / "repros" / "canonical"

# ============================================================================
# Op classification
# ============================================================================

# Non-fusible ops: operations that correspond to heavy BLAS/cuDNN kernels
# which Inductor does NOT fuse with surrounding pointwise/reduction ops.
NON_FUSIBLE_OP_NAMES = {
    "aten.convolution.default",
    "aten.convolution_backward.default",
    "aten.mm.default",
    "aten.bmm.default",
    "aten.addmm.default",
    "aten._scaled_dot_product_flash_attention.default",
    "aten._scaled_dot_product_efficient_attention.default",
    "aten._scaled_dot_product_flash_attention_backward.default",
    "aten._scaled_dot_product_efficient_attention_backward.default",
    "aten._scaled_dot_product_cudnn_attention.default",
    "aten._scaled_dot_product_cudnn_attention_backward.default",
}

# These are ops that appear in the graph but don't correspond to real compute;
# they are view/metadata ops that get eliminated during codegen.
VIEW_OPS = {
    "aten.permute.default",
    "aten.transpose.int",
    "aten.t.default",
    "aten.view.default",
    "aten.reshape.default",
    "aten.expand.default",
    "aten.slice.Tensor",
    "aten.select.int",
    "aten.unsqueeze.default",
    "aten.squeeze.default",
    "aten.squeeze.dim",
    "aten.contiguous.default",
    "aten.as_strided.default",
    "aten.clone.default",
}


def _target_name(target) -> str:
    """Get a standardized string name for an FX node target."""
    s = str(target)
    # Handle the common aten op format
    if hasattr(target, "_schema"):
        # This is an aten op with a schema
        name = target._schema.name
        overload = target._schema.overload_name
        if overload:
            return f"{name}.{overload}"
        return f"{name}.default"
    return s


def _is_non_fusible(target_name: str) -> bool:
    """Check if an op name corresponds to a non-fusible (BLAS/cuDNN) operation."""
    return target_name in NON_FUSIBLE_OP_NAMES


def _is_view_op(target_name: str) -> bool:
    """Check if an op is a pure view/metadata operation."""
    return target_name in VIEW_OPS


def _classify_fusible_op(target_name: str) -> str:
    """Classify a fusible op into a category."""
    if "var_mean" in target_name or "var." in target_name:
        return "reduction"
    if any(r in target_name for r in [
        "sum.", "sum_", "mean.", "mean_", "amax.", "amin.",
        "norm.", "prod.", "any.", "all.",
    ]):
        return "reduction"
    if "softmax" in target_name:
        return "reduction"
    if target_name in VIEW_OPS:
        return "view"
    if "getitem" in target_name:
        return "getitem"
    # Pointwise ops (includes backward pointwise like mul, add etc.)
    if any(pw in target_name for pw in [
        "mul.", "add.", "sub.", "div.", "pow.", "rsqrt", "sqrt", "exp.",
        "neg.", "abs.", "sigmoid", "tanh", "relu", "gelu", "silu",
        "erf.", "where.", "clamp", "maximum.", "minimum.", "addcmul",
        "ne.", "eq.", "gt.", "lt.", "ge.", "le.", "logical",
        "bitwise", "convert_element_type", "to_dtype",
        "fill.", "zeros_like", "ones_like", "full_like",
        "copy.", "scatter.", "gather.", "index.",
        "native_dropout", "threshold_backward",
        "embedding_dense_backward",
        "nll_loss", "log.", "reciprocal",
    ]):
        return "pointwise"
    # View-like ops that rearrange data
    if any(v in target_name for v in [
        "permute", "transpose", "reshape", "view", "expand",
        "slice", "select", "unsqueeze", "squeeze", "contiguous",
        "as_strided", "clone", "cat.", "stack.",
        "split", "chunk", "narrow",
    ]):
        return "view"
    return "other"


# ============================================================================
# Union-Find for partition grouping
# ============================================================================

class UnionFind:
    """Disjoint set data structure for grouping fusible ops into partitions."""

    def __init__(self):
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: str, y: str) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


# ============================================================================
# Data structures for accounting results
# ============================================================================

@dataclass
class NonFusibleOp:
    """A single non-fusible op instance in the graph."""
    target: str
    output_shape: list[int] | None
    output_dtype: str | None
    node_name: str
    # For convolutions: weight shape, stride, groups, etc.
    conv_info: dict[str, Any] | None = None

    def flop_estimate(self) -> int | None:
        """Rough FLOP estimate for the op."""
        if self.conv_info:
            # Conv2d FLOPs ~= 2 * Cout * Cin/groups * Kh * Kw * Hout * Wout * batch
            info = self.conv_info
            wshape = info.get("weight_shape")
            if wshape and len(wshape) == 4:
                cout, cin_per_group, kh, kw = wshape
                groups = info.get("groups", 1)
                # For forward: output shape is [N, Cout, Hout, Wout]
                if self.output_shape and len(self.output_shape) == 4:
                    batch, _, hout, wout = self.output_shape
                    return 2 * batch * cout * cin_per_group * kh * kw * hout * wout
                # For backward: infer from weight shape and input shape
                if "input_shape" in info:
                    ishape = info["input_shape"]
                    if len(ishape) == 4:
                        batch = ishape[0]
                        hout = ishape[2] // max(info.get("stride", [1, 1])[0], 1)
                        wout = ishape[3] // max(info.get("stride", [1, 1])[-1], 1)
                        return 2 * batch * cout * cin_per_group * kh * kw * hout * wout
        if "mm" in self.target and self.output_shape:
            if len(self.output_shape) == 2:
                m, n = self.output_shape
                # k is unknown without input shape, but we can note the output
                return None
        return None


@dataclass
class FusiblePartition:
    """A connected component of fusible ops (forms one Inductor kernel)."""
    partition_id: int
    node_names: list[str]
    op_counts: dict[str, int]  # target_name -> count
    category_counts: dict[str, int]  # "pointwise", "reduction", "view" -> count
    # Representative output shape (largest intermediate)
    max_output_shape: list[int] | None = None
    max_output_numel: int = 0
    # Link to canonical repro (if available via manifest)
    repro_hash: str | None = None
    repro_id: str | None = None

    @property
    def total_ops(self) -> int:
        return sum(self.op_counts.values())

    @property
    def has_reduction(self) -> bool:
        return self.category_counts.get("reduction", 0) > 0

    @property
    def kind(self) -> str:
        """Classify the partition: 'reduction', 'pointwise', or 'mixed'."""
        if self.has_reduction:
            return "reduction"
        if self.category_counts.get("pointwise", 0) > 0:
            return "pointwise"
        return "other"


@dataclass
class GraphAccounting:
    """Complete accounting for one full_graph_*.py file."""
    graph_path: str
    model_name: str
    graph_index: int

    # Node counts
    total_nodes: int = 0
    placeholder_count: int = 0
    output_count: int = 0
    call_function_count: int = 0

    # Op breakdown
    non_fusible_ops: list[NonFusibleOp] = field(default_factory=list)
    fusible_partitions: list[FusiblePartition] = field(default_factory=list)

    # Pattern mapping from manifest
    manifest_patterns: list[str] = field(default_factory=list)

    # Errors during processing
    errors: list[str] = field(default_factory=list)

    @property
    def non_fusible_count(self) -> int:
        return len(self.non_fusible_ops)

    @property
    def fusible_partition_count(self) -> int:
        return len(self.fusible_partitions)

    @property
    def fusible_op_count(self) -> int:
        return sum(p.total_ops for p in self.fusible_partitions)

    def summary(self) -> dict[str, Any]:
        """Compact summary for reporting."""
        nf_breakdown = Counter(op.target for op in self.non_fusible_ops)
        partition_kinds = Counter(p.kind for p in self.fusible_partitions)
        return {
            "graph_path": self.graph_path,
            "model_name": self.model_name,
            "total_nodes": self.total_nodes,
            "non_fusible_ops": self.non_fusible_count,
            "non_fusible_breakdown": dict(nf_breakdown),
            "fusible_partitions": self.fusible_partition_count,
            "fusible_ops": self.fusible_op_count,
            "partition_kinds": dict(partition_kinds),
            "manifest_patterns": len(self.manifest_patterns),
            "errors": self.errors,
        }


@dataclass
class ModelAccounting:
    """Aggregate accounting for all graphs in a model."""
    model_name: str
    model_dir: str
    graphs: list[GraphAccounting] = field(default_factory=list)

    @property
    def total_non_fusible(self) -> int:
        return sum(g.non_fusible_count for g in self.graphs)

    @property
    def total_fusible_partitions(self) -> int:
        return sum(g.fusible_partition_count for g in self.graphs)

    @property
    def total_fusible_ops(self) -> int:
        return sum(g.fusible_op_count for g in self.graphs)

    def summary(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_dir": self.model_dir,
            "num_graphs": len(self.graphs),
            "total_non_fusible_ops": self.total_non_fusible,
            "total_fusible_partitions": self.total_fusible_partitions,
            "total_fusible_ops": self.total_fusible_ops,
            "graphs": [g.summary() for g in self.graphs],
        }


# ============================================================================
# Core graph analysis
# ============================================================================

def _get_non_fusible_targets():
    """Build the set of non-fusible torch op targets at runtime."""
    import torch

    targets = set()
    for name in NON_FUSIBLE_OP_NAMES:
        # Parse "aten.convolution.default" -> torch.ops.aten.convolution.default
        parts = name.split(".")
        if len(parts) >= 3 and parts[0] == "aten":
            op_name = parts[1]
            overload = parts[2] if len(parts) > 2 else "default"
            base = getattr(torch.ops.aten, op_name, None)
            if base is not None:
                overload_fn = getattr(base, overload, None)
                if overload_fn is not None:
                    targets.add(overload_fn)
    return targets


def _extract_conv_info(node) -> dict[str, Any] | None:
    """Extract convolution parameters from node arguments."""
    import torch

    target_name = _target_name(node.target)
    if "convolution" not in target_name:
        return None

    args = node.args
    if len(args) < 3:
        return None

    info = {}

    if "backward" in target_name:
        # aten.convolution_backward.default(grad_output, input, weight, bias_sizes,
        #   stride, padding, dilation, transposed, output_padding, groups, output_mask)
        if len(args) >= 3:
            grad_output_node = args[0]
            input_node = args[1]
            weight_node = args[2]

            grad_val = grad_output_node.meta.get("val") if hasattr(grad_output_node, "meta") else None
            if grad_val is not None and torch.is_tensor(grad_val):
                info["grad_output_shape"] = [int(d) for d in grad_val.shape]

            input_val = input_node.meta.get("val") if hasattr(input_node, "meta") else None
            if input_val is not None and torch.is_tensor(input_val):
                info["input_shape"] = [int(d) for d in input_val.shape]

            weight_val = weight_node.meta.get("val") if hasattr(weight_node, "meta") else None
            if weight_val is not None and torch.is_tensor(weight_val):
                wshape = [int(d) for d in weight_val.shape]
                info["weight_shape"] = wshape
                if len(wshape) == 4:
                    info["out_channels"] = wshape[0]
                    info["in_channels"] = wshape[1]
                    info["kernel_h"] = wshape[2]
                    info["kernel_w"] = wshape[3]

        if len(args) > 4:
            stride = args[4]
            if isinstance(stride, (list, tuple)):
                info["stride"] = list(stride)
        if len(args) > 5:
            padding = args[5]
            if isinstance(padding, (list, tuple)):
                info["padding"] = list(padding)
        if len(args) > 9:
            groups = args[9]
            if isinstance(groups, int):
                info["groups"] = groups
    else:
        # aten.convolution.default(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups)
        input_node = args[0]
        weight_node = args[1]

        input_val = input_node.meta.get("val") if hasattr(input_node, "meta") else None
        if input_val is not None and torch.is_tensor(input_val):
            info["input_shape"] = [int(d) for d in input_val.shape]

        weight_val = weight_node.meta.get("val") if hasattr(weight_node, "meta") else None
        if weight_val is not None and torch.is_tensor(weight_val):
            wshape = [int(d) for d in weight_val.shape]
            info["weight_shape"] = wshape
            if len(wshape) == 4:
                info["out_channels"] = wshape[0]
                info["in_channels"] = wshape[1]
                info["kernel_h"] = wshape[2]
                info["kernel_w"] = wshape[3]
            elif len(wshape) == 3:
                info["out_channels"] = wshape[0]
                info["in_channels"] = wshape[1]
                info["kernel_size"] = wshape[2]

        if len(args) > 3:
            stride = args[3]
            if isinstance(stride, (list, tuple)):
                info["stride"] = list(stride)
        if len(args) > 4:
            padding = args[4]
            if isinstance(padding, (list, tuple)):
                info["padding"] = list(padding)
        if len(args) > 8:
            groups = args[8]
            if isinstance(groups, int):
                info["groups"] = groups

    return info if info else None


def analyze_graph(gm, graph_path: str, model_name: str, graph_index: int) -> GraphAccounting:
    """Analyze a traced FX GraphModule and produce complete accounting."""
    import torch

    non_fusible_targets = _get_non_fusible_targets()

    accounting = GraphAccounting(
        graph_path=graph_path,
        model_name=model_name,
        graph_index=graph_index,
    )

    # First pass: count node types and classify ops
    node_info: dict[str, dict[str, Any]] = {}  # node_name -> info

    for node in gm.graph.nodes:
        accounting.total_nodes += 1
        if node.op == "placeholder":
            accounting.placeholder_count += 1
        elif node.op == "output":
            accounting.output_count += 1
        elif node.op == "call_function":
            accounting.call_function_count += 1
            target_name = _target_name(node.target)

            val = node.meta.get("val")
            output_shape = None
            output_dtype = None
            output_numel = 0

            def _safe_int(d):
                """Convert dimension to int, handling SymInt."""
                try:
                    if hasattr(d, "node") and hasattr(d.node, "hint"):
                        return int(d.node.hint)
                    return int(d)
                except (TypeError, ValueError):
                    return 32  # Default for unresolvable symbolic dims

            if val is not None and torch.is_tensor(val):
                output_shape = [_safe_int(d) for d in val.shape]
                output_dtype = str(val.dtype)
                output_numel = 1
                for d in output_shape:
                    output_numel *= d
            elif val is not None and isinstance(val, (list, tuple)):
                # Multi-output ops (e.g., convolution_backward returns tuple)
                # Use the first tensor output as representative shape
                for item in val:
                    if torch.is_tensor(item):
                        output_shape = [_safe_int(d) for d in item.shape]
                        output_dtype = str(item.dtype)
                        output_numel = 1
                        for d in output_shape:
                            output_numel *= d
                        break

            is_nf = node.target in non_fusible_targets
            node_info[node.name] = {
                "target_name": target_name,
                "is_non_fusible": is_nf,
                "output_shape": output_shape,
                "output_dtype": output_dtype,
                "output_numel": output_numel,
                "node": node,
            }

            if is_nf:
                conv_info = _extract_conv_info(node)
                accounting.non_fusible_ops.append(NonFusibleOp(
                    target=target_name,
                    output_shape=output_shape,
                    output_dtype=output_dtype,
                    node_name=node.name,
                    conv_info=conv_info,
                ))

    # Second pass: build fusible partitions using union-find
    uf = UnionFind()
    fusible_node_names = set()

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        info = node_info.get(node.name)
        if info is None or info["is_non_fusible"]:
            continue

        fusible_node_names.add(node.name)

        # Connect to fusible input nodes
        for inp in node.all_input_nodes:
            if inp.op != "call_function":
                continue
            inp_info = node_info.get(inp.name)
            if inp_info is None or inp_info["is_non_fusible"]:
                continue
            uf.union(node.name, inp.name)

    # Group nodes by partition root
    partition_groups: dict[str, list[str]] = defaultdict(list)
    for name in fusible_node_names:
        root = uf.find(name)
        partition_groups[root].append(name)

    # Build FusiblePartition objects
    for pid, (root, members) in enumerate(sorted(
        partition_groups.items(),
        key=lambda x: -len(x[1]),
    )):
        op_counts: dict[str, int] = Counter()
        category_counts: dict[str, int] = Counter()
        max_numel = 0
        max_shape = None

        for name in members:
            info = node_info[name]
            target_name = info["target_name"]
            op_counts[target_name] += 1
            category = _classify_fusible_op(target_name)
            category_counts[category] += 1

            if info["output_numel"] > max_numel:
                max_numel = info["output_numel"]
                max_shape = info["output_shape"]

        partition = FusiblePartition(
            partition_id=pid,
            node_names=members,
            op_counts=dict(op_counts),
            category_counts=dict(category_counts),
            max_output_shape=max_shape,
            max_output_numel=max_numel,
        )
        accounting.fusible_partitions.append(partition)

    return accounting


# ============================================================================
# Graph loading and tracing
# ============================================================================

def trace_full_graph(graph_path: Path, device: str = "cpu") -> Any:
    """Load a full_graph_*.py file, construct inputs, and trace with make_fx.

    Returns the traced GraphModule with concrete shape metadata on all nodes.
    Falls back to 'fake' tracing mode if 'real' fails (e.g., device mismatches).
    """
    import torch
    from torch.fx.experimental.proxy_tensor import make_fx

    sys.path.insert(0, str(ROOT))
    from full_graph_harness import load_full_graph

    instance, inputs, definition = load_full_graph(graph_path, default_device=device)

    # Try real mode first (gives concrete shapes)
    try:
        with torch.no_grad():
            gm = make_fx(instance, tracing_mode="real")(*inputs)
        return gm
    except Exception:
        pass

    # If real mode fails (e.g. device mismatch with prims.iota), try with cuda
    if device == "cpu" and torch.cuda.is_available():
        try:
            instance_cuda, inputs_cuda, _ = load_full_graph(graph_path, default_device="cuda")
            with torch.no_grad():
                gm = make_fx(instance_cuda, tracing_mode="real")(*inputs_cuda)
            return gm
        except Exception:
            pass

    # Final fallback: symbolic mode (gives symbolic shapes but still useful)
    with torch.no_grad():
        gm = make_fx(instance, tracing_mode="symbolic")(*inputs)
    return gm


def process_model_graph(
    graph_path: Path,
    model_name: str,
    graph_index: int,
    device: str = "cpu",
) -> GraphAccounting:
    """Process a single full_graph_*.py file end-to-end."""
    try:
        gm = trace_full_graph(graph_path, device=device)
        accounting = analyze_graph(gm, str(graph_path), model_name, graph_index)
    except Exception as e:
        accounting = GraphAccounting(
            graph_path=str(graph_path),
            model_name=model_name,
            graph_index=graph_index,
        )
        accounting.errors.append(f"{type(e).__name__}: {e}")
        traceback.print_exc(file=sys.stderr)

    # Load manifest to get pattern hashes
    manifest_path = graph_path.parent / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            accounting.manifest_patterns = manifest.get("patterns", [])
        except (json.JSONDecodeError, OSError):
            pass

    return accounting


def process_model_directory(model_dir: Path, device: str = "cpu") -> ModelAccounting:
    """Process all full_graph_*.py files in a model directory."""
    model_name = model_dir.name
    accounting = ModelAccounting(model_name=model_name, model_dir=str(model_dir))

    graph_files = sorted(model_dir.glob("full_graph_*.py"))
    if not graph_files:
        return accounting

    for idx, gf in enumerate(graph_files):
        print(f"  Processing {gf.name}...", file=sys.stderr)
        ga = process_model_graph(gf, model_name, idx, device=device)
        accounting.graphs.append(ga)

    return accounting


# ============================================================================
# Discovery
# ============================================================================

def discover_model_dirs() -> list[Path]:
    """Find all model directories that contain full_graph_*.py files."""
    if not MODELS_DIR.exists():
        return []

    dirs = set()
    for fg in MODELS_DIR.rglob("full_graph_*.py"):
        dirs.add(fg.parent)
    return sorted(dirs)


def build_hash_to_repro() -> dict[str, str]:
    """Map pattern hash -> canonical repro_id."""
    mapping = {}
    if not CANONICAL_DIR.exists():
        return mapping
    for d in CANONICAL_DIR.iterdir():
        if d.is_dir():
            parts = d.name.rsplit("_", 1)
            if len(parts) == 2 and len(parts[1]) == 12:
                mapping[parts[1]] = d.name
    return mapping


# ============================================================================
# Reporting
# ============================================================================

def format_text_report(accounting: ModelAccounting, hash_to_repro: dict[str, str]) -> str:
    """Format a human-readable text report for a model."""
    lines = []
    lines.append(f"{'='*70}")
    lines.append(f"MODEL: {accounting.model_name}")
    lines.append(f"DIR:   {accounting.model_dir}")
    lines.append(f"{'='*70}")
    lines.append("")
    lines.append(f"Graphs: {len(accounting.graphs)}")
    lines.append(f"Total non-fusible ops: {accounting.total_non_fusible}")
    lines.append(f"Total fusible partitions: {accounting.total_fusible_partitions}")
    lines.append(f"Total fusible ops: {accounting.total_fusible_ops}")
    lines.append("")

    for ga in accounting.graphs:
        lines.append(f"  --- Graph {ga.graph_index}: {Path(ga.graph_path).name} ---")
        lines.append(f"  Total nodes: {ga.total_nodes}")
        lines.append(f"  Placeholders (inputs): {ga.placeholder_count}")
        lines.append(f"  call_function ops: {ga.call_function_count}")
        lines.append(f"  Non-fusible ops: {ga.non_fusible_count}")
        lines.append(f"  Fusible partitions: {ga.fusible_partition_count}")
        lines.append(f"  Fusible ops: {ga.fusible_op_count}")

        if ga.errors:
            lines.append(f"  ERRORS: {ga.errors}")
            lines.append("")
            continue

        # Non-fusible breakdown
        nf_counter = Counter(op.target for op in ga.non_fusible_ops)
        lines.append("")
        lines.append("  Non-fusible ops breakdown:")
        for op_name, count in nf_counter.most_common():
            lines.append(f"    {op_name}: {count}")

        # Non-fusible ops with shapes (unique shapes)
        shape_groups: dict[str, list[NonFusibleOp]] = defaultdict(list)
        for op in ga.non_fusible_ops:
            key = f"{op.target}|{op.output_shape}"
            shape_groups[key].append(op)

        lines.append("")
        lines.append("  Non-fusible ops by shape:")
        for key, ops in sorted(shape_groups.items(), key=lambda x: -len(x[1])):
            op = ops[0]
            flops = op.flop_estimate()
            flops_str = f" (~{flops/1e9:.2f} GFLOP)" if flops else ""
            extra = ""
            if op.conv_info:
                ci = op.conv_info
                if "groups" in ci and ci["groups"] > 1:
                    extra = f" groups={ci['groups']}"
                if "stride" in ci:
                    extra += f" stride={ci['stride']}"
            lines.append(
                f"    {op.target} -> {op.output_shape} x{len(ops)}{flops_str}{extra}"
            )

        # Fusible partitions
        lines.append("")
        lines.append("  Fusible partitions (sorted by size):")
        for p in ga.fusible_partitions[:15]:  # Show top 15
            kind = p.kind
            top_ops = Counter(p.op_counts).most_common(3)
            top_ops_str = ", ".join(f"{op.split('.')[-1]}:{c}" for op, c in top_ops)
            shape_str = f"max_shape={p.max_output_shape}" if p.max_output_shape else ""
            lines.append(
                f"    P{p.partition_id:02d} [{kind:10s}] "
                f"{p.total_ops:3d} ops | {top_ops_str} | {shape_str}"
            )
        if len(ga.fusible_partitions) > 15:
            lines.append(f"    ... and {len(ga.fusible_partitions) - 15} more partitions")

        # Manifest pattern mapping
        if ga.manifest_patterns:
            lines.append("")
            lines.append(f"  Manifest patterns ({len(ga.manifest_patterns)} fusible partitions from capture):")
            for phash in ga.manifest_patterns:
                repro_id = hash_to_repro.get(phash, f"UNKNOWN_{phash}")
                lines.append(f"    {phash} -> {repro_id}")

        lines.append("")

    return "\n".join(lines)


def format_json_report(accounting: ModelAccounting, hash_to_repro: dict[str, str]) -> dict:
    """Format a JSON-serializable report."""
    result = accounting.summary()

    # Enrich with repro mappings
    for g in result["graphs"]:
        patterns = g.get("manifest_patterns", 0)
        # The full graph accounting has it as a count; let's add the actual mapping
        pass

    # Add detailed per-graph info
    detailed_graphs = []
    for ga in accounting.graphs:
        graph_detail = {
            "graph_path": ga.graph_path,
            "graph_index": ga.graph_index,
            "total_nodes": ga.total_nodes,
            "placeholder_count": ga.placeholder_count,
            "call_function_count": ga.call_function_count,
            "non_fusible_count": ga.non_fusible_count,
            "fusible_partition_count": ga.fusible_partition_count,
            "fusible_op_count": ga.fusible_op_count,
            "errors": ga.errors,
            "non_fusible_ops": [
                {
                    "target": op.target,
                    "output_shape": op.output_shape,
                    "output_dtype": op.output_dtype,
                    "conv_info": op.conv_info,
                    "flop_estimate": op.flop_estimate(),
                }
                for op in ga.non_fusible_ops
            ],
            "fusible_partitions": [
                {
                    "partition_id": p.partition_id,
                    "total_ops": p.total_ops,
                    "kind": p.kind,
                    "op_counts": p.op_counts,
                    "category_counts": p.category_counts,
                    "max_output_shape": p.max_output_shape,
                    "max_output_numel": p.max_output_numel,
                    "repro_hash": p.repro_hash,
                    "repro_id": p.repro_id,
                }
                for p in ga.fusible_partitions
            ],
            "manifest_patterns": [
                {
                    "hash": phash,
                    "repro_id": hash_to_repro.get(phash, None),
                }
                for phash in ga.manifest_patterns
            ],
        }
        detailed_graphs.append(graph_detail)

    result["detailed_graphs"] = detailed_graphs
    return result


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Model-level graph accounting via FX graph traversal"
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Model name filter (partial match). E.g. 'convnextv2' or 'hf_Bert'",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Process all models (can be slow)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of text report",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for tracing (cpu or cuda). CPU is sufficient for shape analysis.",
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=0,
        help="Max graphs to process per model (0 = all)",
    )
    args = parser.parse_args()

    if not args.model and not args.all:
        parser.error("Specify --model <name> or --all")

    # Discover model directories
    model_dirs = discover_model_dirs()
    print(f"Found {len(model_dirs)} model directories with full graphs", file=sys.stderr)

    # Filter
    if args.model:
        model_dirs = [
            d for d in model_dirs
            if args.model.lower() in d.name.lower()
        ]
        if not model_dirs:
            print(f"No models matching '{args.model}'", file=sys.stderr)
            print("Available models:", file=sys.stderr)
            all_dirs = discover_model_dirs()
            for d in all_dirs[:20]:
                print(f"  {d.name}", file=sys.stderr)
            sys.exit(1)

    print(f"Processing {len(model_dirs)} model(s)...", file=sys.stderr)

    # Load hash-to-repro mapping
    hash_to_repro = build_hash_to_repro()
    print(f"Loaded {len(hash_to_repro)} canonical repro mappings", file=sys.stderr)

    # Process each model
    all_results = []
    for model_dir in model_dirs:
        print(f"\nProcessing: {model_dir.name}", file=sys.stderr)
        accounting = process_model_directory(model_dir, device=args.device)

        if args.max_graphs > 0:
            accounting.graphs = accounting.graphs[:args.max_graphs]

        all_results.append(accounting)

    # Output
    if args.json:
        output_data = [format_json_report(a, hash_to_repro) for a in all_results]
        if len(output_data) == 1:
            output_data = output_data[0]
        output_text = json.dumps(output_data, indent=2)
    else:
        reports = [format_text_report(a, hash_to_repro) for a in all_results]
        output_text = "\n\n".join(reports)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output_text + "\n")
        print(f"\nOutput written to {args.output}", file=sys.stderr)
    else:
        print(output_text)

    # Print summary
    print(f"\n{'='*70}", file=sys.stderr)
    print("SUMMARY", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    for a in all_results:
        err_count = sum(len(g.errors) for g in a.graphs)
        err_str = f" ({err_count} errors)" if err_count else ""
        print(
            f"  {a.model_name}: "
            f"{len(a.graphs)} graphs, "
            f"{a.total_non_fusible} non-fusible ops, "
            f"{a.total_fusible_partitions} fusible partitions, "
            f"{a.total_fusible_ops} fusible ops"
            f"{err_str}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
