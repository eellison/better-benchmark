"""Small MM benchmark regions, grown locally from real FX contractions.

Regions may overlap.  Each has an explicit benchmark output set; the original
model graph is never rewritten.  No model names or operator-sequence registry
participate in selection.
"""

from __future__ import annotations

import itertools
import math
import operator
from collections import Counter, deque
from dataclasses import dataclass

import torch
from torch.fx import Node
from torch.utils._pytree import tree_leaves

MM_NAMES = frozenset(
    {
        "aten::mm",
        "aten::bmm",
        "aten::matmul",
        "aten::addmm",
        "aten::baddbmm",
        "aten::mv",
        "aten::addmv",
        "aten::_scaled_mm",
        "aten::_int_mm",
    }
)
INDEX_NAMES = frozenset(
    {
        "aten::index",
        "aten::index_select",
        "aten::gather",
        "aten::embedding",
        "aten::index_put",
        "aten::index_put_",
        "aten::index_copy",
        "aten::index_copy_",
        "aten::scatter",
        "aten::scatter_",
        "aten::scatter_add",
        "aten::scatter_add_",
        "aten::scatter_reduce",
        "aten::scatter_reduce_",
    }
)
VIEW_NAMES = frozenset(
    {
        "aten::view",
        "aten::reshape",
        "aten::_unsafe_view",
        "aten::permute",
        "aten::t",
        "aten::transpose",
        "aten::expand",
        "aten::slice",
        "aten::select",
        "aten::squeeze",
        "aten::unsqueeze",
        "aten::as_strided",
        "aten::split",
        "aten::split_with_sizes",
        "aten::unbind",
        "aten::alias",
        "aten::detach",
    }
)
SCAFFOLD_NAMES = VIEW_NAMES | frozenset(
    {
        "aten::_to_copy",
        "prims::convert_element_type",
        "aten::clone",
        "aten::contiguous",
        "aten::copy",
        "aten::copy_",
        "aten::empty",
        "aten::empty_strided",
        "aten::empty_like",
        "aten::zeros",
        "aten::zeros_like",
        "aten::ones",
        "aten::ones_like",
        "aten::full",
        "aten::full_like",
        "aten::lift_fresh_copy",
        "aten::size",
        "aten::sym_size",
        "aten::stride",
        "aten::sym_stride",
        "aten::numel",
        "aten::sym_numel",
    }
)
SHAPE_ONLY_NAMES = frozenset(
    {
        "aten::empty_like",
        "aten::zeros_like",
        "aten::ones_like",
        "aten::full_like",
        "aten::rand_like",
        "aten::randn_like",
        "aten::size",
        "aten::sym_size",
        "aten::stride",
        "aten::sym_stride",
        "aten::numel",
        "aten::sym_numel",
    }
)


@dataclass(frozen=True)
class MatmulLimits:
    """Benchmark size caps plus three contraction budgets.

    Internal contraction axes and live features share max_internal_dim.
    A feature-collapsing reduction may start from a larger MM K, bounded by
    max_reduction_extent; that also bounds cumulative ordinary reduction.
    MM replaces its contracted axis, so its K does not multiply coverage.
    """

    max_mm: int = 4
    max_nodes: int = 96
    max_compute: int = 48
    max_outputs: int = 4
    max_inputs: int = 24
    max_contraction_depth: int = 4
    max_internal_dim: int = 128
    max_reduction_extent: int = 512


class MatmulRegion(list):
    """FX nodes plus the meaningful outputs of this independent benchmark."""

    def __init__(self, nodes, outputs, metrics):
        super().__init__(nodes)
        self.outputs = tuple(outputs)
        self.metrics = metrics
        self.mm_types = tuple(
            sorted({op_name(n).split("::")[-1] for n in nodes if is_matmul(n)})
        )


def op_name(node):
    return getattr(getattr(node.target, "_schema", None), "name", "")


def is_matmul(node):
    return node.op == "call_function" and op_name(node) in MM_NAMES


def tensor_shape(node):
    value = node.meta.get("val")
    if isinstance(value, (tuple, list)):
        value = next((x for x in value if isinstance(x, torch.Tensor)), None)
    if isinstance(value, torch.Tensor):
        dims = value.shape
    else:
        meta = node.meta.get("tensor_meta")
        if meta is None or not hasattr(meta, "shape"):
            return None
        dims = meta.shape
    result = []
    for dim in dims:
        if isinstance(dim, int):
            result.append(dim)
        elif isinstance(dim, torch.SymInt) and dim.node.has_hint():
            result.append(dim.node.hint)
        else:
            return None
    return tuple(result)


def value_inputs(node):
    name = op_name(node)
    if name in SHAPE_ONLY_NAMES:
        reference = node.args[0] if node.args else node.kwargs.get("self")
        return [n for n in node.all_input_nodes if n is not reference]
    if name == "aten::copy_":
        return [n for n in node.all_input_nodes if n is not node.args[0]]
    return node.all_input_nodes


def is_mutating(node):
    schema = getattr(node.target, "_schema", None)
    return bool(schema and schema.is_mutable)


def _neutral(node):
    if len(node.args) < 2:
        return False
    a, b = node.args[:2]
    scalar = lambda x, v: type(x) in (int, float) and x == v
    name = op_name(node)
    if name in {"aten::mul", "prims::mul"}:
        return scalar(a, 1) or scalar(b, 1)
    if name in {"aten::add", "prims::add"}:
        return node.kwargs.get("alpha", 1) == 1 and (scalar(a, 0) or scalar(b, 0))
    if name in {"aten::sub", "prims::sub"}:
        return scalar(b, 0) and node.kwargs.get("alpha", 1) == 1
    if name in {"aten::div", "prims::div", "aten::pow", "prims::pow"}:
        return scalar(b, 1) and node.kwargs.get("rounding_mode") is None
    return False


def is_computation(node):
    """Casts, views, copies and allocations never qualify an MM benchmark."""
    return (
        node.op == "call_function"
        and not is_matmul(node)
        and node.target is not operator.getitem
        and op_name(node) not in SCAFFOLD_NAMES
        and tensor_shape(node) is not None
        and not _neutral(node)
    )


def _operands(node):
    start = 1 if op_name(node) in {"aten::addmm", "aten::baddbmm", "aten::addmv"} else 0
    return node.args[start : start + 2]


class _Collector:
    def __init__(self, gm, supported, limits):
        self.nodes = tuple(gm.graph.nodes)
        self.order = {n: i for i, n in enumerate(self.nodes)}
        self.supported = supported
        self.limits = limits
        self.shapes = {n: tensor_shape(n) for n in self.nodes}
        self.mm_dependent = {}
        self.aliases = {}
        self.rejections = Counter()
        for node in self.nodes:
            self.mm_dependent[node] = node in supported and (
                is_matmul(node) or any(self.mm_dependent[a] for a in value_inputs(node))
            )
            keys = {("node", node)}
            value = node.meta.get("val")
            if isinstance(value, torch.Tensor) and value.layout == torch.strided:
                keys.add(("storage", value.untyped_storage()._cdata))
            if (
                op_name(node) in VIEW_NAMES
                and node.args
                and isinstance(node.args[0], Node)
            ):
                keys.update(self.aliases[node.args[0]])
            if is_mutating(node) and node.args and isinstance(node.args[0], Node):
                keys.update(self.aliases[node.args[0]])
            self.aliases[node] = keys
        self.reads = {
            n: set().union(*(self.aliases[a] for a in n.all_input_nodes))
            for n in self.nodes
        }
        self.writes = {}
        for node in self.nodes:
            schema = getattr(node.target, "_schema", None)
            keys = set()
            if schema is not None:
                for i, arg in enumerate(schema.arguments):
                    if arg.alias_info is not None and arg.alias_info.is_write:
                        value = (
                            node.args[i]
                            if i < len(node.args)
                            else node.kwargs.get(arg.name)
                        )
                        for item in tree_leaves(value):
                            if isinstance(item, Node):
                                keys.update(self.aliases[item])
            self.writes[node] = keys

    def cone(self, roots, stop_at_mm=True, only_mm_paths=False):
        result, pending = set(), list(roots)
        while pending:
            node = pending.pop()
            if node in result or node not in self.supported:
                continue
            if only_mm_paths and not self.mm_dependent[node]:
                continue
            result.add(node)
            if len(result) > self.limits.max_nodes:
                return None
            if is_matmul(node) and stop_at_mm:
                continue
            pending.extend(value_inputs(node))
        return result

    def measure(self, core):
        ordered = sorted(core, key=self.order.__getitem__)
        dots = [n for n in ordered if is_matmul(n)]
        l = self.limits
        if len(core) > l.max_nodes or len(dots) > l.max_mm:
            return None, "size"
        ancestors, depth, red_depth, combined, coverage = {}, {}, {}, {}, {}
        linked, links, geometry = set(), [], {}
        extent, handoff = 1, 0
        for node in ordered:
            inputs = [a for a in value_inputs(node) if a in core]
            upstream = set().union(*(ancestors[a] for a in inputs))
            reduction = not is_matmul(node) and torch.Tag.reduction in getattr(
                node.target, "tags", ()
            )
            depth[node] = max((depth[a] for a in inputs), default=0) + int(
                is_matmul(node)
            )
            red_depth[node] = max((red_depth[a] for a in inputs), default=0) + int(
                reduction
            )
            combined[node] = max((combined[a] for a in inputs), default=0) + int(
                is_matmul(node) or reduction
            )
            shape = self.shapes[node]
            size = math.prod(shape) if shape is not None else None
            full = [
                a
                for a in value_inputs(node)
                if self.shapes[a] is not None and math.prod(self.shapes[a]) == size
            ]
            represented = min(
                (coverage.get(a, 1) for a in full),
                default=max((coverage[a] for a in inputs), default=1),
            )
            ancestors[node] = upstream | ({node} if is_matmul(node) else set())
            if is_matmul(node):
                a, b = _operands(node)
                if not isinstance(a, Node) or not isinstance(b, Node):
                    return None, "metadata"
                sa, sb = self.shapes[a], self.shapes[b]
                if (
                    not sa
                    or not sb
                    or shape is None
                    or not all(d > 0 for d in (*sa, *sb))
                ):
                    return None, "metadata"
                geometry[node] = (
                    sa[-1],
                    sb[-1] if len(sb) >= 2 else 1,
                    len(sb) == 1 or sb[-1] == 1,
                )
                values = [a.meta.get("val"), b.meta.get("val"), node.meta.get("val")]
                devices = {v.device for v in values if isinstance(v, torch.Tensor)}
                if len(devices) > 1 or any(
                    isinstance(v, torch.Tensor) and v.layout != torch.strided
                    for v in values
                ):
                    return None, "device_or_layout"
                represented = 1
                for side, operand in enumerate((a, b)):
                    parents = ancestors.get(operand, set())
                    if not parents:
                        continue
                    s = self.shapes[operand]
                    k = s[-2] if side and len(s) >= 2 else s[-1]
                    handoff = max(handoff, k)
                    linked.update(parents | {node})
                    links.extend((p, node) for p in parents)
            elif reduction:
                source = node.all_input_nodes[0] if node.all_input_nodes else None
                ss = self.shapes.get(source)
                if ss is None or size is None or size == 0:
                    return None, "reduction_metadata"
                bindings = {
                    arg.name: value
                    for arg, value in zip(node.target._schema.arguments, node.args)
                }
                bindings.update(node.kwargs)
                dims = bindings.get("dim", bindings.get("dims"))
                dims = [dims] if isinstance(dims, int) else dims
                dims = range(len(ss)) if dims is None or dims == [] else dims
                if any(ss[d % len(ss)] > l.max_internal_dim for d in dims):
                    return None, "reduction_axis"
                represented = coverage.get(source, 1) * max(1, math.prod(ss) // size)
                extent = max(extent, represented)
                if any(geometry[p][0] > l.max_reduction_extent for p in upstream):
                    return None, "reduction_mm_k"
            coverage[node] = represented
        if handoff > l.max_internal_dim:
            return None, "handoff"
        if linked:
            # A terminal one-value-per-row contraction has the same envelope
            # as sum/max/etc.  This depends on geometry, not an MV op name.
            scalar_handoff = (
                len(linked) == 2 and len(links) == 1 and geometry[links[0][1]][2]
            )
            k_limit = l.max_reduction_extent if scalar_handoff else l.max_internal_dim
            if any(
                geometry[p][0] > k_limit or geometry[p][1] > l.max_internal_dim
                for p in linked
            ):
                return None, "chain_geometry"
        # Independent horizontal MM must have the same surviving row domain.
        for a, b in itertools.combinations(dots, 2):
            if a in ancestors[b] or b in ancestors[a]:
                continue
            domains = []
            for p in (a, b):
                rhs = self.shapes[_operands(p)[1]]
                domains.append(self.shapes[p] if len(rhs) == 1 else self.shapes[p][:-1])
            if domains[0] != domains[1]:
                return None, "parallel_domain"
        compute = sum(is_matmul(n) or is_computation(n) for n in core)
        metrics = {
            "nodes": len(core),
            "compute": compute,
            "mm": len(dots),
            "mm_depth": max(depth.values(), default=0),
            "reduction_depth": max(red_depth.values(), default=0),
            "contraction_depth": max(combined.values(), default=0),
            "reduction_extent": extent,
            "handoff": handoff,
        }
        for key, cap in (
            ("compute", l.max_compute),
            ("contraction_depth", l.max_contraction_depth),
            ("reduction_extent", l.max_reduction_extent),
        ):
            if metrics[key] > cap:
                return None, key
        return metrics, None

    def grow(self, seed):
        core = {seed}
        pending = deque(sorted(seed.users, key=self.order.__getitem__))
        visited = set()
        while pending:
            node = pending.popleft()
            if node in visited or node not in self.supported:
                continue
            visited.add(node)
            branch = self.cone((node,), only_mm_paths=True)
            if is_matmul(node):
                parents = self.cone(value_inputs(node), only_mm_paths=True)
                if parents is None:
                    branch = None
                elif branch is not None:
                    branch |= parents
            if branch is None:
                continue
            trial = core | branch
            _, reason = self.measure(trial)
            if reason:
                self.rejections[reason] += 1
                continue
            added = trial - core
            core = trial
            for n in sorted(added, key=self.order.__getitem__):
                pending.extend(sorted(n.users, key=self.order.__getitem__))
        # Prologues are atomic cones: never collect just the tail of a large norm.
        roots = {a for n in core for a in value_inputs(n) if a not in core}
        for root in sorted(roots, key=self.order.__getitem__):
            # Inter-MM PW belongs to the producing MM's epilogue. Do not
            # collect the same PW again as the consumer MM's prologue.
            # Genuine input prologues have no upstream supported MM.
            if is_matmul(root) or self.mm_dependent[root]:
                continue
            prefix = self.cone((root,))
            if prefix is None:
                continue
            prefix = {n for n in prefix if not is_matmul(n)}
            trial = core | prefix
            _, reason = self.measure(trial)
            if reason is None:
                core = trial
        return core

    def safe(self, core):
        last = max(self.order[n] for n in core)
        pending = [
            u for n in core for u in n.users if u not in core and self.order[u] < last
        ]
        seen = set()
        while pending:
            n = pending.pop()
            if n in seen:
                continue
            seen.add(n)
            if any(u in core for u in n.users):
                return False
            pending.extend(u for u in n.users if self.order[u] < last)
        reads, writes = self.reads, self.writes
        ordered = sorted(core, key=self.order.__getitem__)
        touched = set().union(*(reads[n] | writes[n] for n in core))
        if any(
            writes[n] & touched
            for n in self.nodes[self.order[ordered[0]] : last]
            if n not in core
        ):
            return False
        reach = {}
        for n in ordered:
            reach[n] = set().union(
                *(reach[a] | {a} for a in value_inputs(n) if a in core)
            )
            if any(
                writes[p] & reads[n] and p not in reach[n]
                for p in ordered
                if self.order[p] < self.order[n]
            ):
                return False
        return True

    def finish(self, core, *, allow_trivial=False):
        ancestors, mm_path, math_path = {}, {}, {}
        for n in sorted(core, key=self.order.__getitem__):
            inputs = [a for a in value_inputs(n) if a in core]
            ancestors[n] = set().union(*(ancestors[a] | {a} for a in inputs))
            mm_path[n] = is_matmul(n) or any(mm_path[a] for a in inputs)
            math_path[n] = is_computation(n) or any(math_path[a] for a in inputs)
        mm_count = sum(is_matmul(n) for n in core)
        pure_horizontal = (
            mm_count >= 2
            and not any(is_computation(n) for n in core)
            and not any(
                is_matmul(p) for n in core if is_matmul(n) for p in ancestors[n]
            )
        )
        outputs = [
            n
            for n in core
            if (is_mutating(n) or any(u not in core for u in n.users))
            and mm_path[n]
            and (
                math_path[n]
                or pure_horizontal
                or (allow_trivial and not any(math_path.values()))
            )
        ]
        if not outputs:
            self.rejections["no_meaningful_output"] += 1
            return None
        if len(outputs) > self.limits.max_outputs:
            self.rejections["outputs"] += 1
            return None
        needed = set(outputs).union(*(ancestors[n] for n in outputs))
        if not self.safe(needed):
            self.rejections["cycle_or_alias"] += 1
            return None
        metrics, reason = self.measure(needed)
        if reason:
            self.rejections[reason] += 1
            return None
        metrics["non_mm_compute"] = sum(is_computation(n) for n in needed)
        metrics["outputs"] = len(outputs)
        metrics["inputs"] = len(
            {a for n in needed for a in n.all_input_nodes if a not in needed}
        )
        if metrics["inputs"] > self.limits.max_inputs:
            self.rejections["inputs"] += 1
            return None
        return MatmulRegion(
            sorted(needed, key=self.order.__getitem__),
            sorted(outputs, key=self.order.__getitem__),
            metrics,
        )


def get_matmul_regions(gm, is_supported, *, limits=None, excluded=(), diagnostics=None):
    """Grow at most one bounded candidate per MM, then remove redundant subsets."""
    limits = limits or MatmulLimits()
    excluded = set(excluded)

    def local_copy(node):
        if (
            op_name(node) != "aten::_to_copy"
            or not node.args
            or not isinstance(node.args[0], Node)
        ):
            return True
        source, result = node.args[0].meta.get("val"), node.meta.get("val")
        return not (
            isinstance(source, torch.Tensor)
            and isinstance(result, torch.Tensor)
            and source.device != result.device
        )

    def tensor_operation(node):
        tags = getattr(node.target, "tags", ())
        return op_name(node).startswith(("aten::", "prims::")) and (
            torch.Tag.pointwise in tags or torch.Tag.reduction in tags
        )

    supported = {
        n
        for n in gm.graph.nodes
        if n not in excluded
        and n.op == "call_function"
        and (
            is_matmul(n)
            or op_name(n) in INDEX_NAMES | SCAFFOLD_NAMES
            or n.target is operator.getitem
            or tensor_operation(n)
            or is_supported(n)
        )
        and local_copy(n)
        and not any(x in op_name(n) for x in ("softmax", "attention", "grouped_mm"))
    }
    # Tuple-valued opaque operations are boundaries. Their tensor projections
    # must become tensor inputs, rather than unrecorded tuple placeholders.
    for node in gm.graph.nodes:
        if (
            node in supported
            and node.target is operator.getitem
            and (not isinstance(node.args[0], Node) or node.args[0] not in supported)
        ):
            supported.remove(node)
    collector = _Collector(gm, supported, limits)
    results, seen = [], set()
    for node in collector.nodes:
        if not is_matmul(node) or node not in supported:
            continue
        region = collector.finish(collector.grow(node), allow_trivial=True)
        if region is None:
            continue
        key = frozenset(region), frozenset(region.outputs)
        if key not in seen:
            seen.add(key)
            results.append(region)
    results = [
        r
        for r in results
        if not any(
            set(r) < set(other) and set(r.outputs) <= set(other.outputs)
            for other in results
        )
    ]
    # Independent branches need no join or shared input to be horizontal.
    # Try neighboring candidates only, keeping this pass linear in candidates.
    horizontal = []
    for region in results:
        if horizontal:
            previous = horizontal[-1]
            previous_mm = {n for n in previous if is_matmul(n)}
            current_mm = {n for n in region if is_matmul(n)}
            trial = set(previous) | set(region)
            metrics, reason = collector.measure(trial)
            if (
                not (previous_mm & current_mm)
                and reason is None
                and metrics["contraction_depth"]
                <= max(
                    previous.metrics["contraction_depth"],
                    region.metrics["contraction_depth"],
                )
            ):
                merged = collector.finish(trial)
                if merged is not None and set(previous.outputs + region.outputs) <= set(
                    merged.outputs
                ):
                    horizontal[-1] = merged
                    continue
        horizontal.append(region)
    results = [
        r
        for r in horizontal
        if r.metrics["non_mm_compute"] > 0
        or (r.metrics["mm"] >= 2 and r.metrics["mm_depth"] == 1)
    ]
    if diagnostics is not None:
        diagnostics.update(
            {
                "mm_seeds": sum(is_matmul(n) for n in collector.nodes),
                "regions": len(results),
                "rejections": dict(collector.rejections),
                "selected_mm": len({n for r in results for n in r if is_matmul(n)}),
            }
        )
    return results
