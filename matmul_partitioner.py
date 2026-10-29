"""Pick small, self-contained matmul (MM) benchmark regions from an FX graph.

Input is the post-grad FX graph of a whole model. Output is a list of
MatmulRegion: small pieces of that graph that contain at least one MM plus the
surrounding work a fusing compiler could put into the same kernel, e.g.

    (x - mean) * rstd -> mm(., W) -> + bias -> gelu     prologue + epilogue
    mm(x, Wg), mm(x, Wu) -> silu(.) * .                 horizontal MMs + join
    mm(x, W1) -> relu -> mm(., W2)                      vertical MM chain

Regions are benchmark *windows*, not a partition of the model: two regions may
share nodes, and the model graph is never modified. Selection only looks at op
schemas, value dependencies and tensor metadata. There are no per-model rules,
no pattern templates and no target number of regions.
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

# ---------------------------------------------------------------------------
# Rules (the numbers are MatmulLimits defaults)
# ---------------------------------------------------------------------------
# "Reduction" means an MM-family op (it reduces over K) or an op tagged
# torch.Tag.reduction (sum, mean, amax, var_mean, ...).
#
# 1. Reduction chain. A single dependency path inside a region may contain at
#    most 5 reductions (max_reductions_chain). Parallel branches are counted
#    separately. Five fits "norm -> MM -> norm" even when each norm is
#    written as two chained reductions (mean, then variance), provided the
#    norms are small enough for rule 2.
#
# 2. Reduction size. An ordinary reduction may reduce at most 128 elements
#    per output (max_internal_dim; the product of its reduced axes). An
#    ordinary reduction that is larger, or whose size is unknown, is never
#    inside a region: its result becomes a region input. MMs obey the same
#    limit on K, except the first MM of a chain: a large projection can still
#    anchor a region, but an MM fed by another MM of the region needs K <= 128.
#
# 3. Horizontal MMs. Two MMs where neither depends on the other must produce
#    exactly the same output shape (batch dims, M and N; K may differ). MMs
#    that feed each other may change shape freely.
#
# 4. Useful outputs only. Every region output must depend on an MM *and* on
#    real computation (arithmetic, reductions, indexing). Views, casts, copies
#    and allocations do not count, so "mm -> view -> float()" is not a
#    benchmark. Exception: independent MMs side by side are useful on their
#    own (a horizontal group of bare MMs).
#
# 5. No low-value regions. A single-MM region whose only extra work is
#    alignment padding/cropping, packing the addmm bias, copying a whole
#    input, or a lone "bmm == -inf" mask is dropped (mm_region_quality.py).
#
# 6. Standalone replay. Unbind, softmax, attention and grouped-MM ops are
#    never inside a region, nor are copies to another device. No value may
#    leave a region and flow back into it, and no op outside a region may
#    write memory the region uses while it runs.
#
# Search
# ------
# For every MM (the "seed"):
#   a. Pull in input-side work in front of the seed (the prologue). Several
#      prologue sizes, counted in reductions, are tried separately, so that a
#      long chain after the MM cannot crowd out a "norm -> MM -> norm" window.
#   b. Walk forward through the seed's users. Each step adds one consumer
#      plus whatever other MM results that consumer needs (e.g. the second
#      MM at a join). A step is kept only if every limit still holds.
#   c. Keep the largest snapshot of this growth that has useful outputs,
#      stays within the output/input limits and replays safely. Nodes that
#      only feed dropped values are removed.
# Then drop regions that are strict subsets of another region, merge
# neighboring independent regions into horizontal groups when the rules still
# hold, and apply rules 4 and 5 to the result.
#
# The mm_repro/ corpus is built from these regions later, with tighter size
# limits, normalization and deduplication (scripts/extract_matmul_corpus.py).
# Being selected here does not mean one GPU kernel can run the region or that
# fusing it is faster; that needs a real compiler and measurement.
# ---------------------------------------------------------------------------

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
# Ops that only reinterpret, cast, copy or allocate data, or query sizes.
# They can be inside a region but never count as real computation (rule 4).
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
# Ops that read only the shape/dtype of their first argument, not its values.
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
    """Size limits for one region while searching (see the rules above).

    The exported corpus applies tighter limits afterwards
    (scripts/curate_mm_corpus.py).
    """

    max_mm: int = 4  # MM nodes in the region
    max_nodes: int = 96  # all FX nodes, including views and casts
    max_compute: int = 48  # MMs plus real computation (no views/casts)
    max_outputs: int = 4  # tensors the benchmark returns
    max_inputs: int = 24  # distinct values fed in from outside the region
    max_reductions_chain: int = 5  # reductions on one dependency path (rule 1)
    max_internal_dim: int = 128  # elements per reduction, K of chained MMs (rule 2)


class MatmulRegion(list):
    """The FX nodes of one region, in graph order.

    outputs: nodes whose values the benchmark returns (rule 4 decides which).
    metrics: sizes measured by _Collector.measure() and finish().
    mm_types: sorted MM op names such as ("addmm", "bmm"), used for naming.
    """

    def __init__(self, nodes, outputs, metrics):
        super().__init__(nodes)
        self.outputs = tuple(outputs)
        self.metrics = metrics
        self.mm_types = tuple(
            sorted({op_name(n).split("::")[-1] for n in nodes if is_matmul(n)})
        )


def op_name(node):
    """Schema name such as "aten::mm", or "" for targets without a schema."""
    return getattr(getattr(node.target, "_schema", None), "name", "")


def is_matmul(node):
    return node.op == "call_function" and op_name(node) in MM_NAMES


def is_reduction(node):
    """MM-family ops and ops tagged torch.Tag.reduction."""
    return is_matmul(node) or (
        node.op == "call_function"
        and torch.Tag.reduction in getattr(node.target, "tags", ())
    )


def tensor_shape(node):
    """Concrete shape of the node's tensor (first tensor of a tuple), or None."""
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
    """Inputs whose values the node reads.

    zeros_like/size/... read only metadata of their first argument, and copy_
    overwrites its destination, so those arguments are not value inputs.
    """
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
    """Identity arithmetic such as x * 1, x + 0, x - 0, x / 1 or x ** 1."""
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
    """Real non-MM work: not scaffolding, a tuple projection or identity math.

    Only such nodes make an MM worth benchmarking as a fusion (rule 4).
    """
    return (
        node.op == "call_function"
        and not is_matmul(node)
        and node.target is not operator.getitem
        and op_name(node) not in SCAFFOLD_NAMES
        and tensor_shape(node) is not None
        and not _neutral(node)
    )


def _operands(node):
    """The two matrix operands of an MM (skipping the bias of addmm & co)."""
    start = 1 if op_name(node) in {"aten::addmm", "aten::baddbmm", "aten::addmv"} else 0
    return node.args[start : start + 2]


def horizontal_matmul_shapes_match(dots, ancestors, shapes):
    """Rule 3: MMs that do not depend on each other have equal output shapes.

    ancestors[n] holds the MMs that n depends on. K is not compared, and
    unknown output shapes never count as equal.
    """
    return all(
        shapes[a] is not None and shapes[a] == shapes[b]
        for a, b in itertools.combinations(dots, 2)
        if a not in ancestors[b] and b not in ancestors[a]
    )


def reduction_size(node, shapes):
    """Elements reduced per output: K for an MM, else the reduced axes' product.

    For example sum(x[8, 4, 16], dim=(1, 2)) reduces 64 elements. Returns
    None when the input shape or the reduced axes are unknown.
    """
    if is_matmul(node):
        operands = _operands(node)
        if len(operands) != 2 or not isinstance(operands[0], Node):
            return None
        shape = shapes.get(operands[0])
        return shape[-1] if shape else None
    inputs = value_inputs(node)
    shape = shapes.get(inputs[0]) if inputs else None
    if shape is None:
        return None
    bindings = {
        arg.name: value for arg, value in zip(node.target._schema.arguments, node.args)
    }
    bindings.update(node.kwargs)
    dims = bindings.get("dim", bindings.get("dims"))
    dims = [dims] if isinstance(dims, int) else dims
    dims = range(len(shape)) if dims is None or dims == [] else dims
    if any(not isinstance(dim, int) for dim in dims):
        return None
    axes = {dim % len(shape) for dim in dims} if shape else set()
    return math.prod(shape[axis] for axis in axes)


class _Collector:
    """Grows and checks candidate regions for one graph.

    A candidate ("core") is a set of FX nodes. Graph order is used as a
    topological order throughout.
    """

    def __init__(self, gm, supported, limits):
        self.nodes = tuple(gm.graph.nodes)
        self.order = {n: i for i, n in enumerate(self.nodes)}
        self.supported = supported
        self.limits = limits
        self.shapes = {n: tensor_shape(n) for n in self.nodes}
        self.reduction_sizes = {
            n: reduction_size(n, self.shapes) for n in self.nodes if is_reduction(n)
        }
        # mm_dependent[n]: n depends, through supported nodes, on a supported MM.
        self.mm_dependent = {}
        # prologue_depth[n]: for supported nodes that do not depend on an MM,
        # the most reductions on one path ending at n. It bounds the prologue
        # sizes that are worth trying for a seed (search step a).
        self.prologue_depth = {}
        # aliases[n]: memory n's value may share: n itself, its storage, and
        # everything the base of a view or in-place op may alias.
        self.aliases = {}
        self.rejections = Counter()
        for node in self.nodes:
            self.mm_dependent[node] = node in supported and (
                is_matmul(node) or any(self.mm_dependent[a] for a in value_inputs(node))
            )
            self.prologue_depth[node] = (
                max((self.prologue_depth[a] for a in value_inputs(node)), default=0)
                + int(is_reduction(node))
                if node in supported and not self.mm_dependent[node]
                else 0
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
        # Memory each node reads, and memory its schema says it writes.
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

    def cone(self, roots, only_mm_paths=False, reduction_budget=None):
        """Roots plus their supported producers, walking backward.

        The walk never goes past an MM: an MM is included, its inputs are not.
        only_mm_paths: keep only nodes that depend on a supported MM.
        reduction_budget: how many reductions the walk may include (None means
            unlimited). The first reduction over budget, together with its
            tuple projections, stays outside, so its results become inputs.
        Returns None if the cone has more than max_nodes nodes.
        """
        result, visited = set(), set()
        pending = [(node, reduction_budget) for node in roots]
        while pending:
            node, remaining = pending.pop()
            key = node, remaining
            if key in visited or node not in self.supported:
                continue
            visited.add(key)
            if remaining is not None:
                boundary = node
                while (
                    boundary.target is operator.getitem
                    and boundary.args
                    and isinstance(boundary.args[0], Node)
                ):
                    boundary = boundary.args[0]
                if remaining == 0 and is_reduction(boundary):
                    continue
                remaining -= int(is_reduction(node))
            if only_mm_paths and not self.mm_dependent[node]:
                continue
            result.add(node)
            if len(result) > self.limits.max_nodes:
                return None
            if is_matmul(node):
                continue
            pending.extend((parent, remaining) for parent in value_inputs(node))
        return result

    def measure(self, core):
        """Check a candidate against MatmulLimits and rules 2 and 3.

        Returns (metrics, None), or (None, reason) for the first violation.
        Metrics ("depth" = most such ops on one dependency path):
          nodes, compute (MMs + real computation), mm,
          mm_depth: 1 means no MM feeds another MM,
          reduction_depth: chained ordinary reductions,
          contraction_depth: chained reductions of any kind (MM or ordinary);
              this is what max_reductions_chain limits.
        """
        ordered = sorted(core, key=self.order.__getitem__)
        dots = [n for n in ordered if is_matmul(n)]
        l = self.limits
        if len(core) > l.max_nodes or len(dots) > l.max_mm:
            return None, "size"
        # mm_ancestors[n]: MMs of the candidate that n depends on, n included.
        mm_ancestors, mm_depth, reduction_depth, chain_depth = {}, {}, {}, {}
        for node in ordered:
            inputs = [a for a in value_inputs(node) if a in core]
            upstream = set().union(*(mm_ancestors[a] for a in inputs))
            ordinary_reduction = is_reduction(node) and not is_matmul(node)
            # Rule 2. An MM with no MM upstream in the candidate is exempt.
            if ordinary_reduction or (is_matmul(node) and upstream):
                size = self.reduction_sizes[node]
                if size is None or size <= 0:
                    return None, "reduction_metadata"
                if size > l.max_internal_dim:
                    return None, "chained_reduction_size"
            mm_depth[node] = max((mm_depth[a] for a in inputs), default=0) + int(
                is_matmul(node)
            )
            reduction_depth[node] = max(
                (reduction_depth[a] for a in inputs), default=0
            ) + int(ordinary_reduction)
            chain_depth[node] = max((chain_depth[a] for a in inputs), default=0) + int(
                is_reduction(node)
            )
            mm_ancestors[node] = upstream | ({node} if is_matmul(node) else set())
            shape = self.shapes[node]
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
                values = [a.meta.get("val"), b.meta.get("val"), node.meta.get("val")]
                devices = {v.device for v in values if isinstance(v, torch.Tensor)}
                if len(devices) > 1 or any(
                    isinstance(v, torch.Tensor) and v.layout != torch.strided
                    for v in values
                ):
                    return None, "device_or_layout"
            elif ordinary_reduction:
                source = node.all_input_nodes[0] if node.all_input_nodes else None
                if self.shapes.get(source) is None or shape is None or not math.prod(shape):
                    return None, "reduction_metadata"
        if not horizontal_matmul_shapes_match(dots, mm_ancestors, self.shapes):
            return None, "parallel_output_shape"
        metrics = {
            "nodes": len(core),
            "compute": sum(is_matmul(n) or is_computation(n) for n in core),
            "mm": len(dots),
            "mm_depth": max(mm_depth.values(), default=0),
            "reduction_depth": max(reduction_depth.values(), default=0),
            "contraction_depth": max(chain_depth.values(), default=0),
        }
        if metrics["compute"] > l.max_compute:
            return None, "compute"
        if metrics["contraction_depth"] > l.max_reductions_chain:
            return None, "contraction_depth"
        return metrics, None

    def add_prologues(self, core, reduction_budget=None):
        """Absorb the producers of the candidate's inputs where limits allow.

        Each input that does not come from an MM is tried once, with its whole
        backward cone (up to reduction_budget reductions). Values produced by
        an MM stay with that MM's region: the relu in "mm -> relu -> mm" is an
        epilogue of the first MM, not a prologue of the second.
        """
        roots = {a for n in core for a in value_inputs(n) if a not in core}
        for root in sorted(roots, key=self.order.__getitem__):
            if is_matmul(root) or self.mm_dependent[root]:
                continue
            prefix = self.cone((root,), reduction_budget=reduction_budget)
            if prefix is None:
                continue
            trial = core | prefix
            _, reason = self.measure(trial)
            if reason is None and self.safe(trial):
                core = trial
        return core

    def grow(self, seed, backward_budget=0):
        """Yield ever larger candidates around ``seed`` (search steps a and b).

        The caller keeps the last snapshot that finish() accepts, so an
        oversized final window falls back to a smaller valid one.
        """
        core = {seed}
        yield core
        core = self.add_prologues(core, backward_budget)
        yield core
        pending = deque(sorted(seed.users, key=self.order.__getitem__))
        visited = set()
        while pending:
            node = pending.popleft()
            if node in visited or node not in self.supported:
                continue
            visited.add(node)
            # The consumer plus the MM-dependent values it needs (a join pulls
            # in the other branch up to and including that branch's MM).
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
            yield core
            for n in sorted(added, key=self.order.__getitem__):
                pending.extend(sorted(n.users, key=self.order.__getitem__))
        # Finally try the unlimited prologue in front of the grown window.
        yield self.add_prologues(core)

    def safe(self, core):
        """Rule 6: can the candidate run on its own, in one piece?"""
        last = max(self.order[n] for n in core)
        # (a) No value leaves the candidate and flows back into it.
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
        # (b) No outside op between the candidate's first and last node writes
        #     memory the candidate reads or writes.
        reads, writes = self.reads, self.writes
        ordered = sorted(core, key=self.order.__getitem__)
        touched = set().union(*(reads[n] | writes[n] for n in core))
        if any(
            writes[n] & touched
            for n in self.nodes[self.order[ordered[0]] : last]
            if n not in core
        ):
            return False
        # (c) Inside, reading memory that an earlier node wrote requires a
        #     data dependency on that writer, so replay keeps the order.
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
        """Turn a candidate into a MatmulRegion, or return None.

        Outputs are the nodes used outside the candidate (or mutating) that
        are useful under rule 4. allow_trivial also accepts a window with no
        real computation at all; such bare-MM regions only survive if the
        horizontal pass later groups them (rule 4's exception).
        """
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
    """Return the benchmark regions of ``gm`` (see the rules above).

    is_supported: extra per-node predicate (the capture hook passes its fusion
        support check). MM, indexing, view/cast/copy and pointwise or
        reduction ATen ops are supported regardless, minus rule 6 exceptions.
    limits: MatmulLimits for the search; defaults when None.
    excluded: nodes that must stay outside every region, e.g. work that
        runtime constant folding precomputes.
    diagnostics: optional dict that receives seed/region counts and why
        candidates were rejected; useful when tuning the rules.
    """
    limits = limits or MatmulLimits()
    excluded = set(excluded)

    def local_copy(node):
        """False for a _to_copy that moves data to another device."""
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

    # Nodes that may be inside a region at all.
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
        # Unbind stays a boundary even if is_supported accepts it.
        and op_name(n) != "aten::unbind"
        and not any(x in op_name(n) for x in ("softmax", "attention", "grouped_mm"))
    }
    # Rule 2: ordinary reductions that are too large or of unknown size are
    # boundaries on either side of an MM.
    shapes = {node: tensor_shape(node) for node in gm.graph.nodes}
    for node in tuple(supported):
        if is_reduction(node) and not is_matmul(node):
            size = reduction_size(node, shapes)
            if size is None or size <= 0 or size > limits.max_internal_dim:
                supported.remove(node)
    # A tuple projection (getitem) of an op outside the regions is a boundary
    # too, so each tuple element becomes an ordinary tensor input.
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
        # Search step a: one growth per prologue budget. Budget 0 starts with
        # no reductions in front of the seed. Larger budgets let a pre-norm
        # in first, which leaves less of the chain budget for work after it.
        backward_depth = max(
            (collector.prologue_depth[a] for a in value_inputs(node)), default=0
        )
        for budget in range(min(backward_depth, limits.max_reductions_chain - 1) + 1):
            # Step c: keep the latest snapshot that finish() accepts.
            for core in reversed(list(collector.grow(node, budget))):
                region = collector.finish(core, allow_trivial=True)
                if region is None:
                    continue
                key = frozenset(region), frozenset(region.outputs)
                if key not in seen:
                    seen.add(key)
                    results.append(region)
                break
    # Drop regions that another region contains entirely, outputs included.
    results = [
        r
        for r in results
        if not any(
            set(r) < set(other) and set(r.outputs) <= set(other.outputs)
            for other in results
        )
    ]
    # Horizontal pass: merge each region into its predecessor when they share
    # no MM, the union still meets every rule, no dependency path gets longer
    # than in either region, and no output is lost. Independent regions need
    # no shared input or join. Only neighbors are tried, keeping this linear.
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
    # Rule 4: real computation, or a group of independent MMs.
    results = [
        r
        for r in horizontal
        if r.metrics["non_mm_compute"] > 0
        or (r.metrics["mm"] >= 2 and r.metrics["mm_depth"] == 1)
    ]
    # Rule 5 runs on finished regions, so it cannot change how other regions
    # grew or merged. Only nodes inside the region are inspected: a zero
    # tensor produced outside the region is just an input here.
    from mm_region_quality import classify_region_quality

    core_results = []
    for region in results:
        if classify_region_quality(region)["category"] == "core":
            core_results.append(region)
        else:
            collector.rejections["no_substantive_fusion"] += 1
    results = core_results

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
