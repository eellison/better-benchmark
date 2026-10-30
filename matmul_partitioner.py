"""Extract matmul (MM) micro-graphs from an FX graph, one pattern category at a time.

Input is the post-grad FX graph of a whole model. Output is a list of
MatmulRegion: small pieces of that graph, each one fusion pattern around one
to three MMs, so every category is covered on its own and a benchmark can be
read per category:

  prologue           one MM; only pointwise/gather/transparent work in front
                     of its operands (no work after it)
  epilogue           one MM; only pointwise/scatter/transparent work on its
                     result, possibly with side inputs (no work in front)
  prologue_epilogue  one MM with both
  horizontal         2 or 3 MMs that do not depend on each other, share an
                     operand (through transparent ops) and have the same
                     output shape; prologues/epilogues may surround them, and
                     every output depends on only one of the MMs
  horizontal_combine the same, but some op combines the results of >= 2 of
                     the MMs (e.g. silu(x @ wg) * (x @ wu))
  vertical           a connected chain of >= 2 dependent reduction stages
                     (MMs and ordinary reductions) with >= 1 MM, accepted by
                     the tensor-level slice analysis below

The single-MM categories are exclusive: an MM with work on both sides is only
"prologue_epilogue", never also "prologue" or "epilogue". Likewise a group
with a combining op is only "horizontal_combine". The categories
are structurally disjoint (one MM and no reduction / several independent MMs
and no reduction / chained reductions), so classify_pattern_graph() recovers
the category from a standalone repro.

"Transparent" ops (views, permutes, casts, expand, squeeze, ...) are always
included and never count as work. Producers of side inputs (an epilogue's
residual, an addmm bias) are absorbed through transparent and gather ops and
when computed only from graph inputs and constants (dropout masks, iota
masks, gelu' of a saved activation); side inputs computed from other
intermediates stay outside. Reductions other than MMs only appear in
"vertical"; in the other categories they are boundaries and their results
become inputs. Regions must replay standalone: no value leaves a region and
flows back into it, no outside op writes memory it uses, and unbind, softmax,
attention ops and copies to another device are boundaries.

Regions are benchmark windows, not a partition of the model: regions may
overlap and the graph is never modified. Selection looks only at op schemas,
value dependencies and tensor metadata.

Vertical regions
----------------
Every MM is a logical reduction: H[m, n] = sum_k A[m, k] * B[k, n] over the
domain [M, K, N] with reduced axis K. Tensor axes are tracked symbolically as
"atoms" (union-find classes with sizes): pointwise broadcasting, views,
reshapes (axis split/merge), permutes and MM batch/row/column axes unify
atoms; partial slices, gathers and unknown ops create fresh atoms. Values
computed from region inputs only get fresh atoms at every use, so only
dependencies on reduction stages relate axes.

Independent slices: an atom is a slice axis when it appears in every region
output and in every internal reduction-stage result that output depends on.
Different coordinates along it therefore read disjoint coordinates of every
internally computed reduction result; sharing external inputs does not
matter. For every stage:

  R = product of the reduced axis lengths              must be <= reduction_limit
  F = stage output elements needed per slice by later  must be <= intermediate_limit
      stages (0 if no later stage consumes it)

Candidates whose internal index relations are unknown (a gather/scatter or
unknown op reading an internal reduction result, a reshape that does not
align, an atom that appears twice in one tensor) are rejected. Gathers of
external tables stay eligible. Stages are absorbed greedily in graph order
and the analysis is recomputed for every trial.
"""

from __future__ import annotations

import itertools
import math
import operator
from collections import deque
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
        "aten::_grouped_mm",
        "aten::_scaled_grouped_mm",
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
# They can be inside a region but never count as real computation.
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
    """Real non-MM work: not scaffolding, a tuple projection or identity math."""
    return (
        node.op == "call_function"
        and not is_matmul(node)
        and node.target is not operator.getitem
        and op_name(node) not in SCAFFOLD_NAMES
        and tensor_shape(node) is not None
        and not _neutral(node)
    )


def mm_operands(node):
    """The two multiplied operands of an MM (skipping the bias of addmm & co)."""
    start = 1 if op_name(node) in {"aten::addmm", "aten::baddbmm", "aten::addmv"} else 0
    return [a for a in node.args[start : start + 2] if isinstance(a, Node)]


def horizontal_matmul_shapes_match(dots, ancestors, shapes):
    """MMs that do not depend on each other have equal output shapes.

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
        operands = mm_operands(node)
        if len(operands) != 2:
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


class _ReplaySafety:
    """Can a node subset of the graph run on its own, in one piece?

    Uses value dependencies and the alias/write sets of op schemas: no value
    may leave the subset and flow back into it, and no outside op may write
    memory the subset uses while it runs.
    """

    def __init__(self, gm):
        self.nodes = tuple(gm.graph.nodes)
        self.order = {n: i for i, n in enumerate(self.nodes)}
        self.shapes = {n: tensor_shape(n) for n in self.nodes}
        # aliases[n]: memory n's value may share: n itself, its storage, and
        # everything the base of a view or in-place op may alias.
        self.aliases = {}
        for node in self.nodes:
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

    def safe(self, core):
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


CATEGORIES = (
    "prologue",
    "epilogue",
    "prologue_epilogue",
    "horizontal",
    "horizontal_combine",
    "vertical",
)

_BIASED = {"aten::addmm", "aten::baddbmm", "aten::addmv"}
# Ops whose result is their first argument's values, possibly re-indexed.
_SAME_VALUES = {
    "aten::clone",
    "aten::contiguous",
    "aten::alias",
    "aten::detach",
    "aten::_to_copy",
    "aten::lift_fresh_copy",
    "prims::convert_element_type",
}
_RESHAPES = {
    "aten::view",
    "aten::reshape",
    "aten::_unsafe_view",
    "aten::squeeze",
    "aten::unsqueeze",
    "aten::flatten",
}
# Ops a shared operand may pass through and still count as the same input.
_SHARING_TRANSPARENT = _SAME_VALUES | _RESHAPES | {
    "aten::permute",
    "aten::t",
    "aten::transpose",
    "aten::expand",
}


@dataclass(frozen=True)
class MatmulLimits:
    """Size limits of one extracted region (also applied by curation)."""

    # 128 = one Triton tile without padding; at 192 (padded to 256) fused
    # kernels lost to separate ones (docs/vertical_fusion_study.md).
    reduction_limit: int = 128  # R of every vertical stage
    intermediate_limit: int = 128  # F of every vertical stage
    max_horizontal: int = 3  # MMs in a horizontal group; growth stops at 3
    max_mm: int = 3
    max_nodes: int = 48
    max_compute: int = 32  # MMs plus real computation
    max_inputs: int = 12
    max_outputs: int = 4
    max_stages: int = 8  # reduction stages in a vertical region


class MatmulRegion(list):
    """The FX nodes of one region, in graph order.

    outputs: nodes whose values the benchmark returns.
    metrics: sizes of the region (and the slice analysis for "vertical").
    category: one of CATEGORIES.
    mm_types: sorted MM op names such as ("addmm", "bmm"), used for naming.
    """

    def __init__(self, nodes, outputs, metrics, category):
        super().__init__(nodes)
        self.outputs = tuple(outputs)
        self.metrics = metrics
        self.category = category
        self.metrics["category"] = category
        self.mm_types = tuple(
            sorted({op_name(n).split("::")[-1] for n in nodes if is_matmul(n)})
        )


def stage_size(node, shapes):
    """R: elements reduced per output (K for an MM), or None if unknown."""
    if op_name(node) in {"aten::_grouped_mm", "aten::_scaled_grouped_mm"}:
        return None
    return reduction_size(node, shapes)


def _value_shapes(node):
    """Concrete shape, or a list of shapes for a tuple value, or None."""
    value = node.meta.get("val")
    if isinstance(value, (tuple, list)):
        shapes = []
        for item in value:
            if not isinstance(item, torch.Tensor):
                return None
            shape = _concrete(item.shape)
            if shape is None:
                return None
            shapes.append(shape)
        return shapes
    return tensor_shape(node)


def _concrete(dims):
    result = []
    for dim in dims:
        if isinstance(dim, int):
            result.append(dim)
        elif isinstance(dim, torch.SymInt) and dim.node.has_hint():
            result.append(dim.node.hint)
        else:
            return None
    return tuple(result)


def _bindings(node):
    schema = getattr(node.target, "_schema", None)
    if schema is None:
        return dict(node.kwargs)
    values = {arg.name: value for arg, value in zip(schema.arguments, node.args)}
    values.update(node.kwargs)
    for arg in schema.arguments:
        if arg.name not in values and arg.has_default_value():
            values[arg.name] = arg.default_value
    return values


# ---------------------------------------------------------------------------
# Index analysis for vertical candidates
# ---------------------------------------------------------------------------


class _Atoms:
    """Union-find over axis atoms; a class can be split into smaller atoms."""

    def __init__(self):
        self.parent, self.size, self.split = [], [], {}

    def new(self, size):
        self.parent.append(len(self.parent))
        self.size.append(size)
        return len(self.parent) - 1

    def find(self, atom):
        while self.parent[atom] != atom:
            self.parent[atom] = self.parent[self.parent[atom]]
            atom = self.parent[atom]
        return atom

    def leaves(self, atoms):
        result, stack = [], list(reversed(atoms))
        while stack:
            root = self.find(stack.pop())
            if root in self.split:
                stack.extend(reversed(self.split[root]))
            else:
                result.append(root)
        return result

    def unify(self, xs, ys):
        """Identify two row-major atom sequences of equal total size."""
        a, b = self.leaves(xs), self.leaves(ys)
        i = j = 0
        while i < len(a) and j < len(b):
            x, y = self.find(a[i]), self.find(b[j])
            if x in self.split:
                a[i : i + 1] = self.leaves([x])
                continue
            if y in self.split:
                b[j : j + 1] = self.leaves([y])
                continue
            if x != y:
                sx, sy = self.size[x], self.size[y]
                if sx == sy:
                    self.parent[y] = x
                else:
                    small, big, seq, k = (x, y, b, j) if sx < sy else (y, x, a, i)
                    if self.size[big] % self.size[small]:
                        return False
                    parts = [self.new(self.size[small]), self.new(self.size[big] // self.size[small])]
                    self.split[big] = parts
                    seq[k : k + 1] = parts
                    continue
            i += 1
            j += 1
        return i == len(a) and j == len(b)


class _Unknown(Exception):
    pass


def analyze_vertical(region, outputs, order, limits):
    """Slice analysis of a vertical candidate (see the module comment).

    Returns (summary, None) or (None, reason).
    """
    atoms = _Atoms()
    dims, tainted = {}, {}

    def fresh(shape):
        if shape is None or any(d <= 0 for d in shape):
            raise _Unknown("shape")
        return [[atoms.new(d)] if d != 1 else [] for d in shape]

    def of(node):
        """Dims of a value used by a region node.

        Values that do not depend on a stage (region inputs and work on them
        only) get fresh atoms at every use: they can be recomputed per slice,
        and relating axes through them (a shared bias, an iota mask [S, S])
        could equate unrelated axes.
        """
        if not tainted.get(node):
            shapes = _value_shapes(node)
            if shapes is None or (shapes and isinstance(shapes[0], tuple)):
                raise _Unknown("input_shape")
            return fresh(shapes)
        result = dims[node]
        if isinstance(result, tuple):
            raise _Unknown("tuple_input")
        return result

    def broadcast_into(out, out_shape, arg):
        shape = tensor_shape(arg)
        if shape is None or len(shape) > len(out_shape):
            raise _Unknown("broadcast")
        arg_dims = of(arg)
        for k in range(1, len(shape) + 1):
            if shape[-k] == out_shape[-k] and shape[-k] != 1:
                if not atoms.unify(arg_dims[-k], out[-k]):
                    raise _Unknown("broadcast")
            elif shape[-k] != 1:
                raise _Unknown("broadcast")

    def reshape(source, shape):
        out = fresh(shape)
        if not atoms.unify([a for d in of(source) for a in d], [a for d in out for a in d]):
            raise _Unknown("reshape")
        return out

    def rule(node, is_tainted):
        """Dims of node's value, or raise _Unknown for an unknown index map."""
        name = op_name(node)
        shape = _value_shapes(node)
        args = [a for a in value_inputs(node)]
        bind = _bindings(node)
        if node.target is operator.getitem:
            source, index = node.args
            value = dims.get(source) if source in region else None
            if isinstance(value, tuple):
                return value[index]
            return fresh(shape)
        if is_matmul(node):
            if name not in {"aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm",
                            "aten::mv", "aten::addmv", "aten::_int_mm", "aten::_scaled_mm"}:
                raise _Unknown("mm_kind")
            a, b = mm_operands(node)
            da, db = of(a), of(b)
            if name in {"aten::bmm", "aten::baddbmm"}:
                ok = atoms.unify(da[0], db[0]) and atoms.unify(da[2], db[1])
                out = [list(da[0]), list(da[1]), list(db[2])]
            elif name in {"aten::mv", "aten::addmv"}:
                ok = atoms.unify(da[1], db[0])
                out = [list(da[0])]
            else:
                ok = atoms.unify(da[1], db[0])
                out = [list(da[0]), list(db[1])]
            if not ok:
                raise _Unknown("mm")
            for extra in node.all_input_nodes:
                if extra not in (a, b):
                    broadcast_into(out, shape, extra)  # bias, scales
            return out
        if is_reduction(node):
            source = node.args[0] if node.args else None
            if not isinstance(source, Node):
                raise _Unknown("reduction_input")
            in_shape = tensor_shape(source)
            in_dims = of(source)
            axes = bind.get("dim", bind.get("dims"))
            axes = [axes] if isinstance(axes, int) else axes
            axes = range(len(in_shape)) if axes is None or list(axes) == [] else axes
            axes = {a % len(in_shape) for a in axes} if in_shape else set()
            outs = shape if isinstance(shape, list) else [shape]
            results = []
            for out_shape in outs:
                if len(out_shape) == len(in_shape):
                    expected = tuple(1 if k in axes else d for k, d in enumerate(in_shape))
                    result = [[] if k in axes else list(d) for k, d in enumerate(in_dims)]
                else:
                    expected = tuple(d for k, d in enumerate(in_shape) if k not in axes)
                    result = [list(d) for k, d in enumerate(in_dims) if k not in axes]
                if tuple(out_shape) != expected:
                    raise _Unknown("reduction_shape")
                results.append(result)
            return tuple(results) if isinstance(shape, list) else results[0]
        if shape is None or isinstance(shape, list) and name not in {
            "aten::split", "aten::split_with_sizes", "aten::chunk"
        }:
            raise _Unknown("shape")
        source = node.args[0] if node.args and isinstance(node.args[0], Node) else None
        if name in _SAME_VALUES or name in _RESHAPES:
            if source is None:
                raise _Unknown("view")
            if tensor_shape(source) == shape:
                return [list(d) for d in of(source)]
            return reshape(source, shape)
        if name in {"aten::permute", "aten::transpose", "aten::t"}:
            src = of(source)
            n = len(src)
            if name == "aten::permute":
                perm = [p % n for p in bind["dims"]]
            else:
                perm = list(range(n))
                if n >= 2:
                    d0, d1 = (bind["dim0"] % n, bind["dim1"] % n) if name == "aten::transpose" else (0, 1)
                    perm[d0], perm[d1] = perm[d1], perm[d0]
            return [list(src[p]) for p in perm]
        if name == "aten::expand":
            src, src_shape = of(source), tensor_shape(source)
            out = fresh(shape)
            offset = len(shape) - len(src_shape)
            for k, d in enumerate(src_shape):
                if d == shape[offset + k] and d != 1:
                    out[offset + k] = list(src[k])
            return out
        if name == "aten::slice":
            src, src_shape = of(source), tensor_shape(source)
            out = [list(d) for d in src]
            axis = bind.get("dim", 0) % len(src_shape)
            if shape[axis] != src_shape[axis]:
                out[axis] = fresh((shape[axis],))[0]
            return out
        if name == "aten::select":
            src, src_shape = of(source), tensor_shape(source)
            axis = bind["dim"] % len(src_shape)
            return [list(d) for k, d in enumerate(src) if k != axis]
        if name in {"aten::split", "aten::split_with_sizes", "aten::chunk"}:
            src, src_shape = of(source), tensor_shape(source)
            axis = bind.get("dim", 0) % len(src_shape)
            pieces = []
            for piece in shape:
                out = [list(d) for d in src]
                if piece[axis] != src_shape[axis]:
                    out[axis] = fresh((piece[axis],))[0]
                pieces.append(out)
            return tuple(pieces)
        tags = getattr(node.target, "tags", ())
        if torch.Tag.pointwise in tags and name.startswith(("aten::", "prims::")):
            out = fresh(shape)
            for arg in args:
                broadcast_into(out, shape, arg)
            return out
        # Gathers, scatters, cat and anything else: unknown index relation.
        if is_tainted:
            raise _Unknown(f"index_map:{name or node.target}")
        return fresh(shape)

    ordered = sorted(region, key=order.__getitem__)
    stages = [n for n in ordered if is_reduction(n)]
    deps = {}
    try:
        for node in ordered:
            inputs = [a for a in value_inputs(node) if a in region]
            tainted[node] = is_reduction(node) or any(tainted[a] for a in inputs)
            deps[node] = set().union(*(deps[a] for a in inputs)) | (
                {node} if is_reduction(node) else set()
            )
            if tainted[node]:
                dims[node] = rule(node, any(tainted[a] for a in inputs))
    except _Unknown as error:
        return None, f"unknown_index:{error}"
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
        return None, "unknown_index:rule"

    def leaves(node):
        value = dims[node]
        if isinstance(value, tuple):
            value = value[0]
        return atoms.leaves([a for d in value for a in d])

    for node in ordered:
        if node not in dims:
            continue
        value = dims[node]
        for item in value if isinstance(value, tuple) else (value,):
            flat = atoms.leaves([a for d in item for a in d])
            if len(flat) != len(set(flat)):
                return None, "aliased_axes"
    stage_atoms = {s: set(leaves(s)) for s in stages}
    slices = None
    for output in outputs:
        if not deps[output]:
            continue
        keep = set(leaves(output))
        for s in deps[output]:
            keep &= stage_atoms[s]
        slices = keep if slices is None else slices & keep
    slices = slices or set()
    feeds = {s: any(s in deps[t] for t in stages if t is not s) for s in stages}
    if not any(feeds.values()) or not any(is_matmul(s) for s in stages):
        return None, "no_vertical_chain"
    summary = []
    shapes = {}
    for s in stages:
        for a in s.all_input_nodes:
            shapes[a] = tensor_shape(a)
        r = stage_size(s, shapes)
        f = math.prod(atoms.size[a] for a in stage_atoms[s] - slices) if feeds[s] else 0
        if r is None or r <= 0:
            return None, "reduction_size_unknown"
        if r > limits.reduction_limit:
            return None, "reduction_limit"
        if f > limits.intermediate_limit:
            return None, "intermediate_limit"
        summary.append({"op": op_name(s).split("::")[-1], "R": r, "F": f})
    return {
        "slice_axes": sorted(atoms.size[a] for a in slices),
        "stages": summary,
    }, None


# ---------------------------------------------------------------------------
# Region search
# ---------------------------------------------------------------------------


class _Extractor:
    def __init__(self, gm, supported, limits):
        self.limits = limits
        self.supported = supported
        self.collector = _ReplaySafety(gm)
        self.nodes = self.collector.nodes
        self.order = self.collector.order
        self.shapes = self.collector.shapes
        self.rejections = {}
        # mm_dependent[n]: n depends on a supported MM through supported nodes.
        self.mm_dependent = {}
        # rooted[n]: n is pointwise/transparent work computed only from graph
        # inputs and constants (a dropout mask, an iota mask, gelu'(saved x)).
        self.rooted = {}
        for node in self.nodes:
            self.mm_dependent[node] = node in supported and (
                is_matmul(node) or any(self.mm_dependent[a] for a in value_inputs(node))
            )
            self.rooted[node] = (
                node in supported
                and not is_reduction(node)
                and node.target is not operator.getitem
                and all(
                    a.op in {"placeholder", "get_attr"} or self.rooted[a]
                    for a in value_inputs(node)
                )
            )

    def reject(self, reason):
        self.rejections[reason] = self.rejections.get(reason, 0) + 1

    def sort(self, nodes):
        return sorted(nodes, key=self.order.__getitem__)

    def connector(self, node, core):
        """Pointwise, indexing or transparent node that may join a region."""
        if node not in self.supported or is_reduction(node):
            return False
        if node.target is operator.getitem:
            source = node.args[0]
            return isinstance(source, Node) and (
                source in core or self.connector(source, core)
            )
        return True

    def metrics(self, core):
        dots = [n for n in core if is_matmul(n)]
        depth = {}
        for n in self.sort(core):
            depth[n] = max(
                (depth[a] for a in value_inputs(n) if a in core), default=0
            ) + int(is_reduction(n))
        return {
            "nodes": len(core),
            "compute": sum(is_matmul(n) or is_computation(n) for n in core),
            "mm": len(dots),
            "inputs": len({a for n in core for a in n.all_input_nodes if a not in core}),
            "contraction_depth": max(depth.values(), default=0),
        }

    def fits(self, core):
        m = self.metrics(core)
        l = self.limits
        return (
            m["nodes"] <= l.max_nodes
            and m["compute"] <= l.max_compute
            and m["mm"] <= l.max_mm
            and m["inputs"] <= l.max_inputs
        )

    def grow_backward(self, core, roots, *, side_branch=False):
        """Absorb pointwise/gather/transparent producers of ``roots``.

        Reductions, MMs and values computed from an MM stay outside and become
        inputs. For a side branch (inputs of an epilogue or of a vertical
        chain) only transparent and gather ops, and work computed only from
        graph inputs and constants (``rooted``), are absorbed. Computing a
        side input from other intermediate values (e.g. a residual's norm
        affine) is not work around the MM.
        """
        pending = deque(self.sort(r for r in roots if r not in core))
        visited = set()
        while pending:
            node = pending.popleft()
            if node in core or node in visited:
                continue
            visited.add(node)
            if not self.connector(node, core) or self.mm_dependent[node]:
                continue
            if side_branch and not (
                node.target is operator.getitem
                or op_name(node) in SCAFFOLD_NAMES | INDEX_NAMES
                or self.rooted[node]
            ):
                continue
            trial = core | {node}
            if not self.fits(trial) or not self.collector.safe(trial):
                visited.discard(node)  # may fit once another path is absorbed
                continue
            core = trial
            pending.extend(self.sort(value_inputs(node)))
        return core

    def grow_forward(self, core, sources, accept=None):
        """Yield ever larger windows absorbing consumers of ``sources``."""
        yield core
        pending = deque(self.sort(u for s in sources for u in s.users))
        visited = set()
        while pending:
            node = pending.popleft()
            if node in core or node in visited:
                continue
            visited.add(node)
            if not self.connector(node, core):
                continue
            trial = core | {node}
            if (
                not self.fits(trial)
                or not self.collector.safe(trial)
                or (accept is not None and not accept(trial))
            ):
                # A join is unsafe until its other branch is in the window;
                # adding that branch's last node re-queues it.
                visited.discard(node)
                continue
            core = trial
            yield core
            pending.extend(self.sort(node.users))

    def side_inputs(self, core, exclude=()):
        """Producers of the window's inputs that may be pulled in as branches."""
        return {
            a
            for n in core
            if n not in exclude
            for a in value_inputs(n)
            if a not in core
        }

    def finish(self, core, internal, required=()):
        """Choose outputs, prune to what they need and check every limit."""
        outputs = self.sort(
            n
            for n in core
            if internal(n) and (is_mutating(n) or any(u not in core for u in n.users))
        )
        if not outputs:
            self.reject("no_output")
            return None
        needed = set(outputs)
        stack = list(outputs)
        while stack:
            for a in value_inputs(stack.pop()):
                if a in core and a not in needed:
                    needed.add(a)
                    stack.append(a)
        if any(r not in needed for r in required):
            self.reject("pruned_required")
            return None
        metrics = self.metrics(needed)
        metrics["outputs"] = len(outputs)
        metrics["non_mm_compute"] = sum(is_computation(n) for n in needed)
        if len(outputs) > self.limits.max_outputs or not self.fits(needed):
            self.reject("limits")
            return None
        if not self.collector.safe(needed):
            self.reject("cycle_or_alias")
            return None
        return self.sort(needed), outputs, metrics

    # --- categories 1, 2 and prologue+epilogue ---------------------------

    def single(self, mm):
        operands = mm_operands(mm)
        core = self.grow_backward({mm}, operands)
        prologue = set(core) - {mm}
        snapshots = list(self.grow_forward(core, [mm]))
        for window in reversed(snapshots):
            window = self.grow_backward(window, self.side_inputs(window, prologue | {mm}) | (
                {a for a in mm.all_input_nodes if a not in operands}
            ), side_branch=True)
            # Values computed only from the prologue are not outputs.
            finished = self.finish(window, lambda n: n is mm or self._after(n, mm, window), (mm,))
            if finished is None:
                continue
            nodes, outputs, metrics = finished
            pro = self._prologue_nodes(set(nodes), mm)
            has_pro = any(is_computation(n) for n in pro)
            has_epi = any(is_computation(n) for n in set(nodes) - pro - {mm})
            if not has_pro and not has_epi:
                self.reject("bare_mm")
                return None
            category = (
                "prologue_epilogue" if has_pro and has_epi
                else "prologue" if has_pro else "epilogue"
            )
            return MatmulRegion(nodes, outputs, metrics, category)
        return None

    def _after(self, node, mm, window):
        """node depends on mm inside window."""
        stack, seen = [node], set()
        while stack:
            n = stack.pop()
            if n is mm:
                return True
            for a in value_inputs(n):
                if a in window and a not in seen:
                    seen.add(a)
                    stack.append(a)
        return False

    def _prologue_nodes(self, nodes, mm):
        """Nodes producing mm's multiplied operands (not the addmm bias)."""
        result, stack = set(), [a for a in mm_operands(mm) if a in nodes]
        while stack:
            n = stack.pop()
            if n in result:
                continue
            result.add(n)
            stack.extend(a for a in value_inputs(n) if a in nodes)
        return result

    # --- category 3 --------------------------------------------------------

    def horizontal_groups(self, mms):
        """Groups of 2..max_horizontal independent MMs sharing an operand."""

        def root(value):
            while (
                isinstance(value, Node)
                and op_name(value) in _SHARING_TRANSPARENT
                and value.args
                and isinstance(value.args[0], Node)
            ):
                value = value.args[0]
            return value

        ancestors = {}
        for n in self.nodes:
            ancestors[n] = set().union(
                *(ancestors[a] | ({a} if is_matmul(a) else set()) for a in n.all_input_nodes)
            )
        by_key = {}
        for mm in mms:
            shape = self.shapes.get(mm)
            if shape is None:
                continue
            for operand in mm_operands(mm):
                key = root(operand), shape
                if mm not in by_key.setdefault(key, []):
                    by_key[key].append(mm)
        groups, seen = [], set()
        for (_, _), members in sorted(
            by_key.items(), key=lambda kv: self.order[kv[1][0]]
        ):
            chosen = []
            for mm in self.sort(members):
                if all(mm not in ancestors[c] and c not in ancestors[mm] for c in chosen):
                    chosen.append(mm)
                if len(chosen) == self.limits.max_horizontal:
                    break  # a 4th sharing MM is never added
            key = frozenset(chosen)
            if len(chosen) >= 2 and key not in seen:
                seen.add(key)
                groups.append(chosen)
        return groups

    def horizontal(self, group):
        members = set(group)
        core = self.grow_backward(members, [a for mm in group for a in mm_operands(mm)])
        snapshots = list(self.grow_forward(core, group))
        for window in reversed(snapshots):
            window = self.grow_backward(
                window, self.side_inputs(window, members), side_branch=True
            )
            finished = self.finish(
                window, lambda n: any(self._after(n, mm, window) for mm in group), group
            )
            if finished is not None:
                category = (
                    "horizontal_combine" if combines(finished[0], group) else "horizontal"
                )
                return MatmulRegion(*finished, category)
        return None

    # --- category 4 --------------------------------------------------------

    def reach(self, start, forward, core):
        """Connector nodes reachable from ``start`` and the stages they hit."""
        found, hits = set(), set()
        pending = self.sort(start)
        while pending:
            node = pending.pop()
            for nxt in node.users if forward else value_inputs(node):
                if nxt in start or nxt in found or nxt in core:
                    continue
                if nxt not in self.supported:
                    continue
                if is_reduction(nxt):
                    hits.add(nxt)
                elif nxt.target is operator.getitem and is_reduction(nxt.args[0]):
                    # A tuple projection of a stage (var_mean) is part of
                    # the path to or from that stage.
                    found.add(nxt)
                    if forward:
                        pending.append(nxt)
                    else:
                        hits.add(nxt.args[0])
                elif self.connector(nxt, core | found) and len(found) < self.limits.max_nodes:
                    found.add(nxt)
                    pending.append(nxt)
        return found, hits

    def vertical_check(self, core):
        tainted = self._tainted(core)
        outputs = [
            n for n in core
            if n in tainted and (is_mutating(n) or any(u not in core for u in n.users))
        ]
        return analyze_vertical(core, outputs, self.order, self.limits)

    def _tainted(self, core):
        result = set()
        for n in self.sort(core):
            if is_reduction(n) or any(a in result for a in value_inputs(n)):
                result.add(n)
        return result

    def vertical(self, seed):
        limits = self.limits
        size = stage_size(seed, self.shapes)
        if size is None or size > limits.reduction_limit:
            return None
        core = {seed}
        while sum(is_reduction(n) for n in core) < limits.max_stages:
            forward, ahead = self.reach(core, True, set())
            backward, behind = self.reach(core, False, set())
            candidates = []
            for stage in ahead:
                path = forward & self.reach({stage}, False, set())[0]
                candidates.append((stage, path))
            for stage in behind:
                path = backward & self.reach({stage}, True, set())[0]
                candidates.append((stage, path))
            added = False
            for stage, path in sorted(candidates, key=lambda c: self.order[c[0]]):
                if stage in core or stage not in self.supported:
                    continue
                r = stage_size(stage, self.shapes)
                if r is None or r > limits.reduction_limit:
                    continue
                trial = core | path | {stage}
                # Tuple results (var_mean) bring their projections along.
                trial |= {u for u in stage.users if u.target is operator.getitem and u in self.supported}
                if not self.fits(trial) or not self.collector.safe(trial):
                    continue
                summary, reason = self.vertical_check(trial)
                if summary is None and reason != "no_vertical_chain":
                    self.reject("vertical_" + reason.split(":")[0])
                    continue
                core = trial
                added = True
                break
            if not added:
                break
        if self.vertical_check(core)[0] is None:
            return None
        tainted = self._tainted(core)
        snapshots = list(
            self.grow_forward(
                core, self.sort(tainted),
                accept=lambda t: self.vertical_check(t)[0] is not None,
            )
        )
        for window in reversed(snapshots):
            for extended in (self.grow_backward(window, self.side_inputs(window), side_branch=True), window):
                finished = self.finish(extended, lambda n, w=extended: n in self._tainted(w), (seed,))
                if finished is None:
                    continue
                nodes, outputs, metrics = finished
                summary, reason = analyze_vertical(set(nodes), outputs, self.order, limits)
                if summary is None:
                    self.reject("vertical_final_" + reason.split(":")[0])
                    continue
                metrics["vertical"] = summary
                return MatmulRegion(nodes, outputs, metrics, "vertical")
        return None


def supported_nodes(gm, is_supported, excluded=()):
    """Nodes that may be inside a region at all (standalone-replay boundaries)."""
    excluded = set(excluded)

    def local_copy(node):
        if op_name(node) != "aten::_to_copy" or not node.args or not isinstance(node.args[0], Node):
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
        and op_name(n) != "aten::unbind"
        and not any(x in op_name(n) for x in ("softmax", "attention"))
    }
    for node in gm.graph.nodes:
        if (
            node in supported
            and node.target is operator.getitem
            and (not isinstance(node.args[0], Node) or node.args[0] not in supported)
        ):
            supported.remove(node)
    return supported


def get_matmul_regions(gm, is_supported, *, limits=None, excluded=(), categories=CATEGORIES, diagnostics=None):
    """Return MatmulRegions of ``gm`` for the requested categories."""
    limits = limits or MatmulLimits()
    supported = supported_nodes(gm, is_supported, excluded)
    ex = _Extractor(gm, supported, limits)
    from mm_region_quality import classify_region_quality

    mms = [n for n in ex.nodes if is_matmul(n) and n in supported]
    regions = []
    if {"prologue", "epilogue", "prologue_epilogue"} & set(categories):
        for mm in mms:
            region = ex.single(mm)
            if region is not None and region.category in categories:
                regions.append(region)
    if {"horizontal", "horizontal_combine"} & set(categories):
        for group in ex.horizontal_groups(mms):
            region = ex.horizontal(group)
            if region is not None and region.category in categories:
                regions.append(region)
    if "vertical" in categories:
        vertical, seen = [], set()
        for mm in mms:
            region = ex.vertical(mm)
            if region is not None and frozenset(region) not in seen:
                seen.add(frozenset(region))
                vertical.append(region)
        regions.extend(
            r for r in vertical
            if not any(set(r) < set(o) for o in vertical)
        )
    kept = []
    for region in regions:
        if classify_region_quality(region)["category"] == "core":
            kept.append(region)
        else:
            ex.reject("no_substantive_fusion")
    if diagnostics is not None:
        diagnostics.update(
            {
                "mm_seeds": len(mms),
                "regions": {c: sum(r.category == c for r in kept) for c in CATEGORIES},
                "rejections": dict(sorted(ex.rejections.items())),
            }
        )
    return kept


def classify_pattern_graph(gm):
    """Category of a standalone repro graph, or None if it fits none."""
    nodes = list(gm.graph.nodes)
    mms = [n for n in nodes if is_matmul(n)]
    reductions = [n for n in nodes if is_reduction(n) and not is_matmul(n)]
    ancestors = {}
    for n in nodes:
        ancestors[n] = set().union(*(ancestors[a] | {a} for a in value_inputs(n)))
    chained = any(s in ancestors[t] for s in mms + reductions for t in mms + reductions)
    if chained and mms:
        return "vertical"
    if reductions or not mms:
        return None
    if len(mms) == 1:
        mm = mms[0]
        pro = set()
        for operand in mm_operands(mm):
            pro |= ancestors[operand] | {operand}
        has_pro = any(is_computation(n) for n in pro)
        has_epi = any(is_computation(n) for n in nodes if n not in pro and n is not mm)
        if has_pro and has_epi:
            return "prologue_epilogue"
        return "prologue" if has_pro else "epilogue" if has_epi else None
    shapes = {tensor_shape(m) for m in mms}
    if len(mms) <= 3 and len(shapes) == 1 and None not in shapes:
        return "horizontal_combine" if combines(nodes, mms) else "horizontal"
    return None


def combines(nodes, mms):
    """Some op of ``nodes`` (graph order) depends on >= 2 of ``mms``.

    The graph's output node does not count: returning several MM results
    separately is not combining them.
    """
    reached = {}
    for n in nodes:
        if n.op != "call_function":
            continue
        reached[n] = set().union(
            *(reached[a] for a in value_inputs(n) if a in reached)
        ) | ({n} if n in mms else set())
        if len(reached[n]) >= 2:
            return True
    return False


_COMMUTATIVE = {
    "aten::add", "aten::mul", "aten::maximum", "aten::minimum", "aten::eq",
    "aten::ne", "aten::logical_and", "aten::logical_or", "prims::add", "prims::mul",
}


def structure_key(gm):
    """Hash of a repro's compute DAG and its input/output dtypes.

    Transparent ops (views, casts, copies) and scalar literals are ignored, so
    repros with the same key differ only in layouts, internal casts or literal
    values; curation keeps one representative per key (pattern family).
    """
    import hashlib

    nodes = list(gm.graph.nodes)
    source, label, inputs = {}, {}, {}
    for n in nodes:
        if n.op != "call_function":
            continue
        tensor_args = [source.get(a, a) for a in n.all_input_nodes]
        if n.target is operator.getitem or op_name(n) in SCAFFOLD_NAMES:
            source[n] = tensor_args[0] if tensor_args else n
            if tensor_args:
                continue
        label[n] = op_name(n) or str(n.target)
        inputs[n] = tensor_args
    # Weisfeiler-Lehman refinement over producers; regions are small.
    for _ in range(len(label)):
        new = {}
        for n, args in inputs.items():
            parts = [label.get(a, "input") for a in args]
            if op_name(n) in _COMMUTATIVE:
                parts.sort()
            new[n] = hashlib.sha256(f"{label[n]}({','.join(parts)})".encode()).hexdigest()[:16]
        label = new
    def dtypes(values):
        return ",".join(
            sorted({str(v.meta["val"].dtype) for v in values if isinstance(v.meta.get("val"), torch.Tensor)})
        )

    output = next(n for n in nodes if n.op == "output")
    signature = (
        ",".join(sorted(label.values()))
        + "|in:" + dtypes(n for n in nodes if n.op == "placeholder")
        + "|out:" + dtypes(output.all_input_nodes)
    )
    return hashlib.sha256(signature.encode()).hexdigest()[:12]
