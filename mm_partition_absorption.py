"""Absorb one unassigned mm/addmm/bmm into each unchanged normal BB partition.

Partitions take a producer MM first (epilogue). Then still-unused partitions
without scans and with at most one reduction take a consumer MM through a
multiplied operand (prologue); a reduction must normalize rows of the MM's A
operand along K. addmm bias-only preparation is excluded. Each original
partition/MM is used once; augmented partitions do not grow. This is a disjoint
assignment inside these two categories, not a size-limited window search.
Horizontal benchmark windows are separate.
"""

from __future__ import annotations

from collections import defaultdict, deque

import torch

from matmul_partitioner import is_reduction, is_reduction_or_scan


_MM = {torch.ops.aten.mm, torch.ops.aten.addmm, torch.ops.aten.bmm}


def _bits(value):
    while value:
        bit = value & -value
        yield bit.bit_length() - 1
        value ^= bit


class _PartitionGraph:
    """Reachability of the current partition quotient, including prior merges."""

    def __init__(self, edges):
        degree = {node: 0 for node in edges}
        for children in edges.values():
            for child in children:
                degree[child] += 1
        queue = deque(node for node, n in degree.items() if n == 0)
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for child in edges[node]:
                degree[child] -= 1
                if degree[child] == 0:
                    queue.append(child)
        if len(order) != len(edges):
            raise ValueError("normal BB partition quotient contains a cycle")
        self.ancestors = [0] * len(edges)
        self.descendants = [0] * len(edges)
        for node in order:
            for child in edges[node]:
                self.ancestors[child] |= self.ancestors[node] | (1 << node)
        for node in reversed(order):
            for child in edges[node]:
                self.descendants[node] |= self.descendants[child] | (1 << child)

    def permits(self, source, target):
        interior = self.descendants[source] & self.ancestors[target]
        return not interior & ~((1 << source) | (1 << target))

    def absorb(self, source, target, partition):
        removed = target if partition == source else source
        pair = (1 << source) | (1 << target)
        before = (self.ancestors[source] | self.ancestors[target]) & ~pair
        after = (self.descendants[source] | self.descendants[target]) & ~pair
        for node in _bits(before):
            self.descendants[node] = (
                self.descendants[node] | after | (1 << partition)
            ) & ~(1 << removed)
        for node in _bits(after):
            self.ancestors[node] = (
                self.ancestors[node] | before | (1 << partition)
            ) & ~(1 << removed)
        self.ancestors[partition], self.descendants[partition] = before, after
        self.ancestors[removed] = self.descendants[removed] = 0


def _is_mm(node):
    return (
        node.op == "call_function"
        and isinstance(node.target, torch._ops.OpOverload)
        and node.target.overloadpacket in _MM
    )


def _matmul_inputs(node):
    """Only the two multiplied operands; addmm's self/bias is not a prologue."""
    bindings = {
        argument.name: value
        for argument, value in zip(node.target._schema.arguments, node.args)
    }
    bindings.update(node.kwargs)
    names = (
        ("mat1", "mat2")
        if node.target.overloadpacket is torch.ops.aten.addmm
        else ("self", "mat2")
    )
    return [
        bindings[name] for name in names if isinstance(bindings[name], torch.fx.Node)
    ]


_PERMUTES = {torch.ops.aten.permute, torch.ops.aten.transpose, torch.ops.aten.t}


def _reduced_run(node):
    """Size of the trailing run ``node`` reduces, or None if it is not trailing."""
    source = node.args[0] if node.args else None
    value = source.meta.get("val") if isinstance(source, torch.fx.Node) else None
    if not isinstance(value, torch.Tensor) or value.dim() == 0:
        return None, None
    dims = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
    if isinstance(dims, int):
        dims = [dims]
    if not isinstance(dims, (list, tuple)) or not dims or not all(
        isinstance(d, int) for d in dims
    ):
        return None, value
    rank = value.dim()
    dims = sorted(d % rank for d in dims)
    if dims != list(range(rank - len(dims), rank)):
        return None, value
    run = 1
    for d in dims:
        run *= value.shape[d]
    return run, value


def _k_row_reduction(reduction, mm, partition):
    """``reduction`` normalizes rows of the MM's A operand along K.

    It reduces trailing dims of an operand-sized tensor in runs that tile K
    (a whole row, or per-head groups), feeds A without a permute in between,
    as in RMSNorm/LayerNorm(x) @ W. The kernel can reduce each A row (or
    head) it loads, or fold the scale past the MM: (x / rms) @ W = (x @ W) / rms.
    """
    a = mm.args[1] if mm.target.overloadpacket is torch.ops.aten.addmm else mm.args[0]
    if not isinstance(a, torch.fx.Node) or a not in partition:
        return False
    operand = a.meta.get("val")
    run, value = _reduced_run(reduction)
    if run is None or not isinstance(operand, torch.Tensor) or operand.dim() < 2:
        return False
    k = operand.shape[-1]
    if not run or k % run or value.numel() != operand.numel():
        return False
    # Nodes between the reduction and A: reachable from both directions.
    downstream, stack = set(), [reduction]
    while stack:
        for user in stack.pop().users:
            if user in partition and user not in downstream:
                downstream.add(user)
                stack.append(user)
    if a is not reduction and a not in downstream:
        return False
    upstream, stack = {a}, [a]
    while stack:
        for arg in stack.pop().all_input_nodes:
            if arg in downstream and arg not in upstream:
                upstream.add(arg)
                stack.append(arg)
    return not any(
        isinstance(n.target, torch._ops.OpOverload) and n.target.overloadpacket in _PERMUTES
        for n in upstream
    )


def _storage_keys(value):
    if isinstance(value, torch.Tensor):
        return {value.untyped_storage()._cdata}
    if isinstance(value, (tuple, list)):
        return set().union(*(_storage_keys(v) for v in value))
    if isinstance(value, dict):
        return _storage_keys(list(value.values()))
    return set()


def _written_keys(node):
    schema = getattr(node.target, "_schema", None)
    keys = set()
    if schema is not None:
        from torch.utils._pytree import tree_leaves

        for i, argument in enumerate(schema.arguments):
            if argument.alias_info is None or not argument.alias_info.is_write:
                continue
            value = (
                node.args[i] if i < len(node.args) else node.kwargs.get(argument.name)
            )
            for item in tree_leaves(value):
                if isinstance(item, torch.fx.Node):
                    keys.update(_storage_keys(item.meta.get("val")))
    return keys


def _safe(nodes, region, order):
    first, last = min(order[n] for n in region), max(order[n] for n in region)
    pending = [
        u for n in region for u in n.users if u not in region and order[u] <= last
    ]
    seen = set()
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        if any(user in region for user in node.users):
            return False
        pending.extend(u for u in node.users if order[u] <= last)
    touched, written = set(), set()
    for node in region:
        for item in [node, *node.all_input_nodes]:
            touched.update(_storage_keys(item.meta.get("val")))
        written.update(_written_keys(node))
    for node in nodes[first : last + 1]:
        if node in region:
            continue
        if _written_keys(node) & touched:
            return False
        if any(
            _storage_keys(arg.meta.get("val")) & written for arg in node.all_input_nodes
        ):
            return False
    return True


def absorb_matmuls(gm, *, excluded=()):
    """Return (original partition + one MM, direction) in deterministic order."""
    from capture_hook import TRANSPARENT_OPS, get_fusion_partitions

    nodes = list(gm.graph.nodes)
    order = {node: i for i, node in enumerate(nodes)}
    partitions = [
        sorted(p, key=order.__getitem__)
        for p in get_fusion_partitions(gm, excluded=excluded)
    ]
    partitions.sort(key=lambda p: order[p[0]])
    membership = {node: i for i, p in enumerate(partitions) for node in p}
    vertices = dict(membership)
    next_vertex = len(partitions)
    for node in nodes:
        if node not in vertices:
            vertices[node] = next_vertex
            next_vertex += 1
    edges = {i: set() for i in range(next_vertex)}
    for node in nodes:
        for arg in node.all_input_nodes:
            if vertices[arg] != vertices[node]:
                edges[vertices[arg]].add(vertices[node])
    graph = _PartitionGraph(edges)
    excluded = set(excluded)
    matmuls = [n for n in nodes if _is_mm(n) and n not in excluded]
    # A prologue is computed while the MM loads its operand. Pointwise work and
    # gathers qualify, and so does ONE reduction that normalizes rows of A
    # along K (RMSNorm/LayerNorm before a linear):
    #   eligible:   a[i,k] = table[ids[i], k] * 2                (gather + pointwise)
    #   eligible:   a[i,k] = x[i,k] * rsqrt(mean_k(x[i,:]^2))    (one K-row reduction)
    #   ineligible: two reductions, a scan, or a reduction over M (a column sum)
    # The test is not "every op has Tag.pointwise": the tag rejects gathers,
    # constants and views that Inductor lowers to Pointwise.
    reductions, prologue_eligible = [], []
    for partition in partitions:
        real = [
            n
            for n in partition
            if n.op == "call_function" and n.target not in TRANSPARENT_OPS
        ]
        found = [n for n in real if is_reduction_or_scan(n)]
        reductions.append(found)
        prologue_eligible.append(
            bool(real) and len(found) <= 1 and all(map(is_reduction, found))
        )
    candidates = {"prologue": defaultdict(list), "epilogue": defaultdict(list)}
    for mm in matmuls:
        if mm in membership:
            raise ValueError("normal BB partition unexpectedly contains a target MM")
        front = {membership[n] for n in _matmul_inputs(mm) if n in membership}
        back = {membership[n] for n in mm.users if n in membership}
        for pid in front:
            if prologue_eligible[pid] and all(
                _k_row_reduction(r, mm, set(partitions[pid])) for r in reductions[pid]
            ):
                candidates["prologue"][pid].append(mm)
        for pid in back:
            candidates["epilogue"][pid].append(mm)
    used_partitions, used_mms, regions = set(), set(), []
    # Epilogues are assigned first. In relu(x @ w1) @ w2 the relu partition both
    # consumes mm (epilogue candidate) and feeds mm_1 (prologue candidate):
    #   prologue first: prologue [relu, mm_1]; mm gets no epilogue
    #   epilogue first: epilogue [mm, relu]; mm_1 can still take a prologue from
    #                   another unused eligible partition, e.g. sin(w2)
    # Epilogue fusion is the more common case, so it gets the shared partition.
    for category in ("epilogue", "prologue"):
        for pid in sorted(candidates[category]):
            if pid in used_partitions:
                continue
            for mm in sorted(candidates[category][pid], key=order.__getitem__):
                if mm in used_mms:
                    continue
                region = set(partitions[pid]) | {mm}
                if not _safe(nodes, region, order):
                    continue
                source, target = (
                    (pid, vertices[mm])
                    if category == "prologue"
                    else (vertices[mm], pid)
                )
                if not graph.permits(source, target):
                    continue
                graph.absorb(source, target, pid)
                used_partitions.add(pid)
                used_mms.add(mm)
                regions.append((sorted(region, key=order.__getitem__), category))
                break
    return regions
