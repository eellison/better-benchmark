"""Give every aten._grouped_mm its whole normal BB producer and consumer partitions.

A grouped MM (MoE experts: rows of A sorted by expert, one B matrix per expert,
``offs`` the end row of each expert's group) takes

- **prologue side**: every normal BB partition that produces one of its two
  multiplied operands (``mat_a``, ``mat_b``), with no reduction/scan
  restriction: gpt-oss's routing partition (topk, sort, the row gather and the
  sentinel mask) is a prologue as a whole;
- **epilogue side**: every normal BB partition that consumes its result.

Unpartitioned views between an operand or the result and such a partition join
the region. ``offs`` (and a bias) is not a multiplied operand: its producer
stays outside and ``offs`` is a region input.

Unlike mm_partition_absorption, partitions are not consumed. In gpt-oss the
partition between the gate/up and the down projection (bias, clamp, swiglu) is
the epilogue of the first grouped MM and the prologue of the second, so it is
in both regions:

    region 1: [routing partition] -> _grouped_mm   -> [bias + swiglu partition]
    region 2: [bias + swiglu]     -> _grouped_mm_1 -> [bias + unpermute + residual + norm]

mm/addmm/bmm absorption (mm_partition_absorption) is separate and unchanged.
"""

from __future__ import annotations

import torch

from matmul_partitioner import VIEW_NAMES, op_name

GROUPED_MM_NAMES = frozenset({"aten::_grouped_mm"})


def is_grouped_mm(node):
    return node.op == "call_function" and op_name(node) in GROUPED_MM_NAMES


def grouped_operands(node):
    """(mat_a, mat_b, offs) of a grouped MM; offs may be None."""
    bindings = {
        argument.name: value
        for argument, value in zip(node.target._schema.arguments, node.args)
    }
    bindings.update(node.kwargs)
    return bindings["self"], bindings["mat2"], bindings.get("offs")


def _unpartitioned_view(node, membership):
    return (
        isinstance(node, torch.fx.Node)
        and node not in membership
        and op_name(node) in VIEW_NAMES
        and bool(node.args)
        and isinstance(node.args[0], torch.fx.Node)
    )


def absorb_grouped_matmuls(gm, *, excluded=()):
    """[(region nodes, grouped MM, prologue nodes, epilogue nodes)] in graph order.

    Prologue/epilogue nodes are the absorbed partitions (and views) on each
    side. A region may share partitions with another grouped MM's region.
    """
    from capture_hook import get_fusion_partitions

    nodes = list(gm.graph.nodes)
    order = {node: i for i, node in enumerate(nodes)}
    partitions = [
        sorted(p, key=order.__getitem__)
        for p in get_fusion_partitions(gm, excluded=excluded)
    ]
    partitions.sort(key=lambda p: order[p[0]])
    membership = {node: i for i, p in enumerate(partitions) for node in p}
    excluded = set(excluded)
    results = []
    for mm in nodes:
        if not is_grouped_mm(mm) or mm in excluded:
            continue
        if mm in membership:
            raise ValueError("normal BB partition unexpectedly contains a grouped MM")
        mat_a, mat_b, _offs = grouped_operands(mm)
        prologue, epilogue = set(), set()
        # Producers of the multiplied operands, through unpartitioned views.
        for operand in (mat_a, mat_b):
            path, value = [], operand
            while _unpartitioned_view(value, membership):
                path.append(value)
                value = value.args[0]
            if isinstance(value, torch.fx.Node) and value in membership:
                prologue.update(path)
                prologue.update(partitions[membership[value]])
        # Consumers of the result, through unpartitioned views.
        stack = [(user, []) for user in mm.users]
        while stack:
            user, path = stack.pop()
            if user in membership:
                epilogue.update(path)
                epilogue.update(partitions[membership[user]])
            elif _unpartitioned_view(user, membership):
                stack.extend((u, path + [user]) for u in user.users)
        if prologue & epilogue:
            raise ValueError("a partition both produces and consumes a grouped MM")
        region = sorted(prologue | epilogue | {mm}, key=order.__getitem__)
        results.append((
            region,
            mm,
            sorted(prologue, key=order.__getitem__),
            sorted(epilogue, key=order.__getitem__),
        ))
    return results
