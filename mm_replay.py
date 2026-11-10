"""Ordered FX identity for prologue/epilogue repros."""

from __future__ import annotations

import json

import torch


def execution_hash(gm):
    """Conservative ordered FX identity: literals, input positions and output tree."""
    import hashlib
    import torch.fx as fx

    ids = {node: i for i, node in enumerate(gm.graph.nodes)}

    def encode(value):
        if isinstance(value, fx.Node):
            return ["node", ids[value]]
        if value is None:
            return ["none"]
        if type(value) in (bool, int, str):
            return [type(value).__name__, value]
        if isinstance(value, float):
            return ["float", value.hex()]
        if isinstance(value, (tuple, list)):
            return [type(value).__name__, [encode(v) for v in value]]
        if isinstance(value, dict):
            return ["dict", [[encode(k), encode(v)] for k, v in value.items()]]
        if isinstance(value, slice):
            return [
                "slice",
                encode(value.start),
                encode(value.stop),
                encode(value.step),
            ]
        if isinstance(
            value, (torch.dtype, torch.device, torch.layout, torch.memory_format)
        ):
            return [type(value).__name__, str(value)]
        if isinstance(value, torch.SymInt) and value.node.has_hint():
            return ["symint", int(value.node.hint)]
        raise TypeError(f"Unsupported FX literal: {type(value).__name__}")

    nodes = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            nodes.append(["placeholder"])
        else:
            target = (
                str(node.target)
                if isinstance(node.target, (str, torch._ops.OpOverload))
                else (node.target.__module__ + "." + node.target.__qualname__)
            )
            nodes.append([node.op, target, encode(node.args), encode(node.kwargs)])
    return hashlib.sha256(
        json.dumps(nodes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
