"""execution_hash, the ordered FX identity of prologue/epilogue repros."""

from __future__ import annotations

import torch
from torch import fx

import mm_replay as replay


def graph(*, literal=1.0, reverse=False, as_list=False):
    g = fx.Graph()
    x = g.placeholder("x")
    a = g.call_function(torch.ops.aten.add.Tensor, (x, literal))
    b = g.call_function(torch.ops.aten.mul.Tensor, (x, 3.0))
    values = [b, a] if reverse else [a, b]
    g.output(values if as_list else tuple(values))
    return fx.GraphModule({}, g)


def test_execution_hash_retains_literals_output_order_and_tree_type():
    original = graph()
    key = replay.execution_hash(original)
    assert key == replay.execution_hash(graph())
    assert key != replay.execution_hash(graph(literal=2.0))
    assert key != replay.execution_hash(graph(reverse=True))
    assert key != replay.execution_hash(graph(as_list=True))
    x = torch.ones(2)
    first = original(x)
    reversed_ = graph(reverse=True)(x)
    assert isinstance(first, tuple) and isinstance(graph(as_list=True)(x), list)
    assert not torch.equal(first[0], first[1])
    torch.testing.assert_close(first[0], reversed_[1])
    torch.testing.assert_close(first[1], reversed_[0])
    assert replay.execution_hash(graph(literal=0.0)) != replay.execution_hash(
        graph(literal=-0.0)
    )
