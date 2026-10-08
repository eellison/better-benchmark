"""fusion_partitioner.PrunedCapabilityBasedPartitioner must propose exactly the
partitions torch's CapabilityBasedPartitioner does (same nodes, same order)."""
import inspect
import operator
import random
import sys
from pathlib import Path

import pytest
import torch.fx as fx
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import create_op_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion_partitioner import PrunedCapabilityBasedPartitioner  # noqa: E402

HAS_SKIP_HORIZONTAL = (
    "skip_horizontal_fusion"
    in inspect.signature(CapabilityBasedPartitioner.__init__).parameters
)


def _random_dag(seed: int, n: int = 60, unsupported_frac: float = 0.25):
    """n binary nodes, each reading 1-3 random earlier values, so the graph
    has fan-out, reconvergent paths and unsupported nodes between supported
    ones: the shapes where the cycle check matters."""
    rng = random.Random(seed)
    g = fx.Graph()
    values = [g.placeholder(f"x{i}") for i in range(3)]
    unsupported = set()
    for i in range(n):
        args = rng.sample(values, k=min(len(values), rng.randint(1, 3)))
        node = args[0]
        for a in args[1:]:
            node = g.call_function(operator.add, (node, a))
        node = g.call_function(operator.mul, (node, args[-1]))
        if rng.random() < unsupported_frac:
            unsupported.add(node.name)
        values.append(node)
    g.output(tuple(values[-4:]))
    return fx.GraphModule({}, g), unsupported


def _diamond_through_unsupported():
    """a -> unsupported u -> c and a -> c directly: merging a with c would
    make a cycle through u, so they must stay apart."""
    g = fx.Graph()
    x = g.placeholder("x")
    a = g.call_function(operator.add, (x, x))
    u = g.call_function(operator.mul, (a, a))
    b = g.call_function(operator.add, (a, x))
    c = g.call_function(operator.add, (u, b))
    d = g.call_function(operator.mul, (c, b))
    g.output(d)
    return fx.GraphModule({}, g), {u.name}



def _cycle_behind_second_user():
    """Like the diamond, but a's FIRST user u1 is a dead end and the cycle
    runs through its second user u2, so every user has to be checked, not
    just the first."""
    g = fx.Graph()
    x = g.placeholder("x")
    a = g.call_function(operator.add, (x, x))
    u1 = g.call_function(operator.mul, (a, x))
    u2 = g.call_function(operator.mul, (a, a))
    c = g.call_function(operator.add, (u2, a))
    d = g.call_function(operator.mul, (c, c))
    g.output((u1, d))
    return fx.GraphModule({}, g), {u1.name, u2.name}

def _partitions(cls, gm, unsupported, **kwargs):
    support = create_op_support(
        lambda _mods, node: node.op == "call_function"
        and node.name not in unsupported
    )
    return [list(p.nodes) for p in cls(gm, support, **kwargs).propose_partitions()]


OPTIONS = [
    {"allows_single_node_partition": True},
    {"allows_single_node_partition": False},
]
if HAS_SKIP_HORIZONTAL:
    OPTIONS += [
        {"allows_single_node_partition": True, "skip_horizontal_fusion": True},
        {"allows_single_node_partition": False, "skip_horizontal_fusion": True},
    ]

GRAPHS = [
    ("diamond", _diamond_through_unsupported),
    ("second_user", _cycle_behind_second_user),
] + [
    (f"dag{n}_{frac}_{seed}", lambda n=n, frac=frac, seed=seed: _random_dag(seed, n, frac))
    for n in (40, 80)
    for frac in (0.1, 0.3, 0.6)
    for seed in range(30)
]


@pytest.mark.parametrize("options", OPTIONS, ids=lambda o: ",".join(sorted(o)))
def test_matches_torch_partitioner(options):
    for name, build in GRAPHS:
        gm, unsupported = build()
        expected = _partitions(CapabilityBasedPartitioner, gm, unsupported, **options)
        got = _partitions(PrunedCapabilityBasedPartitioner, gm, unsupported, **options)
        assert got == expected, name


@pytest.mark.parametrize("build", [_diamond_through_unsupported, _cycle_behind_second_user])
def test_cycle_keeps_partitions_apart(build):
    gm, unsupported = build()
    parts = _partitions(
        PrunedCapabilityBasedPartitioner, gm, unsupported,
        allows_single_node_partition=True,
    )
    assert len(parts) > 1
