import sys
from pathlib import Path

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import capture_hook
import jagged_inputs
from full_graph_harness import graph_constraints_from_gm, make_inputs_from_full_graph_specs
from input_codec import compact_from_spec, spec_from_compact
from jagged_inputs import (
    JAGGED_RULES,
    LENGTHS_KIND,
    OFFSETS_KIND,
    jagged_input_specs,
    make_jagged_tensor,
    make_lengths,
    register_lengths_distribution,
)
from mask_rules import register_mask_ops


@torch.library.custom_op("bbjag::to_padded", mutates_args=())
def _to_padded(values: torch.Tensor, offsets: torch.Tensor, max_len: int) -> torch.Tensor:
    return values.new_zeros(offsets.numel() - 1, max_len)


@_to_padded.register_fake
def _(values, offsets, max_len):
    return values.new_empty(offsets.shape[0] - 1, max_len)


register_mask_ops({"jagged": [{"namespace": "bbjag", "pattern": "to_padded"}]})
JAGGED_RULES[("bbjag", "to_padded")] = lambda node: [
    (node.args[1], "offsets", node.args[0].meta["val"].shape[0], node.args[2])]


def _assert_valid_offsets(offsets, total):
    assert offsets[0] == 0
    assert bool((offsets.diff() >= 0).all())
    assert offsets[-1] == total


def test_lengths_add_up_and_respect_a_feasible_cap():
    torch.manual_seed(0)
    lengths = make_lengths(64, 1000, max_len=20)
    assert lengths.sum() == 1000 and lengths.min() >= 0 and lengths.max() <= 20
    # 64 * 10 < 1000: the cap cannot hold, the total still does.
    assert make_lengths(64, 1000, max_len=10).sum() == 1000
    assert make_lengths(5, 0).tolist() == [0] * 5


def test_custom_distribution_is_used_and_checked():
    register_lengths_distribution(
        "at", lambda n, t, p: torch.tensor([t if i == p["index"] else 0 for i in range(n)]))
    assert make_lengths(4, 9, distribution={"family": "at", "index": 2}).tolist() == [0, 0, 9, 0]
    register_lengths_distribution("bad", lambda n, t, p: torch.ones(n, dtype=torch.int64))
    with pytest.raises(ValueError):
        make_lengths(4, 9, distribution="bad")


def test_offsets_start_at_zero_and_end_at_the_total():
    torch.manual_seed(0)
    gen = {"kind": OFFSETS_KIND, "total_value": 77, "max_len": None}
    offsets = make_jagged_tensor(gen, (11,), torch.int32, "cpu")
    assert offsets.dtype == torch.int32
    _assert_valid_offsets(offsets, 77)


def test_spec_alone_uses_the_captured_total_and_fake_mode_skips_values():
    from full_graph_harness import make_tensor_from_spec
    from torch._subclasses.fake_tensor import FakeTensorMode

    spec = {"kind": "tensor", "name": "o", "shape": [5], "dtype": "int32", "device": "cpu",
            "gen": {"kind": OFFSETS_KIND, "total": {"sym": "s", "value": 30}, "max_len": None,
                    "distribution": "uniform"}}
    _assert_valid_offsets(make_tensor_from_spec(spec, default_device="cpu"), 30)
    offset = make_tensor_from_spec({**spec, "storage_offset": 7}, default_device="cpu")
    assert offset.storage_offset() == 7
    _assert_valid_offsets(offset, 30)
    with FakeTensorMode():
        assert make_tensor_from_spec(spec, default_device="cpu").shape == (5,)


def _graph(fn, *inputs):
    return make_fx(fn, tracing_mode="symbolic")(*inputs)


def _jagged_nodes(gm):
    return [n for n in gm.graph.nodes if "jagged" in capture_hook.mask_reasons(n)]


def test_repeat_interleave_and_cumsum_offsets_get_valid_inputs():
    def fn(lengths, values, offsets):
        rep = torch.ops.aten.repeat_interleave.Tensor(lengths, output_size=values.shape[0])
        padded = torch.ops.bbjag.to_padded(values, offsets.to(torch.int64), 8)
        return values[rep], padded

    gm = _graph(fn, torch.tensor([2, 3, 1]), torch.randn(6), torch.tensor([0, 2, 5, 6]))
    jagged = jagged_input_specs(gm, _jagged_nodes(gm))
    assert not jagged.uncovered
    specs = jagged.specs
    assert specs["lengths_1"]["kind"] == LENGTHS_KIND
    assert specs["offsets_1"]["kind"] == OFFSETS_KIND and specs["offsets_1"]["max_len"] == 8
    assert specs["lengths_1"]["total"] == specs["offsets_1"]["total"] == {"size": ["values_1", 0], "value": 6}
    assert capture_hook.graph_mask(gm) == {"reasons": ["jagged"], "scope": "graph"}
    assert capture_hook.graph_mask(gm, jagged) is None

    payload = graph_constraints_from_gm(gm, jagged_specs=specs)
    entries = [compact_from_spec(s, include_name=True) for s in payload["inputs"]]
    input_specs = [spec_from_compact(e) for e in entries]
    for spec in input_specs:
        if spec.get("kind") == "tensor":
            spec["device"] = "cpu"
    # A different point: the totals follow the values input.
    input_specs[1]["shape"] = [40]
    input_specs[2]["shape"] = [11]
    torch.manual_seed(0)
    lengths, values, offsets = make_inputs_from_full_graph_specs(input_specs, default_device="cpu")[:3]
    assert lengths.sum() == 40 and lengths.min() >= 0
    _assert_valid_offsets(offsets, 40)
    assert offsets.diff().max() <= 8


def test_input_read_two_ways_stays_masked():
    def fn(lengths, values):
        rep = torch.ops.aten.repeat_interleave.Tensor(lengths, output_size=values.shape[0])
        return values[rep], torch.ops.bbjag.to_padded(values, lengths, 4)

    gm = _graph(fn, torch.tensor([2, 0, 2, 2]), torch.randn(6))
    jagged = jagged_input_specs(gm, _jagged_nodes(gm))
    assert not jagged.specs and not jagged.covered
    assert capture_hook.graph_mask(gm, jagged) == {"reasons": ["jagged"], "scope": "graph"}


def test_computed_offsets_stay_masked():
    def fn(values, x):
        offsets = torch.cumsum(x, 0)
        return torch.ops.bbjag.to_padded(values, offsets, 4)

    gm = _graph(fn, torch.randn(6), torch.tensor([0, 1, 2]))
    jagged = jagged_input_specs(gm, _jagged_nodes(gm))
    assert not jagged.specs and jagged.uncovered


def test_unknown_jagged_op_stays_masked():
    nodes = [object()]
    jagged = jagged_input_specs(_graph(lambda x: x + 1, torch.ones(2)), nodes)
    assert jagged.uncovered == set(nodes)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
