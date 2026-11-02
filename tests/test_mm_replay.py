"""Small CPU regressions for the single-MM replay adapter (no corpus/GPU work)."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import fx
from torch._subclasses.fake_tensor import FakeTensorMode

import mm_replay as replay
from input_codec import compact_from_spec


def tensor(
    shape, *, dtype="int64", stride=None, offset=0, alias=None, gen=None, data=None
):
    spec = {
        "kind": "tensor",
        "shape": list(shape),
        "dtype": dtype,
        "device": "cpu",
        "storage_offset": offset,
    }
    if stride is not None:
        spec["stride"] = list(stride)
    if alias is not None:
        spec["alias_group"] = alias
    if gen is not None:
        spec["gen"] = gen
    if data is not None:
        spec.update(exact=True, data=data)
    return spec


def config(*specs, nbytes=128, compact=False):
    return {
        "inputs": [compact_from_spec(s) for s in specs] if compact else list(specs),
        "alias_group_nbytes": [nbytes],
    }


@pytest.mark.parametrize("compact", [False, True])
def test_bounded_alias_indices_replay_mm_indexing(compact):
    point = config(
        tensor([3], alias=0, gen={"kind": "index", "low": 0, "high": 2}),
        tensor([3], alias=0, offset=1, gen={"kind": "randint", "low": 0, "high": 2}),
        nbytes=96,
        compact=compact,
    )
    before = copy.deepcopy(point)
    i, j = replay.make_inputs_from_config(point)
    assert point == before
    assert torch._C._is_alias_of(i, j)
    assert i.untyped_storage().nbytes() == 96
    assert (i.storage_offset(), j.storage_offset()) == (0, 1)
    torch.testing.assert_close(i[1:], j[:-1])
    assert bool(((i >= 0) & (i < 2)).all())
    assert bool(((j >= 0) & (j < 2)).all())
    m = torch.mm(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.eye(2))
    actual = m[i] + m[j]
    torch.testing.assert_close(
        actual, torch.stack([m[int(a)] + m[int(b)] for a, b in zip(i, j)])
    )


@pytest.mark.parametrize(
    "gen",
    [
        None,
        {"kind": "index", "low": 2, "high": 3},
        {"kind": "constant", "value": 2},
        {"kind": "positive", "minimum": 0.1},
    ],
)
@pytest.mark.parametrize("compact", [False, True])
def test_nonaliased_offset_contexts_keep_stride_device_and_values(gen, compact):
    a, b = replay.make_inputs_from_config(
        config(
            tensor([2, 2], dtype="float32", stride=[1, 2], gen=gen),
            tensor([2, 2], dtype="float32", stride=[1, 2], offset=4, gen=gen),
            compact=compact,
        )
    )
    assert not torch._C._is_alias_of(a, b)
    assert a.stride() == b.stride() == (1, 2)
    assert a.device == b.device == torch.device("cpu")
    assert (a.storage_offset(), b.storage_offset()) == (0, 4)
    if gen and gen["kind"] != "positive":
        torch.testing.assert_close(a, torch.full((2, 2), 2.0))
        torch.testing.assert_close(b, torch.full((2, 2), 2.0))
    if gen and gen["kind"] == "positive":
        assert bool((b > 0).all())


@pytest.mark.parametrize(
    "other", [{"kind": "index", "low": 2, "high": 5}, {"kind": "constant", "value": 2}]
)
def test_overlapping_bounds_and_constants_are_compatible(other):
    a, b = replay.make_inputs_from_config(
        config(
            tensor([4], alias=0, gen={"kind": "index", "low": 0, "high": 3}),
            tensor([3], alias=0, offset=1, gen=other),
        )
    )
    assert torch._C._is_alias_of(a, b)
    torch.testing.assert_close(a, torch.full((4,), 2, dtype=torch.int64))
    torch.testing.assert_close(b, torch.full((3,), 2, dtype=torch.int64))


@pytest.mark.parametrize(
    "other", [{"kind": "index", "low": 3, "high": 5}, {"kind": "constant", "value": 3}]
)
def test_conflicting_bounds_or_constants_reject(other):
    with pytest.raises(ValueError, match="overlapping|violates"):
        replay.make_inputs_from_config(
            config(
                tensor([4], alias=0, gen={"kind": "index", "low": 0, "high": 3}),
                tensor([3], alias=0, offset=1, gen=other),
            )
        )


def test_conflicting_constants_reject():
    with pytest.raises(ValueError, match="constants"):
        replay.make_inputs_from_config(
            config(
                tensor([3], alias=0, gen={"kind": "constant", "value": 1}),
                tensor([3], alias=0, offset=1, gen={"kind": "constant", "value": 2}),
            )
        )


def test_disjoint_constraints_do_not_conflict_through_an_unconstrained_view():
    a, b, whole = replay.make_inputs_from_config(
        config(
            tensor([2], alias=0, gen={"kind": "constant", "value": 1}),
            tensor([2], alias=0, offset=4, gen={"kind": "index", "low": 3, "high": 4}),
            tensor([6], alias=0),
        )
    )
    torch.testing.assert_close(a, torch.ones(2, dtype=torch.int64))
    torch.testing.assert_close(b, torch.full((2,), 3, dtype=torch.int64))
    torch.testing.assert_close(whole[:2], a)
    torch.testing.assert_close(whole[4:], b)
    assert torch._C._is_alias_of(a, b)


def test_positive_alias_constraints_keep_shared_broadcast_layout():
    a, b = replay.make_inputs_from_config(
        config(
            tensor(
                [2, 3],
                dtype="float32",
                stride=[0, 1],
                alias=0,
                gen={"kind": "positive", "minimum": 0.1},
            ),
            tensor(
                [2],
                dtype="float32",
                offset=1,
                alias=0,
                gen={"kind": "positive", "minimum": 0.5},
            ),
        )
    )
    assert a.stride() == (0, 1)
    assert torch._C._is_alias_of(a, b)
    assert bool((a >= 0.5).all()) and bool((b >= 0.5).all())
    torch.testing.assert_close(a[0, 1:], b)
    assert bool(torch.isfinite(torch.rsqrt(a)).all())


def test_positive_and_index_constraints_intersect():
    a, b = replay.make_inputs_from_config(
        config(
            tensor(
                [3], dtype="float32", alias=0, gen={"kind": "positive", "minimum": 1.5}
            ),
            tensor(
                [3],
                dtype="float32",
                alias=0,
                gen={"kind": "index", "low": 0, "high": 3},
            ),
        )
    )
    torch.testing.assert_close(a, torch.full((3,), 2.0))
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize(
    "other", [{"kind": "constant", "value": 0}, {"kind": "index", "low": 0, "high": 1}]
)
def test_positive_conflicts_reject(other):
    with pytest.raises(ValueError, match="minimum|bounds"):
        replay.make_inputs_from_config(
            config(
                tensor(
                    [3],
                    dtype="float32",
                    alias=0,
                    gen={"kind": "positive", "minimum": 0.5},
                ),
                tensor([3], dtype="float32", alias=0, gen=other),
            )
        )


def test_positive_floor_is_representable_for_half():
    (value,) = replay.make_inputs_from_config(
        config(tensor([4], dtype="float16", gen={"kind": "positive", "minimum": 0.1}))
    )
    assert all(float(x) >= 0.1 for x in value)


def test_same_layout_permutation_aliases_keep_offsets_and_logical_order():
    spec = tensor(
        [2, 4], stride=[1, 2], offset=2, alias=0, gen={"kind": "permutation", "size": 4}
    )
    a, b = replay.make_inputs_from_config(config(spec, copy.deepcopy(spec), nbytes=160))
    assert torch._C._is_alias_of(a, b)
    assert a.stride() == (1, 2) and a.storage_offset() == 2
    assert a.untyped_storage().nbytes() == 160
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(a.sort(-1).values, torch.arange(4).expand(2, 4))


@pytest.mark.parametrize("sorted_", [False, True])
def test_same_layout_runs_aliases_keep_cardinality(sorted_):
    spec = tensor(
        [6],
        offset=2,
        alias=0,
        gen={
            "kind": "runs",
            "low": 0,
            "high": 5,
            "n_unique": 2,
            "n_runs": 2 if sorted_ else 3,
            "sorted": sorted_,
        },
    )
    a, b = replay.make_inputs_from_config(config(spec, copy.deepcopy(spec)))
    assert torch._C._is_alias_of(a, b) and a.storage_offset() == 2
    torch.testing.assert_close(a, b)
    assert a.unique().numel() == 2
    assert 1 + int((a[1:] != a[:-1]).sum()) == spec["gen"]["n_runs"]
    if sorted_:
        assert bool((a[1:] >= a[:-1]).all())


@pytest.mark.parametrize("change", ["layout", "simple", "exact", "overlap"])
def test_unsupported_complex_alias_combinations_reject(change):
    spec = tensor([4], alias=0, gen={"kind": "permutation", "size": 4})
    other = copy.deepcopy(spec)
    if change == "layout":
        other["storage_offset"] = 1
    elif change == "simple":
        other["gen"] = {"kind": "index", "low": 0, "high": 4}
    elif change == "exact":
        other.update(exact=True, data=[0, 0, 0, 0])
    else:
        spec["stride"] = other["stride"] = [0]
    with pytest.raises(ValueError, match="complex"):
        replay.make_inputs_from_config(config(spec, other))


@pytest.mark.parametrize(
    "gen",
    [
        {"kind": "permutation", "size": 3},
        {"kind": "runs", "low": 0, "high": 2, "n_unique": 2, "n_runs": 1},
        {"kind": "offsets", "high": 4},
    ],
)
def test_invalid_or_unsupported_generators_reject(gen):
    with pytest.raises(ValueError):
        replay.make_inputs_from_config(config(tensor([4], gen=gen)))


def test_exact_noncontiguous_offset_payload_and_aliases():
    a = tensor(
        [2, 2],
        dtype="float32",
        stride=[1, 2],
        offset=2,
        alias=0,
        data=[[1.0, 2.0], [3.0, 4.0]],
    )
    b = tensor([2], dtype="float32", stride=[2], offset=2, alias=0, data=[1.0, 2.0])
    x, y = replay.make_inputs_from_config(config(a, b, nbytes=96, compact=True))
    assert x.stride() == (1, 2) and x.storage_offset() == 2
    assert torch._C._is_alias_of(x, y)
    torch.testing.assert_close(x, torch.tensor(a["data"]))
    torch.testing.assert_close(y, x[0])
    single = replay.tensor_from_exact_spec(
        {k: v for k, v in a.items() if k != "alias_group"}
    )
    torch.testing.assert_close(single, x)
    assert single.storage_offset() == 2


@pytest.mark.parametrize("data", [[0, 3], [0, -1]])
def test_exact_payload_cannot_violate_alias_index_bounds(data):
    with pytest.raises(ValueError, match="payload violates"):
        replay.make_inputs_from_config(
            config(
                tensor([2], alias=0, gen={"kind": "index", "low": 0, "high": 2}),
                tensor([2], alias=0, data=data),
            )
        )


def test_exact_overlapping_payload_conflicts_include_signed_zero():
    with pytest.raises(ValueError, match="overlapping view"):
        replay.tensor_from_exact_spec(
            tensor([2], dtype="float32", stride=[0], data=[0.0, -0.0])
        )
    with pytest.raises(ValueError, match="exact alias"):
        replay.make_inputs_from_config(
            config(tensor([1], alias=0, data=[1]), tensor([1], alias=0, data=[2]))
        )


@pytest.mark.parametrize("dtype", ["int64", "float32"])
def test_undersized_alias_allocations_reject(dtype):
    with pytest.raises(ValueError, match="allocation"):
        replay.make_inputs_from_config(
            config(tensor([4], dtype=dtype, alias=0, offset=1), nbytes=8)
        )


def test_constrained_mixed_dtype_aliases_reject():
    with pytest.raises(ValueError, match="mixed-dtype"):
        replay.make_inputs_from_config(
            config(
                tensor(
                    [3],
                    dtype="int64",
                    alias=0,
                    gen={"kind": "index", "low": 0, "high": 2},
                ),
                tensor([3], dtype="float64", alias=0),
            )
        )


def test_integer_sampling_does_not_overflow_narrow_dtype():
    (value,) = replay.make_inputs_from_config(
        config(tensor([64], dtype="int8", gen={"kind": "index", "low": 0, "high": 300}))
    )
    assert bool((value >= 0).all())
    with pytest.raises(ValueError, match="representable"):
        replay.make_inputs_from_config(
            config(
                tensor(
                    [4], dtype="int8", gen={"kind": "index", "low": 200, "high": 300}
                )
            )
        )


def test_faketensor_keeps_alias_layout_without_data_dependent_validation():
    point = config(
        tensor([3], alias=0, offset=1, gen={"kind": "index", "low": 0, "high": 2}),
        tensor([2], alias=0, offset=2, data=[0, 1]),
    )
    with FakeTensorMode():
        a, b = replay.make_inputs_from_config(point)
        assert torch._C._is_alias_of(a, b)
        assert (a.storage_offset(), b.storage_offset()) == (1, 2)
        assert a.device == torch.device("cpu")


@pytest.mark.parametrize("compact", [False, True])
def test_shape_and_scalar_arguments_are_not_tensorized(compact):
    point = config(
        {"kind": "shape", "dims": [2, 3]},
        {"kind": "scalar", "value": 0.25},
        {"kind": "symint", "value": 7},
        compact=compact,
    )
    assert replay.make_inputs_from_config(point) == [[2, 3], 0.25, 7]


def test_empty_exact_payload_is_checked_and_retains_offset():
    spec = tensor([0, 2], dtype="float32", stride=[2, 1], offset=3, data=[])
    value = replay.tensor_from_exact_spec(spec)
    assert value.shape == (0, 2) and value.stride() == (2, 1)
    assert value.storage_offset() == 3
    spec["data"] = [1.0]
    with pytest.raises(ValueError, match="size mismatch"):
        replay.tensor_from_exact_spec(spec)


def test_default_runs_generator_keeps_legacy_default_range():
    (value,) = replay.make_inputs_from_config(config(tensor([4], gen={"kind": "runs"})))
    torch.testing.assert_close(value, torch.zeros(4, dtype=torch.int64))


@pytest.mark.parametrize(
    "gen",
    [
        {"kind": "index", "low": 1, "high": 2},
        {"kind": "constant", "value": 1},
        {"kind": "permutation", "size": 4},
        {"kind": "runs", "low": 1, "high": 2, "n_unique": 1, "n_runs": 1},
    ],
)
def test_nonaliased_integer_offsets_are_preserved(gen):
    (value,) = replay.make_inputs_from_config(config(tensor([4], offset=3, gen=gen)))
    assert value.storage_offset() == 3 and value.stride() == (1,)
    if gen["kind"] == "permutation":
        torch.testing.assert_close(value.sort().values, torch.arange(4))
    else:
        torch.testing.assert_close(value, torch.ones(4, dtype=torch.int64))


def test_unconstrained_mixed_dtype_aliases_share_finite_zero_storage():
    a, b = replay.make_inputs_from_config(
        config(
            tensor([3], dtype="float64", alias=0),
            tensor([3], dtype="int64", alias=0),
        )
    )
    assert torch._C._is_alias_of(a, b)
    assert bool(torch.isfinite(a).all())
    torch.testing.assert_close(a, torch.zeros(3, dtype=torch.float64))
    torch.testing.assert_close(b, torch.zeros(3, dtype=torch.int64))


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


def test_benchmark_patch_restores_legacy_generator_on_exception(monkeypatch):
    old = replay._harness.make_inputs_from_config
    cls, factory = object(), object()

    def fail(*args):
        assert args == ("memory.py", cls, factory)
        assert replay._harness.make_inputs_from_config is replay.make_inputs_from_config
        raise RuntimeError("probe stops before any benchmark work")

    monkeypatch.setattr(replay._harness, "benchmark_repro", fail)
    with pytest.raises(RuntimeError, match="probe stops"):
        replay.benchmark_repro_safe("memory.py", cls, factory)
    assert replay._harness.make_inputs_from_config is old
