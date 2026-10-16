"""Integer index inputs keep the structure of the real values (sorted, runs)."""
import torch

import capture_hook
from input_codec import compact_from_spec, spec_from_compact
from repro_harness import make_inputs_from_config


def _runs(t):
    return int((t[1:] != t[:-1]).sum()) + 1


def test_observed_stats_record_sorted_runs():
    real = torch.arange(32).repeat_interleave(8)  # 32 ids, each in one run of 8
    stats = capture_hook.compute_observed_stats(real)
    assert stats == {"min": 0, "max": 31, "n_unique": 32, "numel": 256,
                     "sorted": True, "n_runs": 32}


def test_structured_index_gets_runs_gen():
    real = torch.arange(32).repeat_interleave(8)
    gen = capture_hook.index_gen_from_observed(
        64, capture_hook.compute_observed_stats(real))
    assert gen == {"kind": "runs", "low": 0, "high": 32, "n_unique": 32,
                   "n_runs": 32, "sorted": True}


def test_unstructured_index_keeps_uniform_gen():
    torch.manual_seed(0)
    real = torch.randint(0, 64, (256,))
    stats = capture_hook.compute_observed_stats(real)
    assert capture_hook.index_gen_from_observed(64, stats) == {
        "kind": "index", "low": 0, "high": 64}
    assert capture_hook.index_gen_from_observed(64, None)["kind"] == "index"


def test_runs_gen_round_trips_through_codec():
    spec = {"kind": "tensor", "shape": [256], "dtype": "int64",
            "gen": {"kind": "runs", "low": 3, "high": 40, "n_unique": 32,
                    "n_runs": 32, "sorted": True}}
    assert spec_from_compact(compact_from_spec(spec))["gen"] == spec["gen"]


def _make(gen, shape=(3072,)):
    cfg = {"inputs": [{"kind": "tensor", "shape": list(shape), "dtype": "int64",
                       "device": "cpu", "gen": gen}]}
    return make_inputs_from_config(cfg)[0]


def test_sorted_runs_regenerate_request_candidate_map():
    t = _make({"kind": "runs", "low": 0, "high": 128, "n_unique": 128,
               "n_runs": 128, "sorted": True})
    assert bool((t[1:] >= t[:-1]).all())
    assert t.unique().numel() == 128 and _runs(t) == 128
    assert int(t.min()) >= 0 and int(t.max()) < 128


def test_unsorted_runs_keep_run_count():
    t = _make({"kind": "runs", "low": 10, "high": 20, "n_unique": 4,
               "n_runs": 64, "sorted": False}, shape=(16, 32))
    flat = t.reshape(-1)
    assert _runs(flat) == 64
    assert flat.unique().numel() == 4
    assert int(flat.min()) >= 10 and int(flat.max()) < 20


def test_unsorted_runs_keep_unique_count():
    # Shuffled request map: every id in exactly one run.
    for seed in range(3):
        torch.manual_seed(seed)
        t = _make({"kind": "runs", "low": 0, "high": 1000, "n_unique": 1000,
                   "n_runs": 1000, "sorted": False}, shape=(8000,))
        assert t.unique().numel() == 1000 and _runs(t) == 1000


def _sorted_ok(t):
    flat = t.reshape(-1)
    return bool((flat[1:] >= flat[:-1]).all())


def test_runs_gen_supports_unsigned_dtype():
    t = make_inputs_from_config({"inputs": [{
        "kind": "tensor", "shape": [100], "dtype": "uint8", "device": "cpu",
        "gen": {"kind": "runs", "low": 0, "high": 10, "n_unique": 10,
                "n_runs": 10, "sorted": True}}]})[0]
    assert t.dtype == torch.uint8
    assert _sorted_ok(t) and _runs(t) == 10


def test_runs_gen_clamps_range_to_dtype():
    t = make_inputs_from_config({"inputs": [{
        "kind": "tensor", "shape": [512], "dtype": "int8", "device": "cpu",
        "gen": {"kind": "runs", "low": 0, "high": 1000, "n_unique": 64,
                "n_runs": 64, "sorted": True}}]})[0]
    assert int(t.min()) >= 0 and int(t.max()) <= 127 and _sorted_ok(t)


def test_runs_structure_preserved_for_non_contiguous_layout():
    # Logical [128, 24] index sorted row-major, stored column-major.
    real = torch.arange(128).repeat_interleave(24).reshape(128, 24)
    stats = capture_hook.compute_observed_stats(real.t().contiguous().t())
    gen = capture_hook.index_gen_from_observed(128, stats)
    assert gen["kind"] == "runs"
    t = make_inputs_from_config({"inputs": [{
        "kind": "tensor", "shape": [128, 24], "stride": [1, 128],
        "dtype": "int64", "device": "cpu", "gen": gen}]})[0]
    assert t.stride() == (1, 128)
    assert _sorted_ok(t) and _runs(t.reshape(-1)) == 128


def test_runs_structure_preserved_for_expanded_layout():
    real = torch.arange(128).view(128, 1).expand(128, 24)
    gen = capture_hook.index_gen_from_observed(
        128, capture_hook.compute_observed_stats(real))
    t = make_inputs_from_config({"inputs": [{
        "kind": "tensor", "shape": [128, 24], "stride": [1, 0],
        "dtype": "int64", "device": "cpu", "gen": gen}]})[0]
    assert t.stride() == (1, 0)
    assert _sorted_ok(t) and _runs(t.reshape(-1)) == 128


def test_runs_structure_preserved_for_expanded_layout_with_offset():
    import full_graph_harness as fgh
    gen = {"kind": "runs", "low": 0, "high": 64, "n_unique": 64, "n_runs": 64,
           "sorted": True}
    t = fgh.make_tensor_from_spec({
        "kind": "tensor", "shape": [64, 8], "stride": [1, 0], "dtype": "int64",
        "device": "cpu", "storage_offset": 3, "gen": gen, "generator": gen})
    assert t.stride() == (1, 0) and t.storage_offset() == 3
    assert _sorted_ok(t) and _runs(t.reshape(-1)) == 64


def test_sorted_without_repeats_keeps_uniform_gen():
    # Position ids (arange) and single-element indices carry no run
    # structure worth reproducing.
    for real in (torch.arange(512), torch.tensor([7]), torch.tensor(7)):
        stats = capture_hook.compute_observed_stats(real)
        assert capture_hook.index_gen_from_observed(2048, stats)["kind"] == "index"


def test_uniform_ids_over_few_values_keep_uniform_gen():
    # Uniform ids over k values average k/(k-1) per run (2 for binary ids),
    # so a threshold at that mean flips between recaptures.
    for k in (2, 3, 5):
        for seed in range(200):
            g = torch.Generator().manual_seed(seed)
            real = torch.randint(0, k, (4096,), generator=g)
            stats = capture_hook.compute_observed_stats(real)
            assert capture_hook.index_gen_from_observed(k, stats)["kind"] == "index", (k, seed)


def test_long_runs_over_few_values_get_runs_gen():
    real = torch.tensor([0, 1] * 64).repeat_interleave(8)  # 128 runs of 8
    gen = capture_hook.index_gen_from_observed(
        2, capture_hook.compute_observed_stats(real))
    assert gen["kind"] == "runs" and gen["n_runs"] == 128
    constant = capture_hook.compute_observed_stats(torch.full((64,), 3))
    assert capture_hook.index_gen_from_observed(8, constant)["kind"] == "runs"


def test_out_of_bound_observed_values_keep_uniform_gen():
    stats = capture_hook.compute_observed_stats(torch.tensor([-3, -3, -2, -2]))
    assert capture_hook.index_gen_from_observed(10, stats)["kind"] == "index"


def test_partially_out_of_bound_observed_values_clip_runs_range():
    stats = capture_hook.compute_observed_stats(
        torch.tensor([-1] * 8 + [5] * 8))
    gen = capture_hook.index_gen_from_observed(10, stats)
    assert gen["kind"] == "runs" and (gen["low"], gen["high"]) == (0, 6)


def test_legacy_text_signature_evaluates_runs():
    from repro_harness import _make_shape_eval_ns, parse_shapes_config
    gen = {"kind": "runs", "low": 3, "high": 40, "n_unique": 32, "n_runs": 32,
           "sorted": True}
    sig = "T([256], i64, gen=Runs(40, low=3, unique=32, runs=32, sorted=True))"
    assert eval(sig, _make_shape_eval_ns())["gen"] == gen
    (t,) = parse_shapes_config(f"({sig},)")
    assert _sorted_ok(t) and _runs(t) == 32
    assert int(t.min()) >= 3 and int(t.max()) < 40


def test_render_shows_runs_low_bound():
    from input_codec import render_T
    s = render_T({"kind": "tensor", "shape": [256], "dtype": "int64",
                  "gen": {"kind": "runs", "low": 3, "high": 40, "n_unique": 32,
                          "n_runs": 32, "sorted": True}})
    assert "low=3" in s and "Runs(40" in s


def test_full_graph_sidecar_uses_runs_for_structured_observed_input():
    import full_graph_harness as fgh
    real = torch.arange(32).repeat_interleave(8)
    gm = torch.fx.symbolic_trace(lambda idx, table: table[idx])
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.passes.fake_tensor_prop import FakeTensorProp
    mode = FakeTensorMode()
    FakeTensorProp(gm, mode).propagate(
        mode.from_tensor(real), mode.from_tensor(torch.randn(64, 4)))
    meta = fgh.graph_constraints_from_gm(
        gm, observed_stats={"idx": capture_hook.compute_observed_stats(real)})
    idx_spec = next(s for s in meta["inputs"] if s.get("name") == "idx")
    assert idx_spec["gen"]["kind"] == "runs"
    t = fgh.make_tensor_from_spec({**idx_spec, "device": "cpu"})
    assert _sorted_ok(t) and _runs(t) == 32


def test_runs_sidecar_input_counts_as_graph_inferred(tmp_path):
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from backfill_integer_constraints import (
        _has_unsafe_integer_inputs, update_sidecar_with_constraints)
    from bench_parallel import _is_graph_inferred_integer_input
    gen = {"kind": "runs", "low": 0, "high": 32, "n_unique": 32, "n_runs": 32,
           "sorted": True}
    spec = {"kind": "tensor", "name": "idx", "shape": [256], "dtype": "int64",
            "gen": gen, "generator": gen, "constraint_source": "graph_inference"}
    assert _is_graph_inferred_integer_input(spec)
    assert not _has_unsafe_integer_inputs([spec], {"inputs": [spec]})

    graph = tmp_path / "g.py"
    graph.write_text("")
    graph.with_suffix(".meta.json").write_text(json.dumps({"inputs": [spec]}))
    assert update_sidecar_with_constraints(graph, {"idx": 64}, {}) == (0, 1)
    sidecar = json.loads(graph.with_suffix(".meta.json").read_text())
    assert sidecar["inputs"][0]["gen"] == gen


_IDX = ["index", 0, 64]
_RUNS_A = ["runs", 0, 32, 32, 32, 1]
_RUNS_B = ["runs", 0, 40, 40, 40, 1]


def _merge(tmp_path, *gens):
    import json
    from merge_captures import _write_shapes_json
    for i, gen in enumerate(gens):
        _write_shapes_json(tmp_path, "abcd1234", "sig", f"m/infer/model{i}",
                           occurrences=1, inputs=[[[256], "i64", {"gen": gen}]])
    point = json.loads((tmp_path / "shapes.json").read_text())["points"][0]
    return point["inputs"][0][2]["gen"]


def test_merge_upgrades_uniform_index_to_runs(tmp_path):
    assert _merge(tmp_path, _IDX, _RUNS_A) == _RUNS_A


def test_merge_keeps_runs_over_later_uniform_index(tmp_path):
    assert _merge(tmp_path, _RUNS_A, _IDX) == _RUNS_A


def test_merge_picks_smallest_index_bound(tmp_path):
    for i, (a, b) in enumerate([(64, 100), (640, 64), (2, 1)]):
        want = ["index", 0, min(a, b)]
        for j, order in enumerate([(a, b), (b, a)]):
            path = tmp_path / f"{i}{j}"
            path.mkdir()
            assert _merge(path, *(["index", 0, h] for h in order)) == want


def test_merge_choice_between_runs_specs_is_order_independent(tmp_path):
    (tmp_path / "ab").mkdir()
    (tmp_path / "ba").mkdir()
    assert (_merge(tmp_path / "ab", _RUNS_A, _RUNS_B)
            == _merge(tmp_path / "ba", _RUNS_B, _RUNS_A))


def _merge_entries(path, entries):
    import json
    from merge_captures import _write_shapes_json
    path.mkdir()
    for i, inputs in enumerate(entries):
        _write_shapes_json(path, "abcd1234", "sig", f"m/infer/model{i}",
                           occurrences=1, inputs=inputs)
    return json.loads((path / "shapes.json").read_text())["points"][0]["inputs"]


def _one(gen, alias=False):
    opts = {"gen": gen} if gen else {}
    if alias:
        opts["alias"] = 0
        return [[[256], "i64", opts], [[256], "i64", {"alias": 0}]]
    return [[[256], "i64", opts], [[256], "i64", {}]]


def test_merge_is_order_independent_for_three_models(tmp_path):
    import itertools
    cases = {
        "runs": ([_one(_IDX), _one(_RUNS_A), _one(_RUNS_B)],
                 _one(_RUNS_A)),
        "alias": ([_one(_IDX, alias=True), _one(_RUNS_A), _one(None)],
                  _one(_RUNS_A, alias=True)),
        "perm": ([_one(_RUNS_A), _one(["perm", 64]), _one(_IDX)],
                 _one(["perm", 64])),
        "const": ([_one(_RUNS_B), _one(["const", 3]), _one(_RUNS_A)],
                  _one(["const", 3])),
    }
    for name, (models, want) in cases.items():
        for i, order in enumerate(itertools.permutations(models)):
            assert _merge_entries(tmp_path / f"{name}{i}", order) == want, (name, order)


def test_update_bounds_round_trips_runs_gen():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "validation"))
    import update_bounds
    gen = {"kind": "runs", "low": 3, "high": 40, "n_unique": 32, "n_runs": 32,
           "sorted": True}
    expr = update_bounds.format_compact_config(
        [{"kind": "tensor", "shape": [256], "dtype": "torch.int64", "gen": gen}])
    assert update_bounds.parse_compact_config(expr)[0]["gen"] == gen


def test_runs_inputs_build_under_fake_mode():
    # The capture write gate rebuilds every full-graph input under
    # FakeTensorMode; a data-dependent output shape fails it.
    import full_graph_harness as fgh
    from roundtrip_validation import fake_tensor_ctx
    for gen, shape in [
            ({"kind": "runs", "low": 0, "high": 10**6, "n_unique": 8,
              "n_runs": 8, "sorted": True}, [1000]),
            ({"kind": "runs", "low": 5, "high": 70, "n_unique": 64,
              "n_runs": 100, "sorted": False}, [10, 30])]:
        spec = {"kind": "tensor", "shape": shape, "dtype": "int64",
                "device": "cpu", "gen": gen, "generator": gen}
        with fake_tensor_ctx():
            t = fgh.make_tensor_from_spec(spec)
        assert list(t.shape) == shape


def test_sampled_unique_values_are_distinct_and_in_range():
    from repro_harness import _sample_unique
    for low, high, k in [(0, 10**6, 8), (3, 40, 37), (0, 5, 1), (7, 9, 2)]:
        v = _sample_unique(low, high, k)
        assert v.numel() == k and v.unique().numel() == k
        assert int(v.min()) >= low and int(v.max()) < high


def _capture(fn, args, tmp_path, dynamic=False):
    import json

    import torch._dynamo
    from capture_hook import install_capture_hook, uninstall_capture_hook
    out, graphs = tmp_path / "out", tmp_path / "graphs"
    torch._dynamo.reset()
    install_capture_hook(str(out), label="idx", validate=False,
                         graph_dir=str(graphs))
    try:
        with torch.no_grad():
            torch.compile(fn, dynamic=dynamic)(*args)
    finally:
        uninstall_capture_hook()
        torch._dynamo.reset()
    captured = json.loads((out / "index.json").read_text())["captured"]
    sidecars = [json.loads(p.read_text()) for p in sorted(graphs.glob("*.meta.json"))]
    return captured, sidecars


def _tensor_specs(entries):
    return [spec_from_compact(e) for e in entries
            if isinstance(e, list) and isinstance(e[0], list)]


def _int_gens(specs):
    return [s.get("gen") for s in specs if "int" in str(s.get("dtype"))]


def test_capture_gives_each_lookup_its_own_index_stats(tmp_path):
    # Region placeholders are renamed positionally by canonicalization, so
    # the uniform lookup's ids are canonically named like the full graph's
    # sorted ids; stats must follow the input, not the name.
    def f(sorted_ids, table_a, uniform_ids, table_b):
        return (table_a[sorted_ids] * 2).relu(), (table_b[uniform_ids] + 1).sigmoid()

    torch.manual_seed(0)
    captured, sidecars = _capture(f, (
        torch.arange(64).repeat_interleave(16), torch.randn(64, 32),
        torch.randint(0, 5000, (1024,)), torch.randn(5000, 32)), tmp_path)
    by_table = {}
    for entry in captured:
        specs = _tensor_specs(entry["inputs"])
        table = next(s["shape"][0] for s in specs if "float" in s["dtype"])
        (by_table[table],) = _int_gens(specs)
    assert by_table == {
        64: {"kind": "runs", "low": 0, "high": 64, "n_unique": 64,
             "n_runs": 64, "sorted": True},
        5000: {"kind": "index", "low": 0, "high": 5000},
    }
    assert sidecars and all(s.get("roundtrip") == "ok" for s in sidecars), [
        s.get("roundtrip") for s in sidecars]


def test_dynamic_capture_region_and_full_graph_share_runs_gen(tmp_path):
    def f(table, idx, w):
        return (table[idx] * w).relu().sum(-1)

    captured, sidecars = _capture(f, (
        torch.randn(100, 32), torch.arange(64).repeat_interleave(16),
        torch.randn(1024, 32)), tmp_path, dynamic=True)
    want = {"kind": "runs", "low": 0, "high": 64, "n_unique": 64,
            "n_runs": 64, "sorted": True}
    (entry,) = captured
    assert _int_gens(_tensor_specs(entry["inputs"])) == [want]
    (sidecar,) = sidecars
    assert _int_gens(_tensor_specs(sidecar["inputs"])) == [want]
    assert sidecar.get("roundtrip") == "ok", sidecar.get("roundtrip")
