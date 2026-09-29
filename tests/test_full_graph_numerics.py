import subprocess
import sys
import threading
from pathlib import Path

import pytest
import torch

import argparse
import json

import full_graph_harness
from full_graph_harness import (
    FullGraphNumericsError,
    compare_outputs,
    rng_dependent_output_leaves,
    run_numerics_gate,
    validate_full_graph_numerics,
)
from scripts.bench_parallel import (
    _benchmark_config_metadata,
    _persistent_worker_script,
    _write_results_output,
)


ROOT = Path(__file__).resolve().parents[1]


class _NestedOutput(torch.nn.Module):
    def forward(self, value):
        return {"main": (value + 1, [value.to(torch.int64)])}


def test_full_graph_numerics_accepts_nested_outputs():
    eager = _NestedOutput()

    def compiled(value):
        return {"main": ((value.double() + 1).float(), [value.to(torch.int64)])}

    eager_out, reference = validate_full_graph_numerics(
        eager, compiled, lambda: [torch.tensor([1.0])]
    )
    assert reference == "fp64"
    assert set(eager_out) == {"main"}


def test_full_graph_numerics_rejects_nested_output_mismatch():
    eager = _NestedOutput()

    def compiled(value):
        return {"main": (value + 1.1, [value.to(torch.int64)])}

    with pytest.raises(FullGraphNumericsError, match="output mismatch"):
        validate_full_graph_numerics(eager, compiled, lambda: [torch.tensor([1.0])])


def test_full_graph_numerics_seeds_eager_and_compiled_identically():
    def random_graph(value):
        return value + torch.rand_like(value)

    validate_full_graph_numerics(random_graph, random_graph, lambda: [torch.zeros(64)])


def test_full_graph_numerics_classifies_eager_self_instability():
    calls = []

    offsets = [0.0, 0.0, 100.0]

    def unstable_eager(value):
        calls.append(None)
        return value + offsets[len(calls) - 1]

    def compiled(value):
        return value + 5

    with pytest.raises(FullGraphNumericsError) as info:
        validate_full_graph_numerics(unstable_eager, compiled, lambda: [torch.zeros(1)])
    assert info.value.classification == "eager_self_instability"
    # eager, fp64 reference, eager rerun
    assert len(calls) == 3


def test_full_graph_numerics_gate_is_opt_in():
    script = _worker_script()
    full_graph = script.split("def bench_full_graph_one", 1)[1].split(
        "@preserve_compile_environment()\ndef bench_one", 1
    )[0]
    assert full_graph.count("if CHECK_NUMERICS:") == 2
    for call in full_graph.split("run_numerics_gate(")[:-1]:
        assert call.rstrip().endswith(("=", "_ =")) and "if CHECK_NUMERICS:" in call
    assert "eager_out = instance(*inputs)" in full_graph


def test_full_graph_worker_gates_default_and_cd_before_timing():
    script = _persistent_worker_script(
        "0",
        {
            "root": str(ROOT),
            "all_shapes": False,
            "no_cd": False,
            "n_warmup": 1,
            "n_rep": 1,
            "strict_gpu_lock": False,
        },
    )
    full_graph = script.split("def bench_full_graph_one", 1)[1].split(
        "@preserve_compile_environment()\ndef bench_one", 1
    )[0]

    assert full_graph.count("run_numerics_gate(") == 2
    assert full_graph.count("fullgraph=CHECK_NUMERICS") == 3
    assert "fullgraph=True" not in full_graph
    assert full_graph.index('label="default compiled"') < full_graph.index(
        "# Measure the default compiled artifact"
    )
    assert full_graph.index('label="coordinate-descent compiled"') < full_graph.index(
        "bench_cd = _make_bench_callable"
    )
    assert '"numerics_failed"' in script
    assert '"failure_classification"' in script


def test_full_graph_worker_excludes_numerics_from_compile_time():
    script = _persistent_worker_script(
        "0",
        {
            "root": str(ROOT),
            "all_shapes": False,
            "no_cd": False,
            "n_warmup": 1,
            "n_rep": 1,
            "strict_gpu_lock": False,
            "compile_time": True,
        },
    )
    full_graph = script.split("def bench_full_graph_one", 1)[1]
    timed = full_graph.split("_t0 = time.perf_counter()", 1)[1].split(
        "compile_time_s = time.perf_counter() - _t0", 1
    )[0]
    assert "run_numerics_gate" not in timed


def _f64(*values):
    return torch.tensor(values, dtype=torch.float64)


def test_compare_outputs_accepts_candidate_as_accurate_as_eager():
    # Cancellation: eager and compiled both miss the fp64 answer by 0.5 in
    # opposite directions. Plain allclose(eager, compiled) would reject this.
    reference = _f64(0.0)
    baseline = torch.tensor([0.5], dtype=torch.bfloat16)
    candidate = torch.tensor([-0.5], dtype=torch.bfloat16)
    ok, details = compare_outputs(reference, baseline, candidate)
    assert ok, details

    worse = torch.tensor([1.75], dtype=torch.bfloat16)
    ok, details = compare_outputs(reference, baseline, worse)
    assert not ok
    assert "rmse_vs_fp64" in details[0]


def test_compare_outputs_uses_allclose_tolerance():
    reference = _f64(1000.0)
    baseline = torch.tensor([1000.0])
    ok, details = compare_outputs(reference, baseline, torch.tensor([1005.0]))
    assert ok, details
    ok, _ = compare_outputs(reference, baseline, torch.tensor([1100.0]))
    assert not ok


def test_compare_outputs_accepts_bf16_rounding_when_reference_matches_eager():
    # Explicit low-precision output casts make the fp64 run round exactly like
    # eager, so eager has zero error; a one-ulp compiled difference still passes.
    baseline = torch.tensor([1632.0], dtype=torch.bfloat16)
    reference = baseline.double()
    candidate = torch.tensor([1640.0], dtype=torch.bfloat16)
    ok, details = compare_outputs(reference, baseline, candidate)
    assert ok, details
    ok, _ = compare_outputs(
        reference, baseline, torch.tensor([1700.0], dtype=torch.bfloat16)
    )
    assert not ok


def test_compare_outputs_checks_nested_outputs_per_leaf():
    reference = {"a": _f64(100.0), "b": [_f64(0.0)]}
    baseline = {"a": torch.tensor([100.0]), "b": [torch.tensor([0.0])]}
    ok, details = compare_outputs(reference, baseline, baseline)
    assert ok, details

    far = {"a": torch.tensor([102.0]), "b": [torch.tensor([0.0])]}
    ok, details = compare_outputs(reference, baseline, far)
    assert not ok
    assert details[0].startswith("leaf 0:")


def test_compare_outputs_requires_matching_nonfinite_values():
    reference = _f64(float("inf"), float("nan"), 1.0)
    baseline = torch.tensor([float("inf"), float("nan"), 1.0])
    ok, details = compare_outputs(reference, baseline, baseline.clone())
    assert ok, details
    for bad in (
        [float("-inf"), float("nan"), 1.0],
        [float("nan"), float("nan"), 1.0],
        [float("inf"), 1.0, 1.0],
    ):
        ok, details = compare_outputs(reference, baseline, torch.tensor(bad))
        assert not ok
        assert "nonfinite" in details[0]
    ok, details = compare_outputs(
        reference, baseline, torch.tensor([float("inf"), float("nan"), 5.0])
    )
    assert not ok
    assert "rmse_vs_fp64" in details[0]


def test_compare_outputs_ignores_positions_eager_overflows():
    reference = _f64(1e40, 1.0)
    baseline = torch.tensor([float("inf"), 1.0])
    candidate = torch.tensor([float("inf"), 1.0])
    ok, details = compare_outputs(reference, baseline, candidate)
    assert ok, details


def test_compare_outputs_requires_exact_integers_and_structure():
    ok, _ = compare_outputs(
        torch.tensor([1, 2]), torch.tensor([1, 2]), torch.tensor([1, 3])
    )
    assert not ok
    ok, _ = compare_outputs(
        {"a": _f64(1.0)}, {"a": torch.tensor([1.0])}, {"b": torch.tensor([1.0])}
    )
    assert not ok
    ok, _ = compare_outputs(
        _f64(1.0), torch.tensor([1.0]), torch.tensor([1.0], dtype=torch.float64)
    )
    assert not ok


def test_numerics_gate_falls_back_to_eager_without_fp64_reference():
    def no_fp64(value):
        if value.dtype == torch.float64:
            raise TypeError("float64 unsupported")
        return value + 1

    summary, eager_out = run_numerics_gate(
        no_fp64,
        no_fp64,
        lambda: [torch.randn(4)],
        label="default compiled",
        variant="default",
        seeds=2,
    )
    assert summary["status"] == "pass"
    assert summary["reference"] == "eager"
    assert summary["fp64_reference_available"] is False
    assert all(seed["pass"] is True for seed in summary["seeds"])
    assert eager_out.dtype == torch.float32

    def no_fp64_wrong(value):
        if value.dtype == torch.float64:
            raise TypeError("float64 unsupported")
        return value + 2

    with pytest.raises(FullGraphNumericsError) as info:
        run_numerics_gate(
            no_fp64,
            no_fp64_wrong,
            lambda: [torch.randn(4)],
            label="default compiled",
            variant="default",
            seeds=1,
        )
    assert info.value.numerics["status"] == "fail"
    assert info.value.numerics["reference"] == "eager"
    assert info.value.numerics["seeds"][0]["reference"] == "eager"


def test_fp64_reference_covers_tuple_inputs():
    def add(value):
        return value + 1

    _, reference = validate_full_graph_numerics(add, add, lambda: (torch.randn(4),))
    assert reference == "fp64"


def test_fp64_reference_upcasts_explicit_low_precision_casts():
    def casts(value, weight):
        hidden = torch.ops.prims.convert_element_type.default(value, torch.bfloat16)
        return torch.mm(weight, hidden.to(torch.float32))

    _, reference = validate_full_graph_numerics(
        casts, casts, lambda: (torch.randn(4, 4), torch.randn(4, 4))
    )
    assert reference == "fp64"


def test_numerics_gate_passes_real_full_graph():
    instance, inputs, _ = full_graph_harness.load_full_graph(
        ROOT / "repros/models/torchbench/infer/lennard_jones/full_graph_000.py",
        default_device="cpu",
    )
    summary, _ = run_numerics_gate(
        instance,
        torch.compile(instance, fullgraph=True),
        lambda: inputs,
        label="default compiled",
        variant="default",
        seeds=1,
    )
    assert summary["status"] == "pass"
    assert summary["reference"] == "fp64"


def test_fp64_reference_upcasts_module_parameters():
    linear = torch.nn.Linear(4, 4)
    seen = []
    linear.register_forward_pre_hook(
        lambda module, args: seen.append((module.weight.dtype, args[0].dtype))
    )
    validate_full_graph_numerics(linear, linear, lambda: [torch.randn(2, 4)])
    assert seen == [
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
        (torch.float32, torch.float32),
    ]
    assert linear.weight.dtype == torch.float32


class _DropoutPartition(torch.nn.Module):
    def forward(self, value):
        mask = torch.ops.aten.rand_like.default(value) > 0.5
        return value.sum(), mask, value * mask


class _InputFreeRng(torch.nn.Module):
    def forward(self, value):
        noise = torch.rand(value.shape, device=value.device)
        return value + 1, value + noise


class _InPlaceRng(torch.nn.Module):
    def forward(self, value):
        buffer = torch.empty_like(value)
        buffer.uniform_()
        return value * 2, buffer.view(-1)


def test_rng_dependent_output_leaves_follows_data_flow():
    def value():
        return [torch.randn(8)]

    assert rng_dependent_output_leaves(_DropoutPartition(), value) == {1, 2}
    assert rng_dependent_output_leaves(_NestedOutput(), value) == set()
    assert rng_dependent_output_leaves(_InputFreeRng(), value) == {1}
    assert rng_dependent_output_leaves(_InPlaceRng(), value) == {1}


class _ViewBeforeInPlaceRng(torch.nn.Module):
    def forward(self, value):
        buffer = torch.empty_like(value)
        view = buffer.view(-1)
        buffer.uniform_()
        return value * 2, view


class _RngCopiedIntoBuffer(torch.nn.Module):
    def forward(self, value):
        buffer = torch.zeros_like(value)
        alias = buffer[:]
        buffer.copy_(torch.rand_like(value))
        return value * 2, alias


def test_rng_taint_follows_storage_aliases():
    def value():
        return [torch.randn(8)]

    assert rng_dependent_output_leaves(_ViewBeforeInPlaceRng(), value) == {1}
    assert rng_dependent_output_leaves(_RngCopiedIntoBuffer(), value) == {1}


def test_numerics_gate_rejects_view_created_before_rng_write():
    def eager(value):
        buffer = torch.empty_like(value)
        view = buffer.view(-1)
        buffer.uniform_()
        return view

    summary, _ = run_numerics_gate(
        eager,
        lambda value: torch.zeros_like(value).view(-1),
        lambda: [torch.randn(4)],
        label="default compiled",
        variant="default",
        seeds=1,
    )
    assert summary["status"] == "numerics_unavailable"
    assert summary["skipped_rng_output_leaves"] == [0]
    assert summary["unjustified_skipped_output_leaves"] == [0]


def test_rng_dependent_output_leaves_reports_detection_failure():
    def broken(value):
        raise ValueError("boom")

    assert rng_dependent_output_leaves(broken, lambda: [torch.randn(2)]) is None


def test_full_graph_numerics_skips_rng_leaves_but_checks_the_rest():
    eager = _DropoutPartition()

    def compiled_rng_differs(value):
        mask = torch.ones_like(value, dtype=torch.bool)
        return value.sum(), mask, value * mask

    def inputs():
        return [torch.randn(64)]

    skip = rng_dependent_output_leaves(eager, inputs)
    with pytest.raises(FullGraphNumericsError):
        validate_full_graph_numerics(eager, compiled_rng_differs, inputs)
    validate_full_graph_numerics(
        eager, compiled_rng_differs, inputs, skip_output_leaves=skip
    )

    def compiled_sum_wrong(value):
        mask = torch.ones_like(value, dtype=torch.bool)
        return value.sum() + 1, mask, value * mask

    with pytest.raises(FullGraphNumericsError, match="leaf 0"):
        validate_full_graph_numerics(
            eager, compiled_sum_wrong, inputs, skip_output_leaves=skip
        )


def test_numerics_gate_records_every_seed_on_pass():
    eager = _NestedOutput()
    summary, eager_out = run_numerics_gate(
        eager,
        eager,
        lambda: [torch.randn(4)],
        label="default compiled",
        variant="default",
        seeds=3,
    )
    assert summary["status"] == "pass"
    assert [seed["seed"] for seed in summary["seeds"]] == [0, 1, 2]
    assert all(seed["pass"] is True for seed in summary["seeds"])
    assert summary["method"] == "torch._dynamo.utils.same"
    assert summary["reference"] == "fp64"
    assert summary["tolerance"] == 1e-2
    assert summary["fp64_reference_available"] is True
    assert isinstance(eager_out, dict)
    assert summary["rng_detection_ok"] is True
    assert summary["skipped_rng_output_leaves"] == []
    assert summary["unjustified_skipped_output_leaves"] == []


def test_numerics_gate_fails_closed_when_rng_skips_lack_output_contract():
    def gate(**kwargs):
        return run_numerics_gate(
            _DropoutPartition(),
            _DropoutPartition(),
            lambda: [torch.randn(8)],
            label="default compiled",
            variant="default",
            seeds=1,
            **kwargs,
        )[0]

    unavailable = gate()
    assert unavailable["status"] == "numerics_unavailable"
    assert unavailable["seeds"][0]["pass"] is True
    assert unavailable["skipped_rng_output_leaves"] == [1, 2]
    assert unavailable["unjustified_skipped_output_leaves"] == [1, 2]

    assert gate(non_semantic_output_leaves={1})["status"] == "numerics_unavailable"
    justified = gate(non_semantic_output_leaves={1, 2})
    assert justified["status"] == "pass"
    assert justified["unjustified_skipped_output_leaves"] == []


def test_numerics_gate_unavailable_when_rng_detection_fails(monkeypatch):
    monkeypatch.setattr(
        full_graph_harness, "rng_dependent_output_leaves", lambda *args: None
    )
    summary, _ = run_numerics_gate(
        torch.rand_like,
        torch.zeros_like,
        lambda: [torch.zeros(64)],
        label="default compiled",
        variant="default",
        seeds=1,
    )
    assert summary["status"] == "numerics_unavailable"
    assert summary["rng_detection_ok"] is False


def test_numerics_gate_runs_all_seeds_and_attaches_structured_failure():
    eager = _NestedOutput()

    def compiled(value):
        offset = 1.1 if value.sum() > 0 else 1.0
        return {"main": (value + offset, [value.to(torch.int64)])}

    with pytest.raises(FullGraphNumericsError) as info:
        run_numerics_gate(
            eager,
            compiled,
            lambda: [torch.tensor([1.0 if torch.initial_seed() == 0 else -1.0])],
            label="coordinate-descent compiled",
            variant="coordinate_descent",
            seeds=2,
        )
    error = info.value
    assert error.failed_variant == "coordinate_descent"
    assert error.classification == "stable_eager_vs_compiled_mismatch"
    numerics = error.numerics
    assert numerics["status"] == "fail"
    first, second = numerics["seeds"]
    assert first["seed"] == 0 and first["pass"] is False
    assert first["failure_classification"] == "stable_eager_vs_compiled_mismatch"
    assert first["eager_self_check"] == {"pass": True}
    assert first["diagnostics"]
    assert second == {
        "seed": 1,
        "pass": True,
        "reference": "fp64",
        "failure_classification": None,
        "eager_self_check": None,
        "diagnostics": [],
        "worst_output_idx": None,
        "per_output": [],
    }
    assert first["worst_output_idx"] == 0
    assert first["per_output"][0]["pass"] is False
    assert first["per_output"][0]["err_compiled"] > first["per_output"][0]["err_eager"]
    json.dumps(numerics)


def _worker_script(**overrides):
    args = {
        "root": str(ROOT),
        "all_shapes": False,
        "no_cd": False,
        "n_warmup": 1,
        "n_rep": 1,
        "strict_gpu_lock": False,
    }
    args.update(overrides)
    return _persistent_worker_script("0", args)


def test_partition_worker_gates_default_and_cd_before_capture_when_enabled():
    script = _worker_script(check_numerics=True, numerics_seeds=3)
    assert "CHECK_NUMERICS = True" in script
    assert "NUMERICS_SEEDS = 3" in script
    assert "NUMERICS_ATOL" not in script

    bench_one = script.split("def bench_one(", 1)[1].split(
        "@preserve_compile_environment()", 1
    )[0]
    default_gate = bench_one.index('"default compiled",\n')
    assert default_gate < bench_one.index(
        "graph_default, default_is_graph = _capture_cudagraph"
    )
    cd_gate = bench_one.index('"coordinate-descent compiled"')
    assert cd_gate < bench_one.index("graph_cd, cd_is_graph = _capture_cudagraph")
    assert '["numerics"] = numerics' in bench_one
    assert '["coord_descent_numerics"] = cd_numerics' in bench_one


def test_partition_worker_numerics_disabled_by_default():
    assert "CHECK_NUMERICS = False" in _worker_script()


def test_worker_keeps_compile_environment_guard_on_bench_one():
    assert "@preserve_compile_environment()\ndef bench_one(" in _worker_script()


def test_worker_reports_structured_numerics_on_success_and_failure():
    script = _worker_script()
    full_graph = script.split("def bench_full_graph_one", 1)[1].split(
        "@preserve_compile_environment()\ndef bench_one", 1
    )[0]
    assert 'result["default"]["numerics"] = numerics' in full_graph
    assert '["coord_descent_numerics"] = cd_numerics' in full_graph
    assert 'variant="coordinate_descent"' in full_graph
    assert '"failed_variant": getattr(e, "failed_variant", None)' in script
    assert '"numerics": getattr(e, "numerics", None)' in script


def test_full_graph_inputs_are_seeded_independent_of_worker_rng_state():
    source = _worker_script(full_graphs=True)
    helper = "def _get_or_load_full_graph" + source.split(
        "def _get_or_load_full_graph", 1
    )[1].split("\n\n", 1)[0]

    def load_full_graph(definition, **kwargs):
        return object(), (torch.randn(8), torch.randint(0, 10, (3,))), definition

    namespace = {
        "torch": torch,
        "_prefetch_lock": threading.Lock(),
        "_prefetch_cache": {},
        "load_full_graph_definition": lambda path: path,
        "load_full_graph": load_full_graph,
    }
    exec(helper, namespace)
    load = namespace["_get_or_load_full_graph"]

    torch.manual_seed(1234)
    _, first, _ = load("graph.py")
    torch.randn(100)
    _, second, _ = load("graph.py")

    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_parent_preserves_structured_numerics_failure_fields():
    source = (ROOT / "scripts" / "bench_parallel.py").read_text()
    for key in ("failure_classification", "failed_variant", "numerics"):
        assert f'error_payload.get("{key}")' in source or (
            f'error_payload.get(\n                                "{key}"' in source
        )
    assert '"failed_variant",\n                "numerics",' in source


def _metadata_args(**overrides):
    args = argparse.Namespace(
        worker_init=[],
        no_cd=False,
        strict_gpu_lock=False,
        gpus=None,
        workers_per_gpu=1,
        full_graphs=False,
        check_numerics=False,
        numerics_seeds=1,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_numerics_metadata_reflects_partition_and_full_graph_modes():
    assert "numerics_check" not in _benchmark_config_metadata(_metadata_args())

    partition = _benchmark_config_metadata(
        _metadata_args(check_numerics=True, numerics_seeds=4)
    )["numerics_check"]
    assert partition == {
        "enabled": True,
        "seeds": 4,
        "method": "torch._dynamo.utils.same",
        "tolerance": 1e-2,
        "reference": "fp64_or_eager",
    }

    assert "numerics_check" not in _benchmark_config_metadata(
        _metadata_args(full_graphs=True)
    )

    full_graph = _benchmark_config_metadata(
        _metadata_args(full_graphs=True, check_numerics=True, numerics_seeds=4)
    )["numerics_check"]
    assert full_graph["enabled"] is True
    assert full_graph["seeds"] == 1


def test_results_output_exposes_numerics_check_metadata(tmp_path):
    output = tmp_path / "results.json"
    config = _benchmark_config_metadata(_metadata_args(check_numerics=True))
    _write_results_output(
        output,
        {},
        {},
        total=0,
        done=0,
        failed=0,
        elapsed=0.0,
        config_metadata=config,
    )
    metadata = json.loads(output.read_text())["_metadata"]
    assert metadata["numerics_check"] == config["numerics_check"]
    assert metadata["numerics_check"]["enabled"] is True


def test_results_output_without_numerics_check_matches_legacy_layout(tmp_path):
    output = tmp_path / "results.json"
    config = _benchmark_config_metadata(_metadata_args())
    assert sorted(config) == [
        "coordinate_descent",
        "gpus",
        "strict_gpu_lock",
        "worker_init",
        "workers_per_gpu",
    ]
    _write_results_output(
        output,
        {},
        {},
        total=0,
        done=0,
        failed=0,
        elapsed=0.0,
        config_metadata=config,
    )
    metadata = json.loads(output.read_text())["_metadata"]
    assert "numerics_check" not in metadata
    assert metadata["benchmark_config"] == config


def test_bench_parallel_loads_via_runpy_with_sibling_imports(tmp_path):
    script = ROOT / "scripts" / "bench_parallel.py"
    command = (
        "import runpy; "
        f"ns = runpy.run_path({str(script)!r}); "
        "assert 'find_full_graphs' in ns"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_numerics_gate_compares_against_eager_when_fp64_is_nonfinite():
    # fp64 gives sqrt(-1) = NaN where float32 rounds x - (x + 1) to 0.
    def eager(value):
        return torch.sqrt(value - (value + 1))

    def wrong(value):
        return torch.full_like(value, 12345.0)

    with pytest.raises(FullGraphNumericsError) as info:
        run_numerics_gate(
            eager,
            wrong,
            lambda: [torch.tensor([2.0**24])],
            label="default compiled",
            variant="default",
            seeds=1,
        )
    assert info.value.numerics["status"] == "fail"
    assert info.value.numerics["reference"] == "eager"
    assert info.value.numerics["seeds"][0]["pass"] is False

    summary, _ = run_numerics_gate(
        eager,
        eager,
        lambda: [torch.tensor([2.0**24])],
        label="default compiled",
        variant="default",
        seeds=1,
    )
    assert summary["status"] == "pass"
    assert summary["reference"] == "eager"

    ok, details = compare_outputs(
        _f64(float("nan")), torch.tensor([0.0]), torch.tensor([0.0])
    )
    assert not ok
    assert "fp64 reference is nonfinite" in details[0]
