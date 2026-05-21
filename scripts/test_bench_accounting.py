import fcntl
import json
import operator
import os
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bench import _count_bytes, _resolve_input_path
from bench_parallel import _compute_worker_count, _filter_gpus, load_benchmark_set
from capture_hook import _CaptureState
import gpu_lock as gpu_lock_module
from canonicalize_repros import _spec_to_T, parse_make_inputs
from repro_harness import load_shape_configs, make_inputs_from_config, make_inputs_safely


def _touch_repro(canonical_dir: Path, name: str) -> None:
    repro_dir = canonical_dir / name
    repro_dir.mkdir(parents=True)
    (repro_dir / "repro.py").write_text("# repro\n")


def test_count_bytes_deduplicates_output_aliases():
    inp = torch.empty(4, dtype=torch.float32)
    out = torch.empty(16, dtype=torch.float32)

    assert _count_bytes([inp], (out, out.view(4, 4), out.reshape(2, 8))) == (
        inp.nelement() + out.nelement()
    ) * inp.element_size()


def test_count_bytes_counts_distinct_outputs():
    inp = torch.empty(4, dtype=torch.float32)
    out0 = torch.empty(16, dtype=torch.float32)
    out1 = torch.empty(16, dtype=torch.float32)

    assert _count_bytes([inp], (out0, out1)) == (
        inp.nelement() + out0.nelement() + out1.nelement()
    ) * inp.element_size()


def test_resolve_input_path_strips_accidental_whitespace():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        repro = tmp_path / "repro.py"
        repro.write_text("# repro\n")
        assert _resolve_input_path(f" {repro}") == repro


def test_resolve_input_path_rejects_missing_path():
    missing = " models/does/not/exist/repro.py"
    try:
        _resolve_input_path(missing)
    except FileNotFoundError as exc:
        assert "Benchmark path does not exist" in str(exc)
    else:
        raise AssertionError("missing benchmark path should fail")


def test_resolve_input_path_rejects_whitespace_only_path():
    try:
        _resolve_input_path(" ")
    except FileNotFoundError as exc:
        assert "Benchmark path does not exist" in str(exc)
    else:
        raise AssertionError("whitespace-only benchmark path should fail")


def test_compute_worker_count_defaults_to_one_worker_per_gpu():
    assert _compute_worker_count(
        num_repros=1133,
        num_gpus=2,
        max_workers=None,
        workers_per_gpu=None,
    ) == (2, 1)


def test_compute_worker_count_max_workers_raises_effective_gpu_cap():
    assert _compute_worker_count(
        num_repros=1133,
        num_gpus=2,
        max_workers=4,
        workers_per_gpu=None,
    ) == (4, 2)


def test_compute_worker_count_explicit_workers_per_gpu_caps_max_workers():
    assert _compute_worker_count(
        num_repros=1133,
        num_gpus=2,
        max_workers=4,
        workers_per_gpu=1,
    ) == (2, 1)


def test_compute_worker_count_never_exceeds_repro_count():
    assert _compute_worker_count(
        num_repros=3,
        num_gpus=2,
        max_workers=8,
        workers_per_gpu=None,
    ) == (3, 4)


def test_load_benchmark_set_accepts_current_patterns_schema():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        canonical_dir = root / "repros" / "canonical"
        _touch_repro(canonical_dir, "foo")
        manifest = root / "v1.json"
        manifest.write_text(json.dumps({
            "version": "v1",
            "patterns": [
                {"name": "foo", "kind": "pointwise"},
                {"name": "missing", "kind": "reduction"},
            ],
        }))

        repros, entries = load_benchmark_set(manifest, canonical_dir=canonical_dir)
        assert entries == 2
        assert repros == [canonical_dir / "foo" / "repro.py"]


def test_load_benchmark_set_accepts_legacy_benchmarks_schema():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        canonical_dir = root / "repros" / "canonical"
        _touch_repro(canonical_dir, "bar")
        manifest = root / "v0.json"
        manifest.write_text(json.dumps({
            "benchmarks": [
                {"repro": "bar", "shape": "default"},
                {"repro": "bar", "shape": "shape_1"},
            ],
        }))

        repros, entries = load_benchmark_set(manifest, canonical_dir=canonical_dir)
        assert entries == 2
        assert repros == [canonical_dir / "bar" / "repro.py"]


def test_filter_gpus_selects_physical_indices():
    gpus = [
        {"index": "0", "name": "NVIDIA H100", "kind": "H100"},
        {"index": "1", "name": "NVIDIA H100", "kind": "H100"},
    ]
    assert _filter_gpus(gpus, "1") == [gpus[1]]
    assert _filter_gpus(gpus, "0,1") == gpus


def test_make_inputs_safely_handles_legacy_bool_randn():
    (value,) = make_inputs_safely(
        lambda: [torch.randn([2, 3], dtype=torch.bool, device="cpu")]
    )
    assert value.dtype is torch.bool
    assert value.shape == (2, 3)


def test_shape_configs_accept_general_input_generators():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "repro.py").write_text("# repro\n")
        (root / "shapes.txt").write_text(
            "case: (T([8], i64, gen=Perm(8)), T([2, 4], i64, gen=Index(4)))\n"
        )

        config = load_shape_configs(str(root / "repro.py"))["case"]
        assert config["inputs"][0]["gen"] == {"kind": "permutation", "size": 8}
        assert config["inputs"][1]["gen"] == {"kind": "index", "low": 0, "high": 4}


def test_make_inputs_from_config_honors_generators_on_cpu():
    perm, bounded = make_inputs_from_config({
        "inputs": [
            {
                "kind": "tensor",
                "shape": [8],
                "dtype": "torch.int64",
                "stride": None,
                "device": "cpu",
                "gen": {"kind": "permutation", "size": 8},
            },
            {
                "kind": "tensor",
                "shape": [64],
                "dtype": "torch.int64",
                "stride": None,
                "device": "cpu",
                "gen": {"kind": "index", "low": 0, "high": 4},
            },
        ],
    })
    assert sorted(perm.tolist()) == list(range(8))
    assert int(bounded.min()) >= 0
    assert int(bounded.max()) < 4


def test_parse_make_inputs_preserves_generators():
    with tempfile.TemporaryDirectory() as tmp:
        repro = Path(tmp) / "repro.py"
        repro.write_text("""
import torch


def make_inputs():
    return [
        torch.randint(0, 4, [2, 4], dtype=torch.int64, device='cuda'),
        torch.randperm(8, dtype=torch.int64, device='cuda').reshape([2, 4]),
    ]
""")

        specs = parse_make_inputs(repro)
        assert specs[0]["gen"] == {"kind": "index", "low": 0, "high": 4}
        assert specs[1]["gen"] == {"kind": "permutation", "size": 8}
        assert _spec_to_T(specs[0]) == "T([2, 4], i64, gen=Index(4))"
        assert _spec_to_T(specs[1]) == "T([2, 4], i64, gen=Perm(8))"


def test_capture_hook_traces_select_to_index_bound():
    graph = torch.fx.Graph()
    indices = graph.placeholder("arg1_1")
    indices.meta["val"] = torch.empty(2, 200, dtype=torch.int64)
    data = graph.placeholder("arg0_1")
    data.meta["val"] = torch.empty(10000, 64, dtype=torch.bfloat16)

    selected = graph.call_function(torch.ops.aten.select.int, (indices, 0, 0))
    selected.meta["val"] = torch.empty(200, dtype=torch.int64)
    indexed = graph.call_function(torch.ops.aten.index.Tensor, (data, [selected]))
    indexed.meta["val"] = torch.empty(200, 64, dtype=torch.bfloat16)
    graph.output(indexed)

    gm = torch.fx.GraphModule({}, graph)
    with tempfile.TemporaryDirectory() as tmp:
        state = _CaptureState(tmp)
        bounds = state._infer_index_bounds(gm, {
            "arg1_1": {"dtype": "torch.int64"},
            "arg0_1": {"dtype": "torch.bfloat16"},
        })

    assert bounds["arg1_1"] == 10000


def test_capture_hook_traces_gathered_values_to_embedding_bound():
    graph = torch.fx.Graph()
    token_types = graph.placeholder("token_types")
    token_types.meta["val"] = torch.empty(1, 512, dtype=torch.int64)
    positions = graph.placeholder("positions")
    positions.meta["val"] = torch.empty(1, 512, dtype=torch.int64)
    table = graph.placeholder("table")
    table.meta["val"] = torch.empty(2, 128, dtype=torch.float32)

    gathered = graph.call_function(
        torch.ops.aten.gather.default,
        (token_types, 1, positions),
    )
    gathered.meta["val"] = torch.empty(1, 512, dtype=torch.int64)
    embedded = graph.call_function(torch.ops.aten.embedding.default, (table, gathered))
    embedded.meta["val"] = torch.empty(1, 512, 128, dtype=torch.float32)
    graph.output(embedded)

    gm = torch.fx.GraphModule({}, graph)
    with tempfile.TemporaryDirectory() as tmp:
        state = _CaptureState(tmp)
        bounds = state._infer_index_bounds(gm, {
            "token_types": {"dtype": "torch.int64"},
            "positions": {"dtype": "torch.int64"},
            "table": {"dtype": "torch.float32"},
        })

    assert bounds["token_types"] == 2
    assert bounds["positions"] == 512


def _real_targets(component):
    return {
        node.target
        for node in component
        if _CaptureState._is_real_compute_node(node)
    }


class _RecordingCaptureState(_CaptureState):
    def __init__(self, output_dir):
        super().__init__(output_dir, validate=False)
        self.generated_targets = []

    def _generate_repro_file(self, gm, placeholder_info, meta, filename, shape_params=None):
        self.generated_targets.append(_real_targets(gm.graph.nodes))
        return str(Path(self.output_dir) / filename)


def test_capture_hook_process_graph_splits_horizontal_fusion_regions():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(4)
    viewed = graph.call_function(torch.ops.aten.view.default, (x, [4]))
    viewed.meta["val"] = torch.empty(4)
    left = graph.call_function(torch.ops.aten.sin.default, (viewed,))
    left.meta["val"] = torch.empty(4)
    right = graph.call_function(torch.ops.aten.cos.default, (viewed,))
    right.meta["val"] = torch.empty(4)
    graph.output((left, right))

    gm = torch.fx.GraphModule({}, graph)
    with tempfile.TemporaryDirectory() as tmp:
        state = _RecordingCaptureState(tmp)
        state.process_graph(gm)

    assert len(state.generated_targets) == 2
    assert {frozenset(targets) for targets in state.generated_targets} == {
        frozenset({torch.ops.aten.sin.default}),
        frozenset({torch.ops.aten.cos.default}),
    }


def test_capture_hook_process_graph_splits_alias_and_detach_branches():
    for transparent_target in (
        torch.ops.aten.alias.default,
        torch.ops.aten.detach.default,
    ):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.empty(4)
        adapter = graph.call_function(transparent_target, (x,))
        adapter.meta["val"] = torch.empty(4)
        left = graph.call_function(torch.ops.aten.sin.default, (adapter,))
        left.meta["val"] = torch.empty(4)
        right = graph.call_function(torch.ops.aten.cos.default, (adapter,))
        right.meta["val"] = torch.empty(4)
        graph.output((left, right))

        gm = torch.fx.GraphModule({}, graph)
        with tempfile.TemporaryDirectory() as tmp:
            state = _RecordingCaptureState(tmp)
            state.process_graph(gm)

        assert len(state.generated_targets) == 2
        assert {frozenset(targets) for targets in state.generated_targets} == {
            frozenset({torch.ops.aten.sin.default}),
            frozenset({torch.ops.aten.cos.default}),
        }


def test_capture_hook_splits_independent_horizontal_chains():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    viewed = graph.call_function(torch.ops.aten.view.default, (x, [4]))
    left = graph.call_function(torch.ops.aten.sin.default, (viewed,))
    right = graph.call_function(torch.ops.aten.cos.default, (viewed,))
    graph.output((left, right))

    gm = torch.fx.GraphModule({}, graph)
    with tempfile.TemporaryDirectory() as tmp:
        state = _CaptureState(tmp)
        components = state._split_data_dependent_components(
            [[viewed, left, right]],
            gm,
        )

    assert len(components) == 2
    assert {frozenset(_real_targets(c)) for c in components} == {
        frozenset({torch.ops.aten.sin.default}),
        frozenset({torch.ops.aten.cos.default}),
    }
    assert all(viewed in component for component in components)


def test_capture_hook_hash_distinguishes_getitem_indices():
    graph = torch.fx.Graph()
    pair = graph.placeholder("pair")
    left_input = graph.call_function(operator.getitem, (pair, 0))
    left_input.meta["val"] = torch.empty(4)
    right_input = graph.call_function(operator.getitem, (pair, 1))
    right_input.meta["val"] = torch.empty(4)
    left = graph.call_function(torch.ops.aten.sin.default, (left_input,))
    left.meta["val"] = torch.empty(4)
    right = graph.call_function(torch.ops.aten.sin.default, (right_input,))
    right.meta["val"] = torch.empty(4)
    graph.output((left, right))

    gm = torch.fx.GraphModule({}, graph)
    with tempfile.TemporaryDirectory() as tmp:
        state = _RecordingCaptureState(tmp)
        state.process_graph(gm)

    assert len(state.generated_targets) == 2
    assert state.generated_targets == [
        {torch.ops.aten.sin.default},
        {torch.ops.aten.sin.default},
    ]


def test_capture_hook_keeps_getitem_data_dependency_together():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    values_and_indices = graph.call_function(torch.ops.aten.max.dim, (x, 1))
    values = graph.call_function(operator.getitem, (values_and_indices, 0))
    out = graph.call_function(torch.ops.aten.sin.default, (values,))
    graph.output(out)

    gm = torch.fx.GraphModule({}, graph)
    with tempfile.TemporaryDirectory() as tmp:
        state = _CaptureState(tmp)
        components = state._split_data_dependent_components(
            [[values_and_indices, values, out]],
            gm,
        )

    assert len(components) == 1
    assert components[0] == [values_and_indices, values, out]


def test_capture_hook_does_not_let_external_getitem_create_horizontal_fusion():
    graph = torch.fx.Graph()
    pair = graph.placeholder("pair")
    left_input = graph.call_function(operator.getitem, (pair, 0))
    right_input = graph.call_function(operator.getitem, (pair, 1))
    left = graph.call_function(torch.ops.aten.sin.default, (left_input,))
    right = graph.call_function(torch.ops.aten.cos.default, (right_input,))
    graph.output((left, right))

    gm = torch.fx.GraphModule({}, graph)
    with tempfile.TemporaryDirectory() as tmp:
        state = _CaptureState(tmp)
        components = state._split_data_dependent_components(
            [[left_input, right_input, left, right]],
            gm,
        )

    assert len(components) == 2
    assert {frozenset(_real_targets(c)) for c in components} == {
        frozenset({torch.ops.aten.sin.default}),
        frozenset({torch.ops.aten.cos.default}),
    }
    assert any(left_input in component and right_input not in component for component in components)
    assert any(right_input in component and left_input not in component for component in components)


def test_capture_hook_prunes_unrelated_transparent_nodes_from_single_compute_region():
    graph = torch.fx.Graph()
    pair = graph.placeholder("pair")
    left_input = graph.call_function(operator.getitem, (pair, 0))
    right_input = graph.call_function(operator.getitem, (pair, 1))
    out = graph.call_function(torch.ops.aten.sin.default, (left_input,))
    graph.output((out, right_input))

    gm = torch.fx.GraphModule({}, graph)
    with tempfile.TemporaryDirectory() as tmp:
        state = _CaptureState(tmp)
        components = state._split_data_dependent_components(
            [[left_input, right_input, out]],
            gm,
        )

    assert components == [[left_input, out]]


def test_capture_hook_prunes_unused_getitem_descendant_from_tuple_compute():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    values_and_indices = graph.call_function(torch.ops.aten.max.dim, (x, 1))
    values = graph.call_function(operator.getitem, (values_and_indices, 0))
    indices = graph.call_function(operator.getitem, (values_and_indices, 1))
    out = graph.call_function(torch.ops.aten.sin.default, (values,))
    graph.output((out, indices))

    gm = torch.fx.GraphModule({}, graph)
    with tempfile.TemporaryDirectory() as tmp:
        state = _CaptureState(tmp)
        components = state._split_data_dependent_components(
            [[values_and_indices, values, indices, out]],
            gm,
        )

    assert components == [[values_and_indices, values, out]]


def test_gpu_lock_metadata_marks_lock_mode():
    with tempfile.TemporaryDirectory() as tmp:
        with gpu_lock_module.gpu_lock(0, lock_dir=tmp, label="exclusive-test") as lock_path:
            metadata = gpu_lock_module._read_lock_metadata(lock_path)
            assert metadata["pid"] == str(os.getpid())
            assert metadata["mode"] == "exclusive"
            assert metadata["label"] == "exclusive-test"

        with gpu_lock_module.gpu_shared_lock(0, lock_dir=tmp, label="shared-test") as lock_path:
            metadata = gpu_lock_module._read_lock_metadata(lock_path)
            assert metadata["pid"] == str(os.getpid())
            assert metadata["mode"] == "shared"
            assert metadata["label"] == "shared-test"


def test_gpu_lock_detects_dead_pid_metadata_while_blocked():
    with tempfile.TemporaryDirectory() as tmp:
        lock_path = Path(tmp) / "gpu_0.lock"
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
        old_stale_grace = gpu_lock_module.STALE_LOCK_GRACE_S
        gpu_lock_module.STALE_LOCK_GRACE_S = 0
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            lock_path.write_text(
                "pid=999999999\n"
                "gpu=0\n"
                "mode=shared\n"
                "label=dead-test\n"
            )

            try:
                with gpu_lock_module.gpu_lock(0, lock_dir=tmp, timeout_s=10):
                    pass
            except TimeoutError as exc:
                assert "appears stale" in str(exc)
                assert "pid=999999999" in str(exc)
            else:
                raise AssertionError("dead PID metadata should fail instead of waiting")
        finally:
            gpu_lock_module.STALE_LOCK_GRACE_S = old_stale_grace
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


if __name__ == "__main__":
    test_count_bytes_deduplicates_output_aliases()
    test_count_bytes_counts_distinct_outputs()
    test_resolve_input_path_strips_accidental_whitespace()
    test_resolve_input_path_rejects_missing_path()
    test_resolve_input_path_rejects_whitespace_only_path()
    test_compute_worker_count_defaults_to_one_worker_per_gpu()
    test_compute_worker_count_max_workers_raises_effective_gpu_cap()
    test_compute_worker_count_explicit_workers_per_gpu_caps_max_workers()
    test_compute_worker_count_never_exceeds_repro_count()
    test_load_benchmark_set_accepts_current_patterns_schema()
    test_load_benchmark_set_accepts_legacy_benchmarks_schema()
    test_filter_gpus_selects_physical_indices()
    test_make_inputs_safely_handles_legacy_bool_randn()
    test_shape_configs_accept_general_input_generators()
    test_make_inputs_from_config_honors_generators_on_cpu()
    test_parse_make_inputs_preserves_generators()
    test_capture_hook_traces_select_to_index_bound()
    test_capture_hook_traces_gathered_values_to_embedding_bound()
    test_capture_hook_process_graph_splits_horizontal_fusion_regions()
    test_capture_hook_process_graph_splits_alias_and_detach_branches()
    test_capture_hook_splits_independent_horizontal_chains()
    test_capture_hook_hash_distinguishes_getitem_indices()
    test_capture_hook_keeps_getitem_data_dependency_together()
    test_capture_hook_does_not_let_external_getitem_create_horizontal_fusion()
    test_capture_hook_prunes_unrelated_transparent_nodes_from_single_compute_region()
    test_capture_hook_prunes_unused_getitem_descendant_from_tuple_compute()
    test_gpu_lock_metadata_marks_lock_mode()
    test_gpu_lock_detects_dead_pid_metadata_while_blocked()
    print("All tests passed.")
