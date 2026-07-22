import fcntl
import inspect
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

from bench import _resolve_input_path
from bench_report import load_results as load_report_results
from bench_parallel import (
    _classify_full_graph_definition,
    _compute_worker_count,
    _filter_gpus,
    _preflight_full_graphs,
    _write_results_output,
    find_full_graphs,
    load_benchmark_set,
)
from byte_accounting import count_bytes_naive
from capture_hook import _CaptureState
import gpu_lock as gpu_lock_module
from canonicalize_repros import _spec_to_T, parse_make_inputs
from full_graph_harness import (
    FullGraphDefinition,
    graph_constraints_from_gm,
    infer_index_bounds_from_gm,
    infer_full_graph_source,
    infer_permutation_indices_from_gm,
    load_full_graph,
    make_tensor_from_spec,
    parse_full_graph_inputs,
    parse_full_graph_tensor_attrs,
    result_metadata,
    write_full_graph_metadata,
)
from input_codec import spec_from_compact
from repro_harness import load_shape_configs, make_inputs_from_config, make_inputs_safely


def _decode_meta_input(entry: list) -> dict:
    """Decode one compact full-graph-metadata input/tensor_attr entry back to a
    verbose spec. write_full_graph_metadata persists inputs/tensor_attrs in the
    input_codec compact form ([[shape], dtype, opts]); the metadata tests assert
    on named fields, so round-trip through the codec. 'exact' is normalized to a
    bool (absent -> False) to match the verbose graph_constraints_from_gm form.
    """
    spec = spec_from_compact(entry)
    spec["exact"] = bool(spec.get("exact"))
    return spec
from model_graph_accounting import (
    GraphAccounting,
    ModelAccounting,
    PartitionOccurrence,
    _rollup_occurrence_timings,
    build_model_report,
    format_text_report,
)


def _partitioner_supports_skip_horizontal_fusion() -> bool:
    from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner

    return (
        "skip_horizontal_fusion"
        in inspect.signature(CapabilityBasedPartitioner.__init__).parameters
    )


def _touch_repro(canonical_dir: Path, name: str) -> None:
    repro_dir = canonical_dir / name
    repro_dir.mkdir(parents=True)
    (repro_dir / "repro.py").write_text("# repro\n")


def test_count_bytes_deduplicates_output_aliases():
    inp = torch.empty(4, dtype=torch.float32)
    out = torch.empty(16, dtype=torch.float32)

    assert count_bytes_naive([inp], (out, out.view(4, 4), out.reshape(2, 8))) == (
        inp.nelement() + out.nelement()
    ) * inp.element_size()


def test_count_bytes_counts_distinct_outputs():
    inp = torch.empty(4, dtype=torch.float32)
    out0 = torch.empty(16, dtype=torch.float32)
    out1 = torch.empty(16, dtype=torch.float32)

    assert count_bytes_naive([inp], (out0, out1)) == (
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


def test_compute_worker_count_full_graph_defaults_to_one_worker_per_gpu():
    # full_graph compiles can hold large GPU memory -> stay at 1/GPU
    # (deterministic, independent of core count).
    assert _compute_worker_count(
        num_repros=1133,
        num_gpus=2,
        max_workers=None,
        workers_per_gpu=None,
        workload_kind="full_graph",
    ) == (2, 1)


def test_compute_worker_count_kernel_default_packs_by_cores():
    # Kernel/oracle workloads (tiny footprint) default to min(4, cores_per_gpu)
    # workers per GPU. Mirror that formula here so the test is independent of
    # the host's core count.
    import os as _os
    num_gpus = 2
    try:
        n_cores = len(_os.sched_getaffinity(0))
    except AttributeError:
        n_cores = _os.cpu_count() or num_gpus
    expected_per_gpu = max(1, min(4, max(1, n_cores // num_gpus)))
    assert _compute_worker_count(
        num_repros=1133,
        num_gpus=num_gpus,
        max_workers=None,
        workers_per_gpu=None,
    ) == (num_gpus * expected_per_gpu, expected_per_gpu)


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
            ],
        }))

        repros, entries = load_benchmark_set(manifest, canonical_dir=canonical_dir)
        assert entries == 1
        assert repros == [canonical_dir / "foo" / "repro.py"]


def test_load_benchmark_set_rejects_missing_patterns():
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

        try:
            load_benchmark_set(manifest, canonical_dir=canonical_dir)
        except FileNotFoundError as exc:
            assert "missing canonical repros" in str(exc)
            assert "missing" in str(exc)
        else:
            raise AssertionError("missing benchmark-set entries should fail")


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


def test_find_full_graphs_accepts_files_and_directories_with_dedupe():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model_dir = root / "repros" / "models" / "hf" / "infer" / "TinyModel"
        model_dir.mkdir(parents=True)
        graph0 = model_dir / "full_graph_000.py"
        graph1 = model_dir / "full_graph_001.py"
        graph0.write_text("# graph 0\n")
        graph1.write_text("# graph 1\n")
        (model_dir / "repro.py").write_text("# not a full graph\n")

        assert find_full_graphs([graph0, model_dir]) == [graph0, graph1]


def _full_graph_definition(
    *,
    input_specs=None,
    tensor_attrs=None,
    sidecar=None,
) -> FullGraphDefinition:
    return FullGraphDefinition(
        path=Path("full_graph_000.py"),
        graph_cls=torch.nn.Module,
        input_specs=input_specs or [],
        tensor_attrs=tensor_attrs or {},
        forward_takes_no_inputs=False,
        metadata={"source": {}, "sidecar": sidecar or {}},
    )


def test_full_graph_preflight_skips_annotation_only_integer_inputs():
    definition = _full_graph_definition(input_specs=[
        {
            "kind": "tensor",
            "name": "idx",
            "shape": [8],
            "dtype": "int64",
            "gen": {"kind": "index", "low": 0, "high": 8},
            "constraint_source": "annotation_default",
        }
    ])

    reasons = _classify_full_graph_definition(definition)

    assert [reason["category"] for reason in reasons] == ["unsafe_integer_input"]
    assert _classify_full_graph_definition(definition, allow_unsafe=True) == []


def test_full_graph_preflight_allows_graph_inferred_integer_inputs():
    definition = _full_graph_definition(input_specs=[
        {
            "kind": "tensor",
            "name": "idx",
            "shape": [8],
            "dtype": "int64",
            "gen": {"kind": "index", "low": 0, "high": 1024},
            "constraint_source": "graph_inference",
        }
    ], sidecar={"inputs": []})

    assert _classify_full_graph_definition(definition) == []


def test_full_graph_preflight_skips_annotation_symbolic_dims():
    definition = _full_graph_definition(input_specs=[
        {"kind": "symint", "name": "s0", "value": 32},
        {
            "kind": "tensor",
            "name": "x",
            "shape": [2, 32],
            "dtype": "float32",
            "symbolic_dims": [{"dim": 1, "symbol": "s0", "default": 32}],
        },
    ])

    categories = [reason["category"] for reason in _classify_full_graph_definition(definition)]

    assert categories == ["symbolic_dim", "symbolic_dim"]


def test_full_graph_preflight_requires_exact_attrs_even_when_unsafe_allowed():
    definition = _full_graph_definition(tensor_attrs={
        "_tensor_constant0": {
            "kind": "tensor",
            "name": "_tensor_constant0",
            "shape": [4],
            "dtype": "int64",
            "requires_exact": True,
        }
    })

    reasons = _classify_full_graph_definition(definition, allow_unsafe=True)

    assert [reason["category"] for reason in reasons] == ["requires_exact_attr"]


def test_full_graph_preflight_records_skips_without_queueing_gpu_work():
    source = '''
import torch

class GraphModule(torch.nn.Module):
    def forward(self, idx: "i64[4]cpu"):
        return (idx,)
'''
    with tempfile.TemporaryDirectory() as tmp:
        graph = Path(tmp) / "full_graph_000.py"
        graph.write_text(source)

        queueable, failures = _preflight_full_graphs([graph])

    assert queueable == []
    assert failures[str(graph)]["status"] == "skipped"
    assert failures[str(graph)]["category"] == "unsafe_integer_input"


def test_write_results_output_counts_skipped_and_uses_reserved_failure_key():
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "results.json"
        _write_results_output(
            output,
            {},
            {"full_graph_000.py": {"status": "skipped", "category": "unsafe_integer_input"}},
            total=1,
            done=0,
            failed=0,
            skipped=1,
            elapsed=0.0,
            workload_kind="full_graph",
        )

        data = json.loads(output.read_text())

    assert data["__summary__"]["skipped"] == 1
    assert data["__summary__"]["failed"] == 0
    assert data["_metadata"]["workload_kind"] == "full_graph"


def test_full_graph_harness_parses_inputs_and_tensor_attrs():
    content = '''
class GraphModule(torch.nn.Module):
    def forward(self, s0: "Sym(s0)", x: "f32[2, s0][s0, 1]cuda:0", idx: "i64[2]cuda:0"):
        _tensor_constant0: "f32[]" = self._tensor_constant0
        _tensor_constant1: "i64[4]" = self._tensor_constant1
        return (x,)
'''
    inputs = parse_full_graph_inputs(content)
    attrs = parse_full_graph_tensor_attrs(content)

    assert inputs[0]["kind"] == "symint"
    assert inputs[0]["value"] == 32
    assert inputs[0]["symbolic_expr"] == "s0"
    assert inputs[1]["shape"] == (2, 32)
    assert inputs[1]["stride"] == (32, 1)
    assert inputs[1]["device"] == "cuda"
    assert inputs[1]["symbolic_dims"] == [{"dim": 1, "symbol": "s0", "default": 32}]
    assert inputs[1]["symbolic_stride_dims"] == [{"dim": 0, "expr": "s0"}]
    assert inputs[2]["dtype"] == "int64"
    assert inputs[2]["gen"]["kind"] == "index"
    assert inputs[2]["gen"]["high"] == 2
    assert attrs["_tensor_constant0"]["shape"] == ()
    assert attrs["_tensor_constant1"]["generator"] == {"kind": "constant", "value": 0}


def test_full_graph_harness_loads_runnable_cpu_graph():
    source = '''
import torch
from torch import device

class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2, 2]cpu"):
        _tensor_constant0: "f32[]" = self._tensor_constant0
        add: "f32[2, 2]cpu" = torch.ops.aten.add.Tensor(x, _tensor_constant0)
        return (add,)
'''
    with tempfile.TemporaryDirectory() as tmp:
        graph = Path(tmp) / "full_graph_000.py"
        graph.write_text(source)

        instance, inputs, definition = load_full_graph(graph, default_device="cpu")

        assert len(inputs) == 1
        assert inputs[0].shape == (2, 2)
        assert inputs[0].device.type == "cpu"
        assert hasattr(instance, "_tensor_constant0")
        assert definition.path == graph
        with torch.no_grad():
            (out,) = instance(*inputs)
        assert out.shape == (2, 2)


def test_full_graph_harness_zeros_fallback_for_annotation_only_integer_tensor_attrs():
    # Policy (9bb85cdbb): an integer tensor attribute with no exact payload is
    # NO LONGER a hard error. zero is always a valid index, so the harness
    # WARNS and falls back to a zeros buffer, keeping the graph runnable for
    # benchmarking. (Was: raise ValueError "requires exact data".)
    import warnings

    source = '''
import torch

class GraphModule(torch.nn.Module):
    def forward(self):
        _tensor_constant0: "i64[4]" = self._tensor_constant0
        return (_tensor_constant0,)
'''
    with tempfile.TemporaryDirectory() as tmp:
        graph = Path(tmp) / "full_graph_000.py"
        graph.write_text(source)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            instance, _inputs, _definition = load_full_graph(graph, default_device="cpu")

        # Loud about the missing data...
        assert any("requires exact data" in str(w.message) for w in caught), (
            f"expected a 'requires exact data' warning, got {[str(w.message) for w in caught]}"
        )
        # ...but the graph is runnable: the constant is a zeros buffer.
        buf = instance._tensor_constant0
        assert buf.dtype == torch.int64
        assert int(buf.abs().sum().item()) == 0, "zeros fallback expected"


def test_full_graph_sidecar_constraints_override_annotations():
    source = '''
import torch

class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2, 2]cpu"):
        return (x,)
'''
    with tempfile.TemporaryDirectory() as tmp:
        graph = Path(tmp) / "repros" / "models" / "genai" / "MyKernel" / "full_graph_000.py"
        graph.parent.mkdir(parents=True)
        graph.write_text(source)
        (graph.parent / "full_graph_000.meta.json").write_text(json.dumps({
            "schema_version": 1,
            "graph": "full_graph_000.py",
            "source": {
                "kind": "microbenchmark",
                "suite": "genai",
                "mode": None,
                "model": "MyKernel",
                "graph": "full_graph_000.py",
            },
            "inputs": [
                {
                    "kind": "tensor",
                    "name": "x",
                    "shape": [2, 2],
                    "dtype": "float32",
                    "stride": [2, 1],
                    "device": "cpu",
                }
            ],
        }))

        instance, inputs, definition = load_full_graph(graph, default_device="cpu")
        metadata = result_metadata(definition)

        assert inputs[0].shape == (2, 2)
        assert metadata["source"]["kind"] == "microbenchmark"
        assert metadata["constraints_source"] == "sidecar"
        assert metadata["constraints"]["inputs"][0]["shape"] == [2, 2]


def test_full_graph_sidecar_validation_rejects_annotation_mismatch():
    source = '''
import torch

class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2, 2]cpu"):
        return (x,)
'''
    with tempfile.TemporaryDirectory() as tmp:
        graph = Path(tmp) / "full_graph_000.py"
        graph.write_text(source)
        graph.with_suffix(".meta.json").write_text(json.dumps({
            "schema_version": 1,
            "graph": "full_graph_000.py",
            "inputs": [
                {
                    "kind": "tensor",
                    "name": "x",
                    "shape": [2, 2],
                    "dtype": "int64",
                    "stride": [2, 1],
                    "device": "cpu",
                }
            ],
        }))

        try:
            load_full_graph(graph, default_device="cpu")
        except ValueError as exc:
            assert "dtype does not match" in str(exc)
        else:
            raise AssertionError("sidecar dtype mismatch should fail")


def test_full_graph_sidecar_allows_symbolic_shape_and_stride_expressions():
    source = '''
import torch

class GraphModule(torch.nn.Module):
    def forward(
        self,
        s0: "Sym(s0)",
        s1: "Sym(s1)",
        s2: "Sym(s2)",
        x: "f32[s0, s1, s2][s1*s2, s2, 1]cpu",
        y: "f32[s0*s1, s2][s2, 1]cpu",
    ):
        return (x, y)
'''
    with tempfile.TemporaryDirectory() as tmp:
        graph = Path(tmp) / "full_graph_000.py"
        graph.write_text(source)
        graph.with_suffix(".meta.json").write_text(json.dumps({
            "schema_version": 1,
            "graph": "full_graph_000.py",
            "inputs": [
                {"kind": "symint", "name": "s0", "value": 2},
                {"kind": "symint", "name": "s1", "value": 3},
                {"kind": "symint", "name": "s2", "value": 4},
                {
                    "kind": "tensor",
                    "name": "x",
                    "shape": [2, 3, 4],
                    "dtype": "float32",
                    "stride": [12, 4, 1],
                    "device": "cpu",
                },
                {
                    "kind": "tensor",
                    "name": "y",
                    "shape": [6, 4],
                    "dtype": "float32",
                    "stride": [4, 1],
                    "device": "cpu",
                },
            ],
        }))

        _instance, inputs, _definition = load_full_graph(
            graph, default_device="cpu"
        )

    assert inputs[3].shape == (2, 3, 4)
    assert inputs[3].stride() == (12, 4, 1)
    assert inputs[4].shape == (6, 4)


def test_full_graph_sidecar_rejects_symbolic_annotation_stride_mismatch():
    source = '''
import torch

class GraphModule(torch.nn.Module):
    def forward(self, s0: "Sym(s0)", x: "f32[2, s0][s0, 1]cpu"):
        return (x,)
'''
    with tempfile.TemporaryDirectory() as tmp:
        graph = Path(tmp) / "full_graph_000.py"
        graph.write_text(source)
        graph.with_suffix(".meta.json").write_text(json.dumps({
            "schema_version": 1,
            "graph": "full_graph_000.py",
            "inputs": [
                {"kind": "symint", "name": "s0", "value": 8},
                {
                    "kind": "tensor",
                    "name": "x",
                    "shape": [2, 8],
                    "dtype": "float32",
                    "stride": [9, 1],
                    "device": "cpu",
                },
            ],
        }))

        try:
            load_full_graph(graph, default_device="cpu")
        except ValueError as exc:
            assert "stride does not match" in str(exc)
        else:
            raise AssertionError("sidecar symbolic stride mismatch should fail")


def test_full_graph_sidecar_rejects_symbolic_annotation_shape_mismatch():
    source = '''
import torch

class GraphModule(torch.nn.Module):
    def forward(self, s0: "Sym(s0)", x: "f32[s0][1]cpu"):
        return (x,)
'''
    with tempfile.TemporaryDirectory() as tmp:
        graph = Path(tmp) / "full_graph_000.py"
        graph.write_text(source)
        graph.with_suffix(".meta.json").write_text(json.dumps({
            "schema_version": 1,
            "graph": "full_graph_000.py",
            "inputs": [
                {"kind": "symint", "name": "s0", "value": 8},
                {
                    "kind": "tensor",
                    "name": "x",
                    "shape": [9],
                    "dtype": "float32",
                    "stride": [1],
                    "device": "cpu",
                },
            ],
        }))

        try:
            load_full_graph(graph, default_device="cpu")
        except ValueError as exc:
            assert "shape does not match" in str(exc)
        else:
            raise AssertionError("sidecar symbolic shape mismatch should fail")


def test_full_graph_sidecar_rejects_unresolved_symbolic_annotation_stride():
    source = '''
import torch

class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[2, 8][unknown, 1]cpu"):
        return (x,)
'''
    with tempfile.TemporaryDirectory() as tmp:
        graph = Path(tmp) / "full_graph_000.py"
        graph.write_text(source)
        graph.with_suffix(".meta.json").write_text(json.dumps({
            "schema_version": 1,
            "graph": "full_graph_000.py",
            "inputs": [{
                "kind": "tensor",
                "name": "x",
                "shape": [2, 8],
                "dtype": "float32",
                "stride": [8, 1],
                "device": "cpu",
            }],
        }))

        try:
            load_full_graph(graph, default_device="cpu")
        except ValueError as exc:
            assert "unbound symbols" in str(exc)
        else:
            raise AssertionError("unresolved symbolic stride should fail")


def test_full_graph_sidecar_rejects_conflicting_symbol_bindings():
    source = '''
import torch

class GraphModule(torch.nn.Module):
    def forward(self, first: "Sym(s0)", second: "Sym(s0)"):
        return (first, second)
'''
    with tempfile.TemporaryDirectory() as tmp:
        graph = Path(tmp) / "full_graph_000.py"
        graph.write_text(source)
        graph.with_suffix(".meta.json").write_text(json.dumps({
            "schema_version": 1,
            "graph": "full_graph_000.py",
            "inputs": [
                {"kind": "symint", "name": "first", "value": 2},
                {"kind": "symint", "name": "second", "value": 3},
            ],
        }))

        try:
            load_full_graph(graph, default_device="cpu")
        except ValueError as exc:
            assert "conflicting values" in str(exc)
        else:
            raise AssertionError("conflicting symbol bindings should fail")


def test_full_graph_source_inference_distinguishes_models_and_microbenchmarks():
    model_source = infer_full_graph_source(
        Path("repros/models/hf/infer/BertForMaskedLM/full_graph_000.py")
    )
    genai_source = infer_full_graph_source(
        Path("repros/models/genai/CrossEntropyForward/full_graph_000.py")
    )

    assert model_source["kind"] == "model"
    assert model_source["suite"] == "hf"
    assert model_source["mode"] == "infer"
    assert model_source["model"] == "BertForMaskedLM"
    assert genai_source["kind"] == "microbenchmark"
    assert genai_source["model"] == "CrossEntropyForward"


def test_write_full_graph_metadata_exports_input_constraints():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(2, 3, dtype=torch.int64)
    out = graph.call_function(torch.ops.aten.sin.default, (x,))
    out.meta["val"] = torch.empty(2, 3, dtype=torch.float32)
    graph.output(out)
    gm = torch.fx.GraphModule({}, graph)

    with tempfile.TemporaryDirectory() as tmp:
        graph_path = Path(tmp) / "full_graph_000.py"
        graph_path.write_text("# graph\n")
        meta_path = write_full_graph_metadata(graph_path, gm)
        payload = json.loads(meta_path.read_text())

    constraints = graph_constraints_from_gm(gm)
    inp0 = _decode_meta_input(payload["inputs"][0])
    assert inp0["name"] == "x"
    assert inp0["shape"] == [2, 3]
    assert inp0["dtype"] == "int64"
    assert inp0["exact"] is False
    assert constraints["outputs"][0]["shape"] == [2, 3]


def test_write_full_graph_metadata_exports_exact_get_attr_payloads():
    graph = torch.fx.Graph()
    const = graph.get_attr("const")
    const.meta["val"] = torch.tensor([3, 5], dtype=torch.int64)
    graph.output(const)
    gm = torch.fx.GraphModule({"const": torch.tensor([3, 5], dtype=torch.int64)}, graph)

    with tempfile.TemporaryDirectory() as tmp:
        graph_path = Path(tmp) / "full_graph_000.py"
        graph_path.write_text("# graph\n")
        payload = json.loads(write_full_graph_metadata(graph_path, gm).read_text())

    const = _decode_meta_input(payload["tensor_attrs"]["const"])
    assert const["exact"] is True
    assert const["data"] == [3, 5]


def test_write_full_graph_metadata_marks_meta_integer_attrs_requires_exact():
    graph = torch.fx.Graph()
    const = graph.get_attr("const")
    const.meta["val"] = torch.empty(2, dtype=torch.int64, device="meta")
    graph.output(const)
    gm = torch.fx.GraphModule({"const": torch.empty(2, dtype=torch.int64, device="meta")}, graph)

    with tempfile.TemporaryDirectory() as tmp:
        graph_path = Path(tmp) / "full_graph_000.py"
        graph_path.write_text("# graph\n")
        payload = json.loads(write_full_graph_metadata(graph_path, gm).read_text())

    const = _decode_meta_input(payload["tensor_attrs"]["const"])
    assert const["exact"] is False
    assert const["requires_exact"] is True


def test_write_full_graph_metadata_exports_index_and_permutation_constraints():
    graph = torch.fx.Graph()
    idx = graph.placeholder("idx")
    idx.meta["val"] = torch.empty(8, dtype=torch.int64)
    token = graph.placeholder("token")
    token.meta["val"] = torch.empty(16, dtype=torch.int64)
    graph.output((idx, token))
    gm = torch.fx.GraphModule({}, graph)

    with tempfile.TemporaryDirectory() as tmp:
        graph_path = Path(tmp) / "full_graph_000.py"
        graph_path.write_text("# graph\n")
        meta_path = write_full_graph_metadata(
            graph_path,
            gm,
            index_bounds={"token": 2048},
            permutation_indices={"idx": 8},
        )
        payload = json.loads(meta_path.read_text())

    assert _decode_meta_input(payload["inputs"][0])["gen"] == {"kind": "permutation", "size": 8}
    assert _decode_meta_input(payload["inputs"][1])["gen"] == {"kind": "index", "low": 0, "high": 2048}


def test_bench_report_flattens_full_graph_and_named_shape_results():
    with tempfile.TemporaryDirectory() as tmp:
        results_path = Path(tmp) / "results.json"
        results_path.write_text(json.dumps({
            "_metadata": {"workload_kind": "full_graph"},
            "/tmp/repros/models/genai/Foo/full_graph_000.py": {
                "__graph__": {
                    "source": {
                        "kind": "microbenchmark",
                        "suite": "genai",
                        "model": "Foo",
                        "graph": "full_graph_000.py",
                    }
                },
                "default": {"compiled_us": 10.0},
                "_shape": {"compiled_us": 11.0},
            },
            "__summary__": {"ok": 1},
        }))

        loaded = load_report_results(results_path)

    assert "microbenchmark/genai/Foo/full_graph_000.py" in loaded
    assert "microbenchmark/genai/Foo/full_graph_000.py[_shape]" in loaded


def test_make_tensor_from_spec_honors_exact_data_stride_and_offset_on_cpu():
    tensor = make_tensor_from_spec(
        {
            "kind": "tensor",
            "shape": [2],
            "dtype": "int64",
            "stride": [2],
            "device": "cpu",
            "storage_offset": 1,
            "exact": True,
            "data": [7, 9],
        },
        default_device="cpu",
    )

    assert tensor.tolist() == [7, 9]
    assert tensor.stride() == (2,)
    assert tensor.storage_offset() == 1


def test_make_tensor_from_spec_uses_kernel_like_index_and_permutation_generators():
    indexed = make_tensor_from_spec(
        {
            "kind": "tensor",
            "shape": [128],
            "dtype": "int64",
            "device": "cpu",
            "gen": {"kind": "index", "low": 3, "high": 7},
        },
        default_device="cpu",
    )
    perm = make_tensor_from_spec(
        {
            "kind": "tensor",
            "shape": [2, 4],
            "dtype": "int64",
            "device": "cpu",
            "gen": {"kind": "permutation", "size": 4},
        },
        default_device="cpu",
    )

    assert int(indexed.min()) >= 3
    assert int(indexed.max()) < 7
    for row in perm.reshape(-1, 4):
        assert sorted(row.tolist()) == list(range(4))


def test_capture_hook_exports_full_graph_constraints_sidecar():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(4, dtype=torch.float32)
    out = graph.call_function(torch.ops.aten.relu.default, (x,))
    out.meta["val"] = torch.empty(4, dtype=torch.float32)
    graph.output(out)
    gm = torch.fx.GraphModule({}, graph)

    with tempfile.TemporaryDirectory() as tmp:
        state = _CaptureState(
            str(Path(tmp) / "capture"),
            label="sidecar_test",
            graph_dir=str(Path(tmp) / "repros" / "models" / "hf" / "infer" / "TinyModel"),
            validate=False,
        )
        state.process_graph(gm)

        meta_path = (
            Path(tmp)
            / "repros"
            / "models"
            / "hf"
            / "infer"
            / "TinyModel"
            / "full_graph_000.meta.json"
        )
        payload = json.loads(meta_path.read_text())

    assert payload["source"]["kind"] == "model"
    assert payload["source"]["model"] == "TinyModel"
    inp0 = _decode_meta_input(payload["inputs"][0])
    assert inp0["shape"] == [4]
    assert inp0["dtype"] == "float32"


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


def test_make_inputs_from_config_honors_batched_permutation_generator_on_cpu():
    (perm,) = make_inputs_from_config({
        "inputs": [
            {
                "kind": "tensor",
                "shape": [2, 3, 4],
                "dtype": "torch.int64",
                "stride": None,
                "device": "cpu",
                "gen": {"kind": "permutation", "size": 4},
            },
        ],
    })

    assert perm.shape == (2, 3, 4)
    for row in perm.reshape(-1, 4):
        assert sorted(row.tolist()) == list(range(4))


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
    bounds = infer_index_bounds_from_gm(gm, {
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
    bounds = infer_index_bounds_from_gm(gm, {
        "token_types": {"dtype": "torch.int64"},
        "positions": {"dtype": "torch.int64"},
        "table": {"dtype": "torch.float32"},
    })

    assert bounds["token_types"] == 2
    assert bounds["positions"] == 512


def test_capture_hook_emits_permutation_config_for_inverse_permutation_indices():
    graph = torch.fx.Graph()
    idx = graph.placeholder("idx")
    idx.meta["val"] = torch.empty(8, dtype=torch.int64)
    empty = graph.call_function(
        torch.ops.aten.empty.memory_format,
        ([8],),
        {
            "dtype": torch.int64,
            "layout": torch.strided,
            "device": torch.device("cuda"),
            "pin_memory": False,
        },
    )
    empty.meta["val"] = torch.empty(8, dtype=torch.int64)
    iota = graph.call_function(
        torch.ops.prims.iota.default,
        (8,),
        {
            "start": 0,
            "step": 1,
            "dtype": torch.int64,
            "device": torch.device("cuda"),
            "requires_grad": False,
        },
    )
    iota.meta["val"] = torch.empty(8, dtype=torch.int64)
    inverse = graph.call_function(torch.ops.aten.index_put.default, (empty, [idx], iota))
    inverse.meta["val"] = torch.empty(8, dtype=torch.int64)
    graph.output(inverse)

    gm = torch.fx.GraphModule({}, graph)
    with tempfile.TemporaryDirectory() as tmp:
        state = _CaptureState(tmp, validate=False)
        # v3 repros no longer embed the signature in repro.py; _generate_repro_file
        # RETURNS (filepath, signature, compact_inputs, alias_group_nbytes).
        _filepath, signature, _compact, _alias = state._generate_repro_file(
            gm,
            {
                "idx": {
                    "dtype": "torch.int64",
                    "shape": [8],
                    "stride": [],
                    "device": "cuda",
                }
            },
            {"pattern_hash": "pattern", "shape_hash": "shape"},
            "repro.py",
        )

    assert "T([8], i64, gen=Perm(8))" in signature


def test_capture_hook_emits_batched_permutation_config_for_scatter_inverse_indices():
    graph = torch.fx.Graph()
    idx = graph.placeholder("idx")
    idx.meta["val"] = torch.empty(2, 3, 4, dtype=torch.int64)
    empty = graph.call_function(
        torch.ops.aten.empty.memory_format,
        ([2, 3, 4],),
        {
            "dtype": torch.int64,
            "layout": torch.strided,
            "device": torch.device("cuda"),
            "pin_memory": False,
        },
    )
    empty.meta["val"] = torch.empty(2, 3, 4, dtype=torch.int64)
    iota = graph.call_function(
        torch.ops.prims.iota.default,
        (4,),
        {
            "start": 0,
            "step": 1,
            "dtype": torch.int64,
            "device": torch.device("cuda"),
            "requires_grad": False,
        },
    )
    iota.meta["val"] = torch.empty(4, dtype=torch.int64)
    viewed = graph.call_function(torch.ops.aten.view.default, (iota, [1, 1, -1]))
    viewed.meta["val"] = torch.empty(1, 1, 4, dtype=torch.int64)
    expanded = graph.call_function(torch.ops.aten.expand.default, (viewed, [2, 3, 4]))
    expanded.meta["val"] = torch.empty(2, 3, 4, dtype=torch.int64)
    inverse = graph.call_function(
        torch.ops.aten.scatter.src,
        (empty, -1, idx, expanded),
    )
    inverse.meta["val"] = torch.empty(2, 3, 4, dtype=torch.int64)
    graph.output(inverse)

    gm = torch.fx.GraphModule({}, graph)
    with tempfile.TemporaryDirectory() as tmp:
        state = _CaptureState(tmp, validate=False)
        _filepath, signature, _compact, _alias = state._generate_repro_file(
            gm,
            {
                "idx": {
                    "dtype": "torch.int64",
                    "shape": [2, 3, 4],
                    "stride": [],
                    "device": "cuda",
                }
            },
            {"pattern_hash": "pattern", "shape_hash": "shape"},
            "repro.py",
        )

    assert "T([2, 3, 4], i64, gen=Perm(4))" in signature


def _real_targets(nodes):
    return {
        node.target
        for node in nodes
        if node.op == "call_function"
        and node.target
        not in {
            operator.getitem,
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten.permute.default,
            torch.ops.aten.slice.Tensor,
            torch.ops.aten.unsqueeze.default,
            torch.ops.aten.squeeze.default,
            torch.ops.aten.squeeze.dim,
            torch.ops.aten.expand.default,
            torch.ops.aten.t.default,
            torch.ops.aten.transpose.int,
            torch.ops.aten.select.int,
            torch.ops.aten.as_strided.default,
        }
    }


class _RecordingCaptureState(_CaptureState):
    def __init__(self, output_dir):
        super().__init__(output_dir, validate=False)
        self.generated_targets = []

    def _generate_repro_file(self, gm, placeholder_info, meta, filename, shape_params=None):
        self.generated_targets.append(_real_targets(gm.graph.nodes))
        return str(Path(self.output_dir) / filename)


def test_capture_hook_process_graph_splits_horizontal_fusion_regions():
    if not _partitioner_supports_skip_horizontal_fusion():
        print("Skipping horizontal fusion capture test; installed PyTorch lacks pytorch/pytorch#170191")
        return

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(4)
    unsupported = graph.call_method("relu", (x,))
    unsupported.meta["val"] = torch.empty(4)
    left = graph.call_function(torch.ops.aten.sin.default, (unsupported,))
    left.meta["val"] = torch.empty(4)
    right = graph.call_function(torch.ops.aten.cos.default, (unsupported,))
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


def test_capture_hook_process_graph_keeps_getitem_data_dependency_together():
    if not _partitioner_supports_skip_horizontal_fusion():
        print("Skipping getitem capture test; installed PyTorch lacks pytorch/pytorch#170191")
        return

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(4, 4)
    values_and_indices = graph.call_function(torch.ops.aten.max.dim, (x, 1))
    values_and_indices.meta["val"] = (
        torch.empty(4),
        torch.empty(4, dtype=torch.int64),
    )
    values = graph.call_function(operator.getitem, (values_and_indices, 0))
    values.meta["val"] = torch.empty(4)
    out = graph.call_function(torch.ops.aten.sin.default, (values,))
    out.meta["val"] = torch.empty(4)
    graph.output(out)

    gm = torch.fx.GraphModule({}, graph)
    with tempfile.TemporaryDirectory() as tmp:
        state = _RecordingCaptureState(tmp)
        state.process_graph(gm)

    assert state.generated_targets == [
        {torch.ops.aten.max.dim, torch.ops.aten.sin.default},
    ]


def _occ(pattern_hash, shape_hash, n_ops=2):
    return PartitionOccurrence(
        graph_name="full_graph_000.py",
        pattern_hash=pattern_hash,
        shape_hash=shape_hash,
        n_ops=n_ops,
        op_counts={"aten.add.Tensor": 1},
        origin_ops=["aten.add.Tensor"],
        input_shapes=["[4]:f32"],
        max_output_shape=[4],
        has_reduction=False,
    )


def _model_accounting(occs):
    ga = GraphAccounting(graph_path="full_graph_000.py", graph_name="full_graph_000.py")
    ga.occurrences = list(occs)
    acc = ModelAccounting(model_name="TinyModel", model_dir="/tmp/TinyModel")
    acc.graphs = [ga]
    return acc


def test_rollup_classifies_matched_fallback_and_unpriced_occurrences():
    timing = {
        "oracle_us": 10.0,
        "compile_us": 9.0,
        "points_by_shape": {"shapeA": {"oracle_us": 5.0, "compile_us": 4.0}},
    }
    occs = [
        _occ("p", "shapeA"),   # matched at own shape
        _occ("p", "shapeB"),   # no point -> dir median fallback
    ]
    roll = _rollup_occurrence_timings(timing, occs)

    assert roll["matched_occ"] == 1
    assert roll["fallback_occ"] == 1
    assert roll["unpriced_occ"] == 0
    assert roll["total_occ"] == 2
    assert roll["mode"] == "partial"
    # 5.0 (own shape) + 10.0 (dir median) = 15.0
    assert roll["oracle_us_total"] == 15.0


def test_rollup_untimed_dir_marks_all_occurrences_unpriced():
    occs = [_occ("p", "shapeA"), _occ("p", "shapeB")]
    roll = _rollup_occurrence_timings(None, occs)

    assert roll["matched_occ"] == 0
    assert roll["fallback_occ"] == 0
    assert roll["unpriced_occ"] == 2
    assert roll["total_occ"] == 2
    assert roll["mode"] == "untimed"
    assert roll["oracle_us_total"] is None


def test_rollup_timed_dir_without_representative_is_unpriced_for_missing_shapes():
    # Dir has a per-shape point for shapeA but no representative oracle_us, so a
    # shapeB occurrence has nothing to fall back to -> unpriced (not silently 0).
    timing = {
        "points_by_shape": {"shapeA": {"oracle_us": 5.0, "compile_us": 4.0}},
    }
    occs = [_occ("p", "shapeA"), _occ("p", "shapeB")]
    roll = _rollup_occurrence_timings(timing, occs)

    assert roll["matched_occ"] == 1
    assert roll["fallback_occ"] == 0
    assert roll["unpriced_occ"] == 1
    assert roll["mode"] == "partial"
    assert roll["oracle_us_total"] == 5.0


def test_build_model_report_gates_projection_when_kernels_unpriced():
    # Pattern "priced" is timed; pattern "untimed" has no timing entry at all.
    acc = _model_accounting([
        _occ("priced", "shapeA"),
        _occ("priced", "shapeA"),
        _occ("untimed", "shapeC"),
    ])
    hash_to_repro = {"priced": "pointwise_priced", "untimed": "pointwise_untimed"}
    timings = {
        "pointwise_priced": {
            "oracle_us": 7.0,
            "compile_us": 6.0,
            "points_by_shape": {"shapeA": {"oracle_us": 7.0, "compile_us": 6.0}},
        },
        # pointwise_untimed deliberately ABSENT -> untimed.
    }
    report = build_model_report(acc, hash_to_repro, timings, {})

    assert report["projection_complete"] is False
    # Headline floor must NOT be a number when coverage is partial.
    assert report["fusible_oracle_us_total"] is None
    assert report["fusible_compile_us_total"] is None
    assert report["fusible_oracle_us_total_is_lower_bound"] is True
    # The lower-bound partial sum is still exposed for diagnostics (2 x 7.0).
    assert report["fusible_oracle_us_total_partial"] == 14.0
    assert report["unpriced_occurrences"] == 1
    assert report["priced_occurrences"] == 2
    assert report["total_occurrences"] == 3
    unpriced = report["unpriced_kernels"]
    assert len(unpriced) == 1
    assert unpriced[0]["pattern_hash"] == "untimed"
    assert unpriced[0]["unpriced_occ"] == 1
    # Text report surfaces INCOMPLETE rather than a number.
    text = format_text_report(report)
    assert "INCOMPLETE" in text
    assert "UNAVAILABLE" in text


def test_build_model_report_complete_projection_reports_number_unchanged():
    acc = _model_accounting([
        _occ("priced", "shapeA"),
        _occ("priced", "shapeA"),
    ])
    hash_to_repro = {"priced": "pointwise_priced"}
    timings = {
        "pointwise_priced": {
            "oracle_us": 7.0,
            "compile_us": 6.0,
            "points_by_shape": {"shapeA": {"oracle_us": 7.0, "compile_us": 6.0}},
        },
    }
    report = build_model_report(acc, hash_to_repro, timings, {})

    assert report["projection_complete"] is True
    assert report["fusible_oracle_us_total"] == 14.0
    assert report["fusible_oracle_us_total_partial"] == 14.0
    assert report["fusible_oracle_us_total_is_lower_bound"] is False
    assert report["unpriced_occurrences"] == 0
    assert report["unpriced_kernels"] == []


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
    test_find_full_graphs_accepts_files_and_directories_with_dedupe()
    test_full_graph_harness_parses_inputs_and_tensor_attrs()
    test_full_graph_harness_loads_runnable_cpu_graph()
    test_full_graph_harness_rejects_annotation_only_integer_tensor_attrs()
    test_full_graph_sidecar_constraints_override_annotations()
    test_full_graph_sidecar_validation_rejects_annotation_mismatch()
    test_full_graph_source_inference_distinguishes_models_and_microbenchmarks()
    test_write_full_graph_metadata_exports_input_constraints()
    test_write_full_graph_metadata_exports_exact_get_attr_payloads()
    test_write_full_graph_metadata_marks_meta_integer_attrs_requires_exact()
    test_write_full_graph_metadata_exports_index_and_permutation_constraints()
    test_bench_report_flattens_full_graph_and_named_shape_results()
    test_make_tensor_from_spec_honors_exact_data_stride_and_offset_on_cpu()
    test_make_tensor_from_spec_uses_kernel_like_index_and_permutation_generators()
    test_capture_hook_exports_full_graph_constraints_sidecar()
    test_make_inputs_safely_handles_legacy_bool_randn()
    test_shape_configs_accept_general_input_generators()
    test_make_inputs_from_config_honors_generators_on_cpu()
    test_make_inputs_from_config_honors_batched_permutation_generator_on_cpu()
    test_parse_make_inputs_preserves_generators()
    test_capture_hook_traces_select_to_index_bound()
    test_capture_hook_traces_gathered_values_to_embedding_bound()
    test_capture_hook_emits_permutation_config_for_inverse_permutation_indices()
    test_capture_hook_emits_batched_permutation_config_for_scatter_inverse_indices()
    test_capture_hook_process_graph_splits_horizontal_fusion_regions()
    test_capture_hook_process_graph_keeps_getitem_data_dependency_together()
    test_rollup_classifies_matched_fallback_and_unpriced_occurrences()
    test_rollup_untimed_dir_marks_all_occurrences_unpriced()
    test_rollup_timed_dir_without_representative_is_unpriced_for_missing_shapes()
    test_build_model_report_gates_projection_when_kernels_unpriced()
    test_build_model_report_complete_projection_reports_number_unchanged()
    test_gpu_lock_metadata_marks_lock_mode()
    test_gpu_lock_detects_dead_pid_metadata_while_blocked()
    print("All tests passed.")
