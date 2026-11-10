"""Tiny CPU/Fake integration regressions for normal-partition MM captures.

Generated Python is imported/executed, never parsed as text. All emitted fixtures
live in pytest's temporary directories; these tests never scan/extract the corpus.
"""
from __future__ import annotations

import builtins
from dataclasses import asdict
import importlib.util
import io
import json
from pathlib import Path
import shutil

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

import capture_hook as capture
import full_graph_harness
from input_codec import compact_from_spec, spec_from_compact
from matmul_partitioner import MatmulLimits, get_matmul_regions, is_matmul
from mm_replay import execution_hash
from repro_harness import load_shape_configs
from scripts.curate_mm_corpus import curate_corpus
from scripts.extract_matmul_corpus import load_source_graph
from scripts.mm_corpus_dedup import trace_repro


NORMAL = ("prologue", "epilogue")


def _state(path):
    return capture._CaptureState(str(path), only_include_mm=True, validate=False)


def _capture(state, fn, *args):
    with FakeTensorMode(allow_non_fake_inputs=True):
        gm = make_fx(fn, tracing_mode="fake")(*args)
        regions = get_matmul_regions(gm, capture.partition_node_is_supported, categories=NORMAL)
        observations = state.process_graph(gm, components=regions)
    assert not state.dropped
    assert len(observations) == len(regions)
    return gm, regions, observations


def _config(entry):
    return {"inputs": [spec_from_compact(value) for value in entry["inputs"]],
            "alias_group_nbytes": entry.get("alias_group_nbytes", [])}


def _load(path):
    spec = importlib.util.spec_from_file_location("mm_fixture_" + path.parent.name + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(path, value):
    path.write_text(json.dumps(value))


def _raw_stage(path, state, selections):
    """Give curation the capture's structured data, without legacy text parsers."""
    state.finalize()  # Stamp the actual pre-dedup occurrence counts in entries.
    metrics, rows = {}, []
    for gm, regions, observations in selections:
        row = {"source": "unit_fixture", "original_mm": sum(is_matmul(n) for n in gm.graph.nodes), "regions": []}
        for region, observed in zip(regions, observations, strict=True):
            key = observed["pattern_hash"], observed["shape_hash"]
            metrics[key] = {**region.metrics, "normal_pattern_hash": observed["normal_pattern_hash"]}
            row["regions"].append({"pattern_hash": key[0], "shape_hash": key[1]})
        rows.append(row)
    grouped = {}
    for entry in state.captured:
        pattern = entry["pattern_hash"]
        directory = path / "canonical" / ("_".join(entry["mm_types"]) + "_" + pattern)
        directory.mkdir(parents=True, exist_ok=True)
        if pattern not in grouped:
            # Copy authoritative capture output verbatim; no source parsing.
            shutil.copyfile(entry["file"], directory / "repro.py")
            _json(directory / "meta.json", {"pattern_hash": pattern, "kind": "matmul"})
            grouped[pattern] = directory, []
        point = {"shape_hash": entry["shape_hash"], "inputs": entry["inputs"],
                 "models": {"unit/infer/fixture": {"occurrences": entry["occurrences"]}},
                 "partition_metrics": metrics[(pattern, entry["shape_hash"])]}
        if entry.get("alias_group_nbytes"):
            point["alias_group_nbytes"] = entry["alias_group_nbytes"]
        grouped[pattern][1].append(point)
    for directory, points in grouped.values():
        _json(directory / "shapes.json", {"points": points})
    _json(path / "occurrences.json", rows)
    _json(path / "report.json", {"policy": asdict(MatmulLimits()), "policy_id": "unit-fixture"})
    return path


def test_dry_run_derives_inputs_without_writing_files(tmp_path, monkeypatch):
    state = _state(tmp_path / "capture")
    with FakeTensorMode(allow_non_fake_inputs=True):
        gm = make_fx(lambda x, w: torch.sin(x) @ w, tracing_mode="fake")(
            torch.empty(2, 2, device="cpu"), torch.empty(2, 2, device="cpu"),
        )
        (region,) = get_matmul_regions(gm, capture.partition_node_is_supported, categories=NORMAL)
        pattern = capture.compute_partition_pattern(region, gm)
        before = list((tmp_path / "capture").iterdir())

        def guard(original):
            def open_read_only(file, mode="r", *args, **kwargs):
                assert not any(flag in mode for flag in "wax+"), "dry-run attempted file writing"
                return original(file, mode, *args, **kwargs)
            return open_read_only

        with monkeypatch.context() as patch:
            patch.setattr(builtins, "open", guard(builtins.open))
            patch.setattr(io, "open", guard(io.open))
            filepath, _signature, inputs, sizes = state._generate_repro_file(
                pattern["sub_gm"], pattern["placeholder_info"], {"normal_partition": True},
                "must_not_exist.py", pattern["shape_params"], dry_run=True,
            )
        assert filepath is None
        assert len(inputs) == 2 and sizes == []
        assert all(spec_from_compact(value)["device"] == "cpu" for value in inputs)
        assert list((tmp_path / "capture").iterdir()) == before


def test_same_shape_offset_contexts_survive_raw_capture_and_curation(tmp_path):
    def fn(a, b, w1, w2):
        return (a * 2.) @ w1, (b * 2.) @ w2

    state = _state(tmp_path / "capture")
    selection = _capture(
        state, fn,
        torch.empty(8, device="cpu").as_strided((2, 2), (2, 1), 0),
        torch.empty(8, device="cpu").as_strided((2, 2), (2, 1), 4),
        torch.empty(2, 2, device="cpu"), torch.empty(2, 2, device="cpu"),
    )
    _gm, regions, observations = selection
    assert [r.category for r in regions] == ["prologue", "prologue"]
    first, second = observations
    assert first["pattern_hash"] == second["pattern_hash"]
    assert first["normal_pattern_hash"] == second["normal_pattern_hash"]
    assert first["shape_hash"] != second["shape_hash"]
    assert all(len(item["shape_hash"]) == 8 for item in observations)
    assert len(state.captured) == 2
    offsets = {max(spec.get("storage_offset", 0) for spec in _config(entry)["inputs"])
               for entry in state.captured}
    assert offsets == {0, 4}

    raw = _raw_stage(tmp_path / "raw", state, [selection])
    curated = tmp_path / "curated"
    report = curate_corpus(raw, curated)
    assert report["unique_repros"] == 1 and report["shape_points"] == 2
    (path,) = (curated / "canonical" / "prologue").glob("*/repro.py")
    module = _load(path)
    configs = load_shape_configs(str(path))
    assert len(configs) == 2
    replay_offsets = set()
    for config in configs.values():
        args = module.make_inputs(shape_config=config)
        assert all(value.device == torch.device("cpu") for value in args)
        replay_offsets.add(args[0].storage_offset())
        torch.testing.assert_close(module.Repro()(*args), (args[0] * 2.) @ args[1])
    assert replay_offsets == {0, 4}


@pytest.mark.parametrize("category", NORMAL)
def test_cpu_devices_survive_capture_and_generated_input_adapter(tmp_path, category):
    fn = (lambda x, w: torch.sin(x) @ w) if category == "prologue" else (lambda x, w: torch.sin(x @ w))
    state = _state(tmp_path / "capture")
    _capture(state, fn, torch.empty(2, 2, device="cpu"), torch.empty(2, 2, device="cpu"))
    (entry,) = state.captured
    assert entry["matmul"]["category"] == category
    config = _config(entry)
    assert all(spec["device"] == "cpu" for spec in config["inputs"])
    module = _load(Path(entry["file"]))
    args = module.make_inputs(shape_config=config)
    assert all(value.device == torch.device("cpu") for value in args)
    result = module.Repro()(*args)
    assert result.device == torch.device("cpu")
    expected = torch.sin(args[0]) @ args[1] if category == "prologue" else torch.sin(args[0] @ args[1])
    torch.testing.assert_close(result, expected)


def test_structured_source_restores_aliases_and_preserves_existing_dotted_attr(tmp_path, monkeypatch):
    source = tmp_path / "full_graph_fixture.py"
    # An authored source fixture, not a parsed/rebuilt rendering of a graph.
    source.write_text('''import torch
class Source(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sub = torch.nn.Module()
        self.sub.register_buffer("fixed", torch.tensor([[1., 2.], [3., 4.]], device="cpu"))
    def forward(self, a, b, w):
        return a, b, (a + b + self.sub.fixed) @ w
''')
    inputs = [{"kind": "tensor", "name": name, "shape": [2, 2], "stride": [2, 1],
               "dtype": "float32", "device": "cpu", "storage_offset": 1 if name == "b" else 0}
              for name in ("a", "b", "w")]
    fixed = {"kind": "tensor", "name": "sub.fixed", "shape": [2, 2], "stride": [2, 1],
             "dtype": "float32", "device": "cpu", "exact": True, "data": [[1., 2.], [3., 4.]]}
    _json(source.with_suffix(".meta.json"), {
        "schema_version": 2,
        "inputs": [compact_from_spec(spec, include_name=True) for spec in inputs],
        "storage_groups": [["a", "b"]],
        "tensor_attrs": {"sub.fixed": compact_from_spec(fixed, include_name=True)},
    })
    original = full_graph_harness.make_tensor_from_spec
    rebuilt = []

    def record_reconstruction(spec, **kwargs):
        rebuilt.append(spec.get("name"))
        return original(spec, **kwargs)

    monkeypatch.setattr(full_graph_harness, "make_tensor_from_spec", record_reconstruction)
    gm = load_source_graph(source)
    a, b, w = gm.graph.find_nodes(op="placeholder")
    assert "sub.fixed" not in rebuilt  # The original class's buffer wins.
    assert torch._C._is_alias_of(a.meta["val"], b.meta["val"])
    assert not torch._C._is_alias_of(a.meta["val"], w.meta["val"])
    assert (a.meta["val"].storage_offset(), b.meta["val"].storage_offset()) == (0, 1)
    assert all(n.meta["val"].device == torch.device("cpu") for n in (a, b, w))
    assert any(n.op == "get_attr" for n in gm.graph.nodes)


def test_opposite_ordered_literals_remain_distinct_executable_variants(tmp_path):
    def first(x, w):
        t = (x @ w).relu()
        a, b = t + 1., t + 2.
        return a, b

    def reversed_(x, w):
        t = (x @ w).relu()
        b, a = t + 2., t + 1.
        return b, a

    state = _state(tmp_path / "capture")
    args = torch.empty(2, 2, device="cpu"), torch.empty(2, 2, device="cpu")
    selections = [_capture(state, fn, *args) for fn in (first, first, reversed_)]
    first_observed = selections[0][2][0]
    reversed_observed = selections[2][2][0]
    assert first_observed["normal_pattern_hash"] == reversed_observed["normal_pattern_hash"]
    assert first_observed["pattern_hash"] != reversed_observed["pattern_hash"]
    assert len(state.captured) == 2
    for entry, expected in zip(state.captured, ((1., 2.), (2., 1.)), strict=True):
        module = _load(Path(entry["file"]))
        values = module.Repro()(torch.zeros(2, 2, device="cpu"), torch.eye(2, device="cpu"))
        assert isinstance(values, tuple) and len(values) == 2
        for value, literal in zip(values, expected, strict=True):
            torch.testing.assert_close(value, torch.full((2, 2), literal, device="cpu"))

    raw = _raw_stage(tmp_path / "raw", state, selections)
    curated = tmp_path / "curated"
    report = curate_corpus(raw, curated)
    assert report["family_members"] == 2 and report["unique_repros"] == 1
    (path,) = (curated / "canonical" / "epilogue").glob("*/repro.py")
    module = _load(path)
    # Curation must choose the twice-observed variant, not mix the other code/points.
    values = module.Repro()(torch.zeros(2, 2, device="cpu"), torch.eye(2, device="cpu"))
    for value, literal in zip(values, (1., 2.), strict=True):
        torch.testing.assert_close(value, torch.full((2, 2), literal, device="cpu"))
    metadata = json.loads(path.with_name("meta.json").read_text())
    assert metadata["pattern_hash_mode"] == "ordered_fx_v1"
    assert metadata["pattern_family"] == first_observed["normal_pattern_hash"]
    (config,) = load_shape_configs(str(path)).values()
    with FakeTensorMode(allow_non_fake_inputs=True):
        retraced, _infos = trace_repro(path, config)
        retraced, _shapes = capture.lift_shape_params(retraced)
        assert execution_hash(retraced) == metadata["pattern_hash"]
