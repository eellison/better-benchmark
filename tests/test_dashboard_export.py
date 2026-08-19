import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_model_accounting import (
    build_artifact,
    validate_git_occurrence_manifest,
)
from scripts.dashboard_export import _dtype, export_records, load_kernel_points
from scripts.perf_ab_rollup import rollup_models


def _write_sweep(tmp_path: Path) -> tuple[Path, str, str]:
    pattern_hash = "abc123def456"
    shape_hash = "1234abcd"
    repro_dir = tmp_path / f"pointwise_{pattern_hash}"
    repro_dir.mkdir()
    repro = repro_dir / "repro.py"
    repro.write_text("# synthetic repro\n")
    (repro_dir / "shapes.json").write_text(
        json.dumps(
            {
                "points": [
                    {
                        "shape_hash": shape_hash,
                        "models": {"timm/infer/resnet18": {"occurrences": 2}},
                        "inputs": [[[16, 16], "bf16"], [[16, 16], "f32"]],
                    }
                ]
            }
        )
    )
    sweep = tmp_path / "sweep.json"
    sweep.write_text(
        json.dumps(
            {
                str(repro): {
                    f"resnet18_{shape_hash}": {
                        "compiled_us": 10.0,
                        "coord_descent_us": 8.0,
                        "gap_default": 1.5,
                        "gap_cd": 1.25,
                    }
                },
                "__failures__": {},
            }
        )
    )
    return sweep, pattern_hash, shape_hash


def _write_accounting(
    tmp_path: Path,
    pattern_hash: str,
    shape_hash: str,
    *,
    missing: bool = False,
) -> Path:
    accounting = tmp_path / "model_accounting.json"
    accounting.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "synthetic",
                "hardware": "NVIDIA B200",
                "models": {
                    "timm/infer/resnet18": {
                        "suite": "timm",
                        "mode": "infer",
                        "model": "resnet18",
                        "fusible": {
                            pattern_hash: {"missing" if missing else shape_hash: 2}
                        },
                        "extern_total_us": 5.0,
                        "unpriced_extern_occurrences": 0,
                        "trace_errors": [],
                    }
                },
            }
        )
    )
    return accounting


def test_export_records_contains_stable_kernel_and_projected_model(tmp_path):
    sweep, pattern_hash, shape_hash = _write_sweep(tmp_path)
    accounting = _write_accounting(tmp_path, pattern_hash, shape_hash)

    records = export_records(
        sweep,
        model_accounting=accounting,
        device="cuda",
        arch="NVIDIA B200",
    )

    by_metric = {record["metric"]["name"]: record for record in records}
    assert set(by_metric) == {
        "latency_us",
        "gap_vs_sol",
        "model_coverage_ratio",
        "projected_model_latency_us",
    }
    kernel = by_metric["latency_us"]
    assert kernel["model"]["name"] == f"pointwise_{pattern_hash}[{shape_hash}]"
    assert kernel["benchmark"]["dtype"] == "bfloat16"
    assert kernel["benchmark"]["mode"] == "infer"
    assert kernel["benchmark"]["extra_info"]["record_type"] == "kernel"
    assert kernel["benchmark"]["extra_info"]["pattern_hash"] == pattern_hash
    assert kernel["benchmark"]["extra_info"]["timing"] == "coord_descent_us"
    assert kernel["metric"]["benchmark_values"] == [8.0]

    model = by_metric["projected_model_latency_us"]
    assert model["model"]["name"] == "timm/infer/resnet18"
    assert model["metric"]["benchmark_values"] == [21.0]
    assert model["benchmark"]["extra_info"]["included"] == "true"
    assert (
        model["benchmark"]["dtype"]
        == by_metric["model_coverage_ratio"]["benchmark"]["dtype"]
    )
    assert len(model["benchmark"]["extra_info"]["accounting_digest"]) == 64


def test_export_can_select_compiled_timing(tmp_path):
    sweep, pattern_hash, shape_hash = _write_sweep(tmp_path)
    accounting = _write_accounting(tmp_path, pattern_hash, shape_hash)

    records = export_records(
        sweep,
        model_accounting=accounting,
        device="cuda",
        arch="NVIDIA B200",
        timing="compiled_us",
    )

    by_metric = {record["metric"]["name"]: record for record in records}
    assert by_metric["latency_us"]["metric"]["benchmark_values"] == [10.0]
    assert by_metric["projected_model_latency_us"]["metric"]["benchmark_values"] == [
        25.0
    ]


def test_auto_timing_does_not_fall_back_when_coord_descent_is_invalid(tmp_path):
    sweep, pattern_hash, shape_hash = _write_sweep(tmp_path)
    payload = json.loads(sweep.read_text())
    measurement = next(iter(next(iter(payload.values())).values()))
    measurement["coord_descent_us"] = float("nan")
    sweep.write_text(json.dumps(payload))
    accounting = _write_accounting(tmp_path, pattern_hash, shape_hash)

    with pytest.raises(ValueError, match="no valid kernel points"):
        export_records(
            sweep,
            model_accounting=accounting,
            device="cuda",
            arch="NVIDIA B200",
        )


def test_absolute_projection_matches_perf_ab_rollup(tmp_path):
    base, pattern_hash, shape_hash = _write_sweep(tmp_path)
    accounting = _write_accounting(tmp_path, pattern_hash, shape_hash)
    head = tmp_path / "head.json"
    head_payload = json.loads(base.read_text())
    measurement = next(iter(next(iter(head_payload.values())).values()))
    measurement["compiled_us"] = 5.0
    measurement["coord_descent_us"] = 4.0
    head.write_text(json.dumps(head_payload))

    occurrence_dir = tmp_path / "occurrences"
    occurrence_dir.mkdir()
    (occurrence_dir / "resnet18.json").write_text(
        json.dumps(
            {
                "suite": "timm",
                "mode": "infer",
                "model": "resnet18",
                "fusible": {pattern_hash: {shape_hash: 2}},
                "extern": [
                    {
                        "target": "aten.mm.default",
                        "count": 1,
                        "baseline_us": 5.0,
                    }
                ],
                "trace_errors": [],
            }
        )
    )

    base_records = export_records(
        base,
        model_accounting=accounting,
        device="cuda",
        arch="NVIDIA B200",
    )
    head_records = export_records(
        head,
        model_accounting=accounting,
        device="cuda",
        arch="NVIDIA B200",
    )

    def projection(records):
        return next(
            record["metric"]["benchmark_values"][0]
            for record in records
            if record["metric"]["name"] == "projected_model_latency_us"
        )

    dashboard_reduction = (
        1.0 - projection(head_records) / projection(base_records)
    ) * 100.0
    rollup, _ = rollup_models(base, head, occurrence_dir, timing="auto")
    assert dashboard_reduction == pytest.approx(rollup["timm/infer/resnet18"])


def test_incomplete_model_emits_coverage_but_not_projection(tmp_path):
    sweep, pattern_hash, shape_hash = _write_sweep(tmp_path)
    accounting = _write_accounting(tmp_path, pattern_hash, shape_hash, missing=True)

    records = export_records(
        sweep,
        model_accounting=accounting,
        device="cuda",
        arch="NVIDIA B200",
    )

    model_records = [
        record
        for record in records
        if record["benchmark"]["extra_info"]["record_type"] == "model"
    ]
    assert [record["metric"]["name"] for record in model_records] == [
        "model_coverage_ratio"
    ]
    assert model_records[0]["metric"]["benchmark_values"] == [0.0]
    assert model_records[0]["benchmark"]["extra_info"]["included"] == "false"
    assert (
        model_records[0]["benchmark"]["extra_info"]["exclusion_reasons"]
        == "unmatched_kernel"
    )


def test_hardware_mismatch_fails_export(tmp_path):
    sweep, pattern_hash, shape_hash = _write_sweep(tmp_path)
    accounting = _write_accounting(tmp_path, pattern_hash, shape_hash)

    with pytest.raises(ValueError, match="does not match run architecture"):
        export_records(
            sweep,
            model_accounting=accounting,
            device="cuda",
            arch="NVIDIA H100",
        )


def test_hardware_alias_matches_canonical_device_kind(tmp_path):
    sweep, pattern_hash, shape_hash = _write_sweep(tmp_path)
    accounting = _write_accounting(tmp_path, pattern_hash, shape_hash)
    payload = json.loads(accounting.read_text())
    payload["hardware"] = "B200"
    accounting.write_text(json.dumps(payload))

    records = export_records(
        sweep,
        model_accounting=accounting,
        device="cuda",
        arch="NVIDIA B200",
    )

    assert "projected_model_latency_us" in {
        record["metric"]["name"] for record in records
    }


def test_hardware_gate_preserves_sku_details_and_casefolds(tmp_path):
    sweep, pattern_hash, shape_hash = _write_sweep(tmp_path)
    accounting = _write_accounting(tmp_path, pattern_hash, shape_hash)
    payload = json.loads(accounting.read_text())
    payload["hardware"] = "NVIDIA H100 PCIe"
    accounting.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="does not match run architecture"):
        export_records(
            sweep,
            model_accounting=accounting,
            device="cuda",
            arch="NVIDIA H100 80GB HBM3",
        )

    payload["hardware"] = "mi300x"
    accounting.write_text(json.dumps(payload))
    matching = export_records(
        sweep,
        model_accounting=accounting,
        device="cuda",
        arch="MI300X",
    )
    assert "projected_model_latency_us" in {
        record["metric"]["name"] for record in matching
    }


def test_model_export_rejects_empty_run_architecture(tmp_path):
    sweep, pattern_hash, shape_hash = _write_sweep(tmp_path)
    accounting = _write_accounting(tmp_path, pattern_hash, shape_hash)

    with pytest.raises(ValueError, match="run architecture is required"):
        export_records(
            sweep,
            model_accounting=accounting,
            device="cuda",
            arch="",
        )


def test_export_rejects_full_graph_sweeps_explicitly(tmp_path):
    sweep = tmp_path / "full_graph.json"
    sweep.write_text(
        json.dumps(
            {
                "_metadata": {"workload_kind": "full_graph"},
                "repros/models/timm/infer/model/full_graph_000.py": {
                    "default": {"compiled_us": 10.0}
                },
            }
        )
    )

    with pytest.raises(ValueError, match="requires a canonical kernel sweep"):
        export_records(
            sweep,
            model_accounting=None,
            device="cuda",
            arch="NVIDIA B200",
        )


def test_export_skips_one_invalid_measurement(tmp_path):
    sweep, _, _ = _write_sweep(tmp_path)
    payload = json.loads(sweep.read_text())
    measurements = next(iter(payload.values()))
    measurements["bad_deadbeef"] = {
        "compiled_us": 0,
        "coord_descent_us": float("nan"),
    }
    sweep.write_text(json.dumps(payload))

    records = export_records(
        sweep,
        model_accounting=None,
        device="cuda",
        arch="NVIDIA B200",
    )

    latency_records = [
        record for record in records if record["metric"]["name"] == "latency_us"
    ]
    assert len(latency_records) == 1


def test_sweep_quality_deduplicates_shape_failures_and_counts_metadata_gaps(
    tmp_path,
):
    sweep, _, _ = _write_sweep(tmp_path)
    payload = json.loads(sweep.read_text())
    repro_path = next(key for key in payload if key != "__failures__")
    payload[repro_path]["missing_deadbeef"] = {"compiled_us": 3.0}
    payload["__failures__"] = {
        f"{repro_path}::SHAPE::failed": {"error": "partial failure"},
        "missing/repro.py::SHAPE::failed": {"error": "full failure"},
    }
    sweep.write_text(json.dumps(payload))

    points = load_kernel_points(sweep)
    run_info = next(iter(points.values()))["run_info"]

    assert run_info == {
        "sweep_total_repros": "2",
        "sweep_failed_repros": "1",
        "sweep_invalid_measurements": "0",
        "sweep_missing_shape_files": "0",
        "sweep_unresolved_shape_metadata": "1",
    }


def test_build_artifact_compacts_unchanged_extern_latency():
    records = [
        (
            "model.json",
            {
                "suite": "timm",
                "mode": "infer",
                "model": "resnet18",
                "fusible": {"pattern": {"shape": 3}},
                "extern": [
                    {"target": "aten.mm", "count": 2, "baseline_us": 4.5},
                    {"target": "aten.add", "count": 1, "baseline_us": 2.0},
                ],
                "trace_errors": [],
            },
        )
    ]

    artifact = build_artifact(records, hardware="NVIDIA B200", source="test")

    model = artifact["models"]["timm/infer/resnet18"]
    assert model["extern_total_us"] == 11.0
    assert model["extern_occurrences"] == 3
    assert model["unpriced_extern_occurrences"] == 0


def test_build_artifact_rejects_invalid_counts():
    records = [
        (
            "model.json",
            {
                "suite": "timm",
                "mode": "infer",
                "model": "resnet18",
                "fusible": {},
                "extern": [{"target": "aten.mm", "count": 0, "baseline_us": 1}],
            },
        )
    ]
    with pytest.raises(ValueError, match="invalid extern occurrence count"):
        build_artifact(records, hardware="NVIDIA B200", source="test")


def test_git_accounting_source_validates_manifest_digests(monkeypatch):
    sidecar = {
        "suite": "timm",
        "mode": "infer",
        "model": "resnet18",
        "fusible": {},
        "extern": [],
        "trace_errors": ["retry → failed"],
    }
    digest = hashlib.sha256(
        json.dumps(
            sidecar,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "hardware_key": "B200",
        "expected_sidecars": {"timm/infer/resnet18": "resnet18.json"},
        "sidecar_digests": {"timm/infer/resnet18": digest},
    }
    files = {
        "results/b200/occurrences/_metadata.json": json.dumps(manifest).encode(),
        "results/b200/occurrences/resnet18.json": json.dumps(
            sidecar, ensure_ascii=False
        ).encode(),
    }
    tree_paths = ["results/b200/occurrences/resnet18.json"]

    def check_output(command, **kwargs):
        if command[1] == "ls-tree":
            return "".join(f"{path}\n" for path in tree_paths)
        path = command[2].split(":", 1)[1]
        value = files[path]
        return value.decode() if kwargs.get("text") else value

    monkeypatch.setattr(
        "scripts.build_model_accounting.subprocess.check_output", check_output
    )

    validated, manifest_digest = validate_git_occurrence_manifest(
        "revision", "results/b200/occurrences"
    )
    assert validated["hardware_key"] == "B200"
    assert len(manifest_digest) == 64

    files["results/b200/occurrences/resnet18.json"] = b"{}"
    with pytest.raises(ValueError, match="does not match manifest digest"):
        validate_git_occurrence_manifest("revision", "results/b200/occurrences")

    files["results/b200/occurrences/resnet18.json"] = json.dumps(
        sidecar, ensure_ascii=False
    ).encode()
    nested = "results/b200/occurrences/nested/resnet18.json"
    files[nested] = files["results/b200/occurrences/resnet18.json"]
    tree_paths.append(nested)
    with pytest.raises(ValueError, match="nested occurrence sidecars"):
        validate_git_occurrence_manifest("revision", "results/b200/occurrences")


def test_dtype_metadata_preserves_integer_and_boolean_inputs():
    dtype, actual = _dtype({"inputs": [[[4], "i64"], [[4], "i32"], [[4], "b8"]]})

    assert dtype == "mixed"
    assert actual == "bool,int32,int64"
