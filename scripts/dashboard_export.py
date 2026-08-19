#!/usr/bin/env python3
"""Export one Better Benchmark sweep as PyTorch v3 dashboard records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BENCHMARK_NAME = "inductor-kernel-benchmark"
RESERVED_KEYS = {"_metadata", "__failures__", "__summary__"}

DTYPE_NAMES = {
    "b8": "bool",
    "bf16": "bfloat16",
    "f16": "float16",
    "f32": "float32",
    "f64": "float64",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "u8": "uint8",
}


def _hardware_key(name: str) -> str:
    return " ".join(name.casefold().split()).removeprefix("nvidia ")
TIMING_FIELDS = ("compiled_us", "coord_descent_us")


def _pattern_hash(repro_path: str) -> str:
    return Path(repro_path).parent.name.rsplit("_", 1)[-1]


def _finite_positive(value, context: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{context}: expected a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{context}: expected a positive finite number")
    return number


def _finite_nonnegative(value, context: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{context}: expected a non-negative finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{context}: expected a non-negative finite number")
    return number


def _selected_timing(measurement: dict, timing: str) -> tuple[object, str]:
    if timing == "auto":
        for field in ("coord_descent_us", "compiled_us"):
            value = measurement.get(field)
            if value is None:
                continue
            return value, field
        return None, ""
    return measurement.get(timing), timing


def _shape_file(repro_path: str) -> Path:
    path = Path(repro_path)
    if path.is_absolute():
        return path.parent / "shapes.json"
    return ROOT / path.parent / "shapes.json"


def _shape_metadata(repro_path: str) -> dict[str, dict]:
    path = _shape_file(repro_path)
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    return {
        point["shape_hash"]: point
        for point in payload.get("points", [])
        if isinstance(point, dict) and isinstance(point.get("shape_hash"), str)
    }


def _collect_dtypes(value, output: set[str]) -> None:
    if isinstance(value, list):
        if len(value) >= 2 and isinstance(value[1], str) and value[1] in DTYPE_NAMES:
            output.add(DTYPE_NAMES[value[1]])
            return
        for item in value:
            _collect_dtypes(item, output)


def _dtype(point: dict) -> tuple[str, str]:
    dtypes: set[str] = set()
    _collect_dtypes(point.get("inputs", []), dtypes)
    ordered = sorted(dtypes)
    if not ordered:
        return "unknown", ""
    # The corpus is mixed-precision: f32 parameters commonly accompany bf16
    # activations. Report the workload precision as the dashboard dtype while
    # preserving the exact set in actual_dtypes.
    for workload_dtype in ("bfloat16", "float16"):
        if workload_dtype in dtypes:
            return workload_dtype, ",".join(ordered)
    return (ordered[0] if len(ordered) == 1 else "mixed"), ",".join(ordered)


def _sources(point: dict) -> tuple[str, str, list[str]]:
    models = sorted((point.get("models") or {}).keys())
    suites = set()
    modes = set()
    for model in models:
        parts = model.split("/", 2)
        if len(parts) == 3:
            suites.add(parts[0])
            modes.add(parts[1])
    suite = next(iter(suites)) if len(suites) == 1 else "mixed"
    mode = next(iter(modes)) if len(modes) == 1 else "mixed"
    return suite, mode, models


def load_kernel_points(path: Path, timing: str = "auto") -> dict[tuple[str, str], dict]:
    if timing not in (*TIMING_FIELDS, "auto"):
        raise ValueError(f"unknown timing axis: {timing}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: sweep root must be an object")
    metadata = payload.get("_metadata")
    workload_kind = metadata.get("workload_kind") if isinstance(metadata, dict) else None
    if workload_kind not in (None, "repro"):
        raise ValueError(
            f"{path}: dashboard export requires a canonical kernel sweep, "
            f"not workload_kind={workload_kind!r}"
        )

    points = {}
    skipped_measurements = 0
    invalid_gaps = 0
    missing_shape_files = set()
    unresolved_shape_metadata = set()
    for repro_path, measurements in payload.items():
        if (
            repro_path in RESERVED_KEYS
            or not isinstance(repro_path, str)
            or not isinstance(measurements, dict)
        ):
            continue
        if Path(repro_path).name.startswith("full_graph_"):
            raise ValueError(
                f"{path}: dashboard export does not support full-graph sweeps"
            )
        pattern_hash = _pattern_hash(repro_path)
        repro_dir = Path(repro_path).parent.name
        metadata = _shape_metadata(repro_path)
        if not _shape_file(repro_path).is_file():
            missing_shape_files.add(str(_shape_file(repro_path)))
        for label, measurement in measurements.items():
            if label == "__graph__" or not isinstance(measurement, dict):
                continue
            raw_us, selected_field = _selected_timing(measurement, timing)
            if raw_us is None:
                continue
            try:
                us = _finite_positive(raw_us, f"{repro_path}/{label}/{selected_field}")
            except (TypeError, ValueError):
                skipped_measurements += 1
                continue
            shape_hash = label.rsplit("_", 1)[-1]
            key = (pattern_hash, shape_hash)
            if key in points:
                raise ValueError(f"duplicate kernel point {pattern_hash}/{shape_hash}")
            point = metadata.get(shape_hash, {})
            if shape_hash not in metadata:
                unresolved_shape_metadata.add((repro_path, shape_hash))
            dtype, actual_dtypes = _dtype(point)
            suite, mode, source_models = _sources(point)
            gap_field = (
                "gap_cd" if selected_field == "coord_descent_us" else "gap_default"
            )
            gap = measurement.get(gap_field)
            gap_vs_sol = None
            if gap is not None:
                try:
                    gap_vs_sol = _finite_positive(
                        gap, f"{repro_path}/{label}/{gap_field}"
                    )
                except (TypeError, ValueError):
                    invalid_gaps += 1
            points[key] = {
                "pattern_hash": pattern_hash,
                "shape_hash": shape_hash,
                "repro_dir": repro_dir,
                "display_name": f"{repro_dir}[{shape_hash}]",
                "timing": selected_field,
                "us": us,
                "gap_vs_sol": gap_vs_sol,
                "suite": suite,
                "mode": mode,
                "dtype": dtype,
                "actual_dtypes": actual_dtypes,
                "source_models": source_models,
            }
    if not points:
        raise ValueError(f"{path}: no valid kernel points found")
    if skipped_measurements:
        print(
            f"Skipped {skipped_measurements} invalid timing measurements",
            file=sys.stderr,
        )
    if invalid_gaps:
        print(f"Skipped {invalid_gaps} invalid gap measurements", file=sys.stderr)
    if missing_shape_files:
        print(
            f"Missing shapes.json metadata for {len(missing_shape_files)} repros",
            file=sys.stderr,
        )
    if unresolved_shape_metadata:
        print(
            f"Missing shape metadata for {len(unresolved_shape_metadata)} points",
            file=sys.stderr,
        )
    failures = payload.get("__failures__", {})
    successful_repros = {
        key
        for key, value in payload.items()
        if key not in RESERVED_KEYS and isinstance(value, dict)
    }
    failed_repro_paths = (
        {key.split("::SHAPE::", 1)[0] for key in failures}
        if isinstance(failures, dict)
        else set()
    )
    failed_repro_paths.difference_update(successful_repros)
    run_info = {
        "sweep_total_repros": str(len(successful_repros | failed_repro_paths)),
        "sweep_failed_repros": str(len(failed_repro_paths)),
        "sweep_invalid_measurements": str(skipped_measurements),
        "sweep_missing_shape_files": str(len(missing_shape_files)),
        "sweep_unresolved_shape_metadata": str(len(unresolved_shape_metadata)),
    }
    for point in points.values():
        point["run_info"] = run_info
    return points


def _record(
    *,
    model_name: str,
    model_type: str,
    metric: str,
    value: float,
    mode: str,
    dtype: str,
    device: str,
    arch: str,
    extra_info: dict[str, str],
) -> dict:
    return {
        "benchmark": {
            "name": BENCHMARK_NAME,
            "mode": mode,
            "dtype": dtype,
            "extra_info": {
                "device": device,
                "arch": arch,
                **extra_info,
            },
        },
        "model": {
            "name": model_name,
            "type": model_type,
            "origins": ["pytorch"],
        },
        "metric": {
            "name": metric,
            "benchmark_values": [value],
        },
    }


def kernel_records(
    points: dict,
    *,
    device: str,
    arch: str,
    timing_policy: str,
) -> list[dict]:
    records = []
    for point in points.values():
        info = {
            "record_type": "kernel",
            "pattern_hash": point["pattern_hash"],
            "shape_hash": point["shape_hash"],
            "kernel_name": point["repro_dir"],
            "suite": point["suite"],
            "source_mode": point["mode"],
            "actual_dtypes": point["actual_dtypes"],
            "example_model": (
                point["source_models"][0] if point["source_models"] else ""
            ),
            "source_model_count": str(len(point["source_models"])),
            "timing": point["timing"],
            "timing_policy": timing_policy,
            **point["run_info"],
        }
        records.append(
            _record(
                model_name=point["display_name"],
                model_type="kernel",
                metric="latency_us",
                value=point["us"],
                mode=point["mode"],
                dtype=point["dtype"],
                device=device,
                arch=arch,
                extra_info=info,
            )
        )
        if point["gap_vs_sol"] is not None:
            records.append(
                _record(
                    model_name=point["display_name"],
                    model_type="kernel",
                    metric="gap_vs_sol",
                    value=point["gap_vs_sol"],
                    mode=point["mode"],
                    dtype=point["dtype"],
                    device=device,
                    arch=arch,
                    extra_info=info,
                )
            )
    return records


def model_records(
    points: dict,
    accounting_path: Path,
    *,
    device: str,
    arch: str,
    timing_policy: str,
) -> list[dict]:
    accounting_bytes = accounting_path.read_bytes()
    accounting = json.loads(accounting_bytes)
    schema_version = accounting.get("schema_version")
    models = accounting.get("models")
    if schema_version != 1 or not isinstance(models, dict):
        raise ValueError(f"{accounting_path}: unsupported accounting schema")
    accounting_hardware = accounting.get("hardware")
    if not isinstance(accounting_hardware, str) or not accounting_hardware:
        raise ValueError(f"{accounting_path}: missing accounting hardware")
    if not isinstance(arch, str) or not arch.strip():
        raise ValueError("run architecture is required for projected model records")
    if _hardware_key(accounting_hardware) != _hardware_key(arch):
        raise ValueError(
            f"accounting hardware {accounting_hardware!r} does not match "
            f"run architecture {arch!r}"
        )
    accounting_digest = hashlib.sha256(
        json.dumps(
            accounting, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()

    records = []
    for name, model in sorted(models.items()):
        model_accounting_digest = hashlib.sha256(
            json.dumps(
                model, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        ).hexdigest()
        required_fields = (
            "suite",
            "mode",
            "model",
            "fusible",
            "extern_total_us",
            "unpriced_extern_occurrences",
            "trace_errors",
        )
        missing_fields = [field for field in required_fields if field not in model]
        if missing_fields:
            raise ValueError(f"{name}: missing accounting fields {missing_fields}")
        if not isinstance(model["fusible"], dict):
            raise TypeError(f"{name}: fusible must be an object")
        if not isinstance(model["trace_errors"], list):
            raise TypeError(f"{name}: trace_errors must be a list")
        unpriced_extern = model["unpriced_extern_occurrences"]
        if (
            isinstance(unpriced_extern, bool)
            or not isinstance(unpriced_extern, int)
            or unpriced_extern < 0
        ):
            raise ValueError(f"{name}: invalid unpriced extern occurrence count")

        total_occurrences = 0
        matched_occurrences = 0
        fusible_total_us = 0.0
        for pattern_hash, shapes in model["fusible"].items():
            if not isinstance(pattern_hash, str) or not isinstance(shapes, dict):
                raise TypeError(f"{name}: invalid fusible occurrence map")
            for shape_hash, count in shapes.items():
                if (
                    not isinstance(shape_hash, str)
                    or isinstance(count, bool)
                    or not isinstance(count, int)
                    or count <= 0
                ):
                    raise ValueError(f"{name}: invalid occurrence count")
                total_occurrences += count
                point = points.get((pattern_hash, shape_hash))
                if point is None:
                    continue
                matched_occurrences += count
                fusible_total_us += point["us"] * count

        coverage = matched_occurrences / total_occurrences if total_occurrences else 1.0
        extern_total_us = _finite_nonnegative(
            model["extern_total_us"], f"{name}/extern_total_us"
        )
        exclusion_reasons = []
        if matched_occurrences != total_occurrences:
            exclusion_reasons.append("unmatched_kernel")
        if unpriced_extern:
            exclusion_reasons.append("unpriced_extern")
        if model["trace_errors"]:
            exclusion_reasons.append("trace_errors")
        if fusible_total_us + extern_total_us <= 0:
            exclusion_reasons.append("no_priced_baseline")
        complete = not exclusion_reasons
        model_dtype = "mixed"
        common_info = {
            "record_type": "model",
            "suite": model["suite"],
            "source_mode": model["mode"],
            "accounting_schema": str(schema_version),
            "accounting_source": str(accounting.get("source", "")),
            "accounting_source_manifest_digest": str(
                accounting.get("source_manifest_digest", "")
            ),
            "accounting_digest": accounting_digest,
            "model_accounting_digest": model_accounting_digest,
            "accounting_hardware": accounting_hardware,
            "timing_policy": timing_policy,
            "matched_occurrences": str(matched_occurrences),
            "total_occurrences": str(total_occurrences),
            "included": str(complete).lower(),
            "exclusion_reasons": ",".join(exclusion_reasons),
        }
        records.append(
            _record(
                model_name=name,
                model_type="projected-model",
                metric="model_coverage_ratio",
                value=coverage,
                mode=model["mode"],
                dtype=model_dtype,
                device=device,
                arch=arch,
                extra_info=common_info,
            )
        )
        if complete:
            projected_total_us = fusible_total_us + extern_total_us
            records.append(
                _record(
                    model_name=name,
                    model_type="projected-model",
                    metric="projected_model_latency_us",
                    value=_finite_positive(
                        projected_total_us, f"{name}/projected_total_us"
                    ),
                    mode=model["mode"],
                    dtype=model_dtype,
                    device=device,
                    arch=arch,
                    extra_info=common_info,
                )
            )
    return records


def detect_arch() -> str:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("could not detect GPU architecture with nvidia-smi") from error
    lines = output.strip().splitlines()
    if not lines:
        raise RuntimeError("nvidia-smi returned no GPU architecture")
    return lines[0]


def export_records(
    input_path: Path,
    *,
    model_accounting: Path | None,
    device: str,
    arch: str,
    timing: str = "auto",
) -> list[dict]:
    points = load_kernel_points(input_path, timing)
    records = kernel_records(
        points,
        device=device,
        arch=arch,
        timing_policy=timing,
    )
    if model_accounting is not None:
        records.extend(
            model_records(
                points,
                model_accounting,
                device=device,
                arch=arch,
                timing_policy=timing,
            )
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export one sweep as PyTorch v3 dashboard records"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--ci-json", required=True, type=Path)
    parser.add_argument("--model-accounting", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arch", default="")
    parser.add_argument(
        "--timing",
        choices=("auto", *TIMING_FIELDS),
        default="auto",
        help=(
            "Timing axis. auto uses coord_descent_us when present and falls back "
            "to compiled_us only when coordinate-descent timing is absent."
        ),
    )
    args = parser.parse_args()

    for label, path in (
        ("input", args.input),
        ("model accounting", args.model_accounting),
    ):
        if path is not None and not path.is_file():
            parser.error(f"{label} file not found: {path}")

    records = export_records(
        args.input,
        model_accounting=args.model_accounting,
        device=args.device,
        arch=args.arch or detect_arch(),
        timing=args.timing,
    )
    args.ci_json.parent.mkdir(parents=True, exist_ok=True)
    args.ci_json.write_text(json.dumps(records, indent=2))
    print(f"Wrote {len(records)} records to {args.ci_json}")


if __name__ == "__main__":
    main()
