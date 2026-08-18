#!/usr/bin/env python3
"""Build the compact model-accounting input used by dashboard exports.

The source directory is the occurrence sidecar directory consumed by
``perf_ab_rollup.py``.  The compact artifact preserves exact kernel-shape
occurrence counts and the total unchanged extern latency per model, which is
all a single-run exporter needs to reconstruct projected model latency.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

SCHEMA_VERSION = 1


def _compact_record(record: dict, source: str) -> tuple[str, dict]:
    for field in ("suite", "mode", "model"):
        if not isinstance(record.get(field), str) or not record[field]:
            raise ValueError(f"{source}: missing non-empty {field}")
    if not isinstance(record.get("fusible"), dict):
        raise TypeError(f"{source}: fusible must be an object")
    if not isinstance(record.get("extern"), list):
        raise TypeError(f"{source}: extern must be a list")
    for pattern_hash, shapes in record["fusible"].items():
        if not isinstance(pattern_hash, str) or not isinstance(shapes, dict):
            raise TypeError(f"{source}: invalid fusible occurrence map")
        for shape_hash, count in shapes.items():
            if (
                not isinstance(shape_hash, str)
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
            ):
                raise ValueError(
                    f"{source}: invalid fusible occurrence "
                    f"{pattern_hash}/{shape_hash}={count!r}"
                )

    extern_total_us = 0.0
    extern_occurrences = 0
    unpriced_extern_occurrences = 0
    for extern in record["extern"]:
        if not isinstance(extern, dict):
            raise TypeError(f"{source}: every extern entry must be an object")
        count = extern.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"{source}: invalid extern occurrence count")
        extern_occurrences += count
        baseline_us = extern.get("baseline_us")
        if baseline_us is None:
            unpriced_extern_occurrences += count
            continue
        value = float(baseline_us)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{source}: invalid extern baseline_us={value}")
        extern_total_us += value * count

    trace_errors = record.get("trace_errors", [])
    if not isinstance(trace_errors, list):
        raise TypeError(f"{source}: trace_errors must be a list")

    name = f"{record['suite']}/{record['mode']}/{record['model']}"
    return name, {
        "suite": record["suite"],
        "mode": record["mode"],
        "model": record["model"],
        "fusible": record["fusible"],
        "extern_total_us": extern_total_us,
        "extern_occurrences": extern_occurrences,
        "unpriced_extern_occurrences": unpriced_extern_occurrences,
        "trace_errors": trace_errors,
    }


def _records_from_directory(path: Path):
    for sidecar in sorted(path.glob("*.json")):
        if sidecar.name.startswith("_"):
            continue
        yield sidecar.name, json.loads(sidecar.read_text())


def _records_from_git(revision: str, git_path: str):
    try:
        output = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", revision, git_path],
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            f"could not read {git_path} from git revision {revision}"
        ) from exc
    for path in output.splitlines():
        if not path.endswith(".json") or Path(path).name.startswith("_"):
            continue
        content = subprocess.check_output(
            ["git", "show", f"{revision}:{path}"],
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        yield path, json.loads(content)


def build_artifact(
    records,
    *,
    hardware: str,
    source: str,
) -> dict:
    models = {}
    for name, raw in records:
        model_key, compact = _compact_record(raw, name)
        if model_key in models:
            raise ValueError(f"duplicate model accounting record: {model_key}")
        models[model_key] = compact
    if not models:
        raise ValueError("no model accounting records found")
    return {
        "schema_version": SCHEMA_VERSION,
        "hardware": hardware,
        "source": source,
        "models": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compact perf_ab_rollup occurrence sidecars"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--occdir", type=Path)
    source.add_argument("--git-revision")
    parser.add_argument(
        "--git-path",
        default="results/b200/occurrences",
        help="Occurrence directory inside --git-revision",
    )
    parser.add_argument("--hardware", default="NVIDIA B200")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.occdir is not None:
        if not args.occdir.is_dir():
            parser.error(f"occurrence directory not found: {args.occdir}")
        records = _records_from_directory(args.occdir)
        source_name = str(args.occdir)
    else:
        records = _records_from_git(args.git_revision, args.git_path)
        source_name = f"{args.git_revision}:{args.git_path}"

    artifact = build_artifact(
        records,
        hardware=args.hardware,
        source=source_name,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, sort_keys=True, separators=(",", ":")))
    print(f"Wrote {len(artifact['models'])} models to {args.output}")


if __name__ == "__main__":
    main()
