#!/usr/bin/env python3
"""Build the compact model-accounting input used by dashboard exports.

The source directory is the occurrence sidecar directory consumed by
``perf_ab_rollup.py``.  The compact artifact preserves exact kernel-shape
occurrence counts and the total unchanged extern latency per model, which is
all a single-run exporter needs to reconstruct projected model latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parent.parent


def _logical_source(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return path.name


def _manifest_content_digest(manifest: dict) -> str:
    content = {
        key: manifest.get(key)
        for key in (
            "schema_version",
            "corpus_digest",
            "hardware",
            "hardware_key",
            "extern_cache_fingerprint",
            "expected_sidecars",
            "sidecar_digests",
        )
    }
    return hashlib.sha256(
        json.dumps(
            content, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


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
        yield sidecar.name, json.loads(sidecar.read_text(encoding="utf-8"))


def _records_from_git(revision: str, git_path: str):
    try:
        output = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", revision, git_path],
            text=True,
            cwd=ROOT,
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
            cwd=ROOT,
        )
        yield path, json.loads(content)


def _git_file(revision: str, path: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"{revision}:{path}"],
            cwd=ROOT,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"could not read {path} from git revision {revision}") from exc


def validate_git_occurrence_manifest(
    revision: str, git_path: str
) -> tuple[dict, str]:
    manifest_path = f"{git_path.rstrip('/')}/_metadata.json"
    manifest_bytes = _git_file(revision, manifest_path)
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{revision}:{manifest_path}: unsupported schema version")
    if manifest.get("status") != "complete":
        raise ValueError(f"{revision}:{manifest_path}: generation is not complete")
    expected = manifest.get("expected_sidecars")
    digests = manifest.get("sidecar_digests")
    if not isinstance(expected, dict) or not isinstance(digests, dict):
        raise TypeError(f"{revision}:{manifest_path}: missing sidecar inventory")
    if set(expected) != set(digests):
        raise ValueError(f"{revision}:{manifest_path}: missing sidecar digests")

    prefix = f"{git_path.rstrip('/')}/"
    actual_files = set()
    for path, _record in _records_from_git(revision, git_path):
        if not path.startswith(prefix):
            raise ValueError(f"{revision}:{path}: sidecar is outside {git_path}")
        relative = path[len(prefix) :]
        if "/" in relative:
            raise ValueError(
                f"{revision}:{path}: nested occurrence sidecars are not supported"
            )
        actual_files.add(relative)
    if set(expected.values()) != actual_files:
        raise ValueError(
            f"{revision}:{manifest_path}: sidecar inventory does not match git tree"
        )
    for identity, filename in expected.items():
        raw = json.loads(_git_file(revision, f"{git_path.rstrip('/')}/{filename}"))
        digest = hashlib.sha256(
            json.dumps(
                raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        ).hexdigest()
        if digest != digests[identity]:
            raise ValueError(
                f"{revision}:{git_path}/{filename}: "
                "content does not match manifest digest"
            )
    return manifest, _manifest_content_digest(manifest)


def _hardware_key(value: str) -> str:
    return " ".join(value.upper().split()).removeprefix("NVIDIA ")


def _manifest_hardware(manifest: dict) -> str:
    hardware = manifest.get("hardware")
    if isinstance(hardware, dict):
        device_name = hardware.get("device_name")
        if isinstance(device_name, str) and device_name.strip():
            return device_name
    return str(manifest.get("hardware_key"))


def build_artifact(
    records,
    *,
    hardware: str,
    source: str,
    source_manifest_digest: str = "",
    source_provenance: dict | None = None,
) -> dict:
    models = {}
    for name, raw in records:
        model_key, compact = _compact_record(raw, name)
        if model_key in models:
            raise ValueError(f"duplicate model accounting record: {model_key}")
        models[model_key] = compact
    if not models:
        raise ValueError("no model accounting records found")
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "hardware": hardware,
        "source": source,
        "models": models,
    }
    if source_manifest_digest:
        artifact["source_manifest_digest"] = source_manifest_digest
    if source_provenance:
        artifact["source_provenance"] = source_provenance
    return artifact


def validate_occurrence_manifest(path: Path) -> tuple[dict | None, str]:
    manifest_path = path / "_metadata.json"
    if not manifest_path.is_file():
        return None, ""
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{manifest_path}: unsupported schema version")
    if manifest.get("status") != "complete":
        raise ValueError(f"{manifest_path}: generation is not complete")
    expected = manifest.get("expected_sidecars")
    digests = manifest.get("sidecar_digests")
    if not isinstance(expected, dict) or not isinstance(digests, dict):
        raise TypeError(f"{manifest_path}: missing sidecar inventory")
    expected_files = set(expected.values())
    actual_files = {
        item.name
        for item in path.glob("*.json")
        if not item.name.startswith("_")
    }
    if expected_files != actual_files:
        raise ValueError(
            f"{manifest_path}: sidecar inventory does not match output directory"
        )
    if set(expected) != set(digests):
        raise ValueError(f"{manifest_path}: missing sidecar digests")
    for identity, filename in expected.items():
        sidecar = path / filename
        digest = hashlib.sha256(
            json.dumps(
                json.loads(sidecar.read_text(encoding="utf-8")),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        if digest != digests[identity]:
            raise ValueError(f"{sidecar}: content does not match manifest digest")
    return manifest, _manifest_content_digest(manifest)


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
    parser.add_argument("--hardware")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.occdir is not None:
        if not args.occdir.is_dir():
            parser.error(f"occurrence directory not found: {args.occdir}")
        manifest, manifest_digest = validate_occurrence_manifest(args.occdir)
        records = _records_from_directory(args.occdir)
        source_name = _logical_source(args.occdir)
        hardware = (
            _manifest_hardware(manifest)
            if manifest is not None
            else args.hardware or "NVIDIA B200"
        )
    else:
        manifest, manifest_digest = validate_git_occurrence_manifest(
            args.git_revision, args.git_path
        )
        records = _records_from_git(args.git_revision, args.git_path)
        source_name = f"{args.git_revision}:{args.git_path}"
        hardware = _manifest_hardware(manifest)
        if args.hardware and _hardware_key(args.hardware) != _hardware_key(hardware):
            parser.error(
                f"--hardware {args.hardware!r} disagrees with manifest "
                f"hardware {hardware!r}"
            )
    if not hardware or hardware == "None":
        parser.error("accounting hardware is missing")

    artifact = build_artifact(
        records,
        hardware=hardware,
        source=source_name,
        source_manifest_digest=manifest_digest,
        source_provenance=(
            {
                key: manifest.get(key)
                for key in (
                    "better_benchmark_commit",
                    "generated_at",
                    "completed_at",
                    "corpus_digest",
                    "hardware",
                    "priced_extern_points",
                    "failed_extern_points",
                )
            }
            if manifest is not None
            else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(artifact['models'])} models to {args.output}")


if __name__ == "__main__":
    main()
