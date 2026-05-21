#!/usr/bin/env python3
"""Report canonical/model repro corpus metadata.

This script is intentionally read-only. It counts current and legacy canonical
repros, model manifest references, and per-model full_graph availability, then
prints a safe recapture/deletion sequence for review.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


UNVERSIONED_REPRO_VERSION = 0
DEFAULT_CURRENT_REPRO_VERSION = 2
KNOWN_MODES = {"infer", "train"}
_REPRO_VERSION_RE = re.compile(r"^_repro_version\s*=\s*(\d+)\s*(?:#.*)?$")
_REPRO_VERSION_ASSIGN_RE = re.compile(r"^_repro_version\s*=.*$", re.MULTILINE)
_CURRENT_VERSION_RE = re.compile(
    r"^CURRENT_REPRO_VERSION\s*=\s*(\d+)\s*(?:#.*)?$",
    re.MULTILINE,
)
_HASH_SUFFIX_RE = re.compile(r"([0-9a-f]{12})$")


def parse_repro_version(source: str) -> tuple[int | None, str | None]:
    """Return (version, error). Unversioned legacy repros are version 0."""
    assignments = _REPRO_VERSION_ASSIGN_RE.findall(source)
    if not assignments:
        return UNVERSIONED_REPRO_VERSION, None
    if len(assignments) > 1:
        return None, "_repro_version must appear exactly once"

    match = _REPRO_VERSION_RE.match(assignments[0])
    if match is None:
        return None, "_repro_version must be a top-level integer assignment"
    return int(match.group(1)), None


def detect_current_repro_version(repo_root: Path) -> int:
    harness = repo_root / "repro_harness.py"
    if not harness.exists():
        return DEFAULT_CURRENT_REPRO_VERSION
    match = _CURRENT_VERSION_RE.search(harness.read_text())
    if match is None:
        return DEFAULT_CURRENT_REPRO_VERSION
    return int(match.group(1))


def has_top_level_assignment(source: str, name: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return True
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == name:
                return True
    return False


def _rel(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _limited(values: list[Any], max_examples: int) -> list[Any]:
    if max_examples < 0:
        return values
    return values[:max_examples]


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        return None, str(exc)
    if not isinstance(data, dict):
        return None, "top-level JSON value is not an object"
    return data, None


def _canonical_hash(repro_dir: Path, repo_root: Path) -> tuple[str, str]:
    meta_path = repro_dir / "meta.json"
    data, error = _read_json(meta_path) if meta_path.exists() else (None, None)
    if data is not None:
        pattern_hash = data.get("pattern_hash")
        if isinstance(pattern_hash, str) and pattern_hash:
            return pattern_hash, "meta.json"

    match = _HASH_SUFFIX_RE.search(repro_dir.name)
    if match is not None:
        return match.group(1), "directory_name"

    if error:
        return repro_dir.name, f"directory_name_meta_error:{error}"
    return repro_dir.name, "directory_name"


def scan_canonical_repros(
    canonical_dir: Path,
    repo_root: Path,
    current_version: int,
    max_examples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], set[str]]:
    repros: list[dict[str, Any]] = []
    version_counts: Counter[str] = Counter()
    hash_to_repros: dict[str, list[str]] = defaultdict(list)

    if canonical_dir.exists():
        dirs = sorted(d for d in canonical_dir.iterdir() if (d / "repro.py").exists())
    else:
        dirs = []

    for repro_dir in dirs:
        repro_py = repro_dir / "repro.py"
        source = repro_py.read_text()
        version, version_error = parse_repro_version(source)
        has_shapes_config = has_top_level_assignment(source, "_shapes_config")
        pattern_hash, hash_source = _canonical_hash(repro_dir, repo_root)
        rel_dir = _rel(repro_dir, repo_root)

        if version_error is not None:
            version_key = "invalid"
        else:
            version_key = str(version)
        version_counts[version_key] += 1
        hash_to_repros[pattern_hash].append(rel_dir)

        repros.append(
            {
                "path": rel_dir,
                "name": repro_dir.name,
                "pattern_hash": pattern_hash,
                "hash_source": hash_source,
                "version": version,
                "version_error": version_error,
                "has_shapes_config": has_shapes_config,
            }
        )

    current = [r["path"] for r in repros if r["version"] == current_version]
    unversioned = [
        r["path"]
        for r in repros
        if r["version"] == UNVERSIONED_REPRO_VERSION and r["version_error"] is None
    ]
    old_versioned = [
        r["path"]
        for r in repros
        if r["version"] is not None
        and UNVERSIONED_REPRO_VERSION < r["version"] < current_version
        and r["version_error"] is None
    ]
    newer_versioned = [
        r["path"]
        for r in repros
        if r["version"] is not None
        and r["version"] > current_version
        and r["version_error"] is None
    ]
    invalid = [
        {"path": r["path"], "error": r["version_error"]}
        for r in repros
        if r["version_error"] is not None
    ]
    missing_shapes = [r["path"] for r in repros if not r["has_shapes_config"]]
    collisions = {
        pattern_hash: paths
        for pattern_hash, paths in sorted(hash_to_repros.items())
        if len(paths) > 1
    }

    summary = {
        "directory": _rel(canonical_dir, repo_root),
        "total_repros": len(repros),
        "unique_pattern_hashes": len(hash_to_repros),
        "version_counts": dict(sorted(version_counts.items())),
        "current_version": {
            "version": current_version,
            "count": len(current),
            "examples": _limited(current, max_examples),
        },
        "unversioned": {
            "count": len(unversioned),
            "examples": _limited(unversioned, max_examples),
        },
        "old_versioned": {
            "count": len(old_versioned),
            "examples": _limited(old_versioned, max_examples),
        },
        "newer_versioned": {
            "count": len(newer_versioned),
            "examples": _limited(newer_versioned, max_examples),
        },
        "invalid_version_markers": {
            "count": len(invalid),
            "examples": _limited(invalid, max_examples),
        },
        "missing_shapes_config": {
            "count": len(missing_shapes),
            "examples": _limited(missing_shapes, max_examples),
        },
        "hash_collisions": {
            "count": len(collisions),
            "examples": dict(_limited(list(collisions.items()), max_examples)),
        },
    }
    return summary, repros, set(hash_to_repros)


def _manifest_location(
    manifest_path: Path,
    models_dir: Path,
) -> tuple[str, str, str, str, Path | None]:
    parts = manifest_path.relative_to(models_dir).parts
    suite = parts[0] if parts else "-"

    if manifest_path.name == "manifest.json":
        container = parts[:-1]
        if len(container) >= 3 and container[1] in KNOWN_MODES:
            mode = container[1]
            model = "/".join(container[2:])
        else:
            mode = "-"
            model = "/".join(container[1:]) if len(container) > 1 else suite
        return suite, mode, model, "directory", manifest_path.parent

    if len(parts) >= 3 and parts[1] in KNOWN_MODES:
        return suite, parts[1], manifest_path.stem, "flat", None
    model = manifest_path.stem
    return suite, "-", model, "flat", None


def _manifest_patterns(data: dict[str, Any]) -> tuple[list[str], dict[str, int]]:
    patterns_raw = data.get("patterns")
    shapes_raw = data.get("shapes")
    patterns = [p for p in patterns_raw if isinstance(p, str)] if isinstance(patterns_raw, list) else []
    shape_keys = list(shapes_raw) if isinstance(shapes_raw, dict) else []
    refs = patterns or [p for p in shape_keys if isinstance(p, str)]

    stats = {
        "non_string_patterns": (
            sum(1 for p in patterns_raw if not isinstance(p, str))
            if isinstance(patterns_raw, list)
            else 0
        ),
        "shape_keys_without_patterns": len(set(shape_keys) - set(patterns)),
        "patterns_without_shape_keys": len(set(patterns) - set(shape_keys)),
    }
    return refs, stats


def scan_model_manifests(
    models_dir: Path,
    repo_root: Path,
    canonical_hashes: set[str],
    max_examples: int,
) -> tuple[dict[str, Any], dict[str, Any], set[str]]:
    manifests: list[dict[str, Any]] = []
    by_suite_mode: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "manifest_files": 0,
            "directory_manifest_files": 0,
            "flat_manifest_files": 0,
            "total_pattern_references": 0,
            "unique_pattern_references": set(),
        }
    )
    invalid_json: list[dict[str, str]] = []
    all_refs: list[str] = []

    if models_dir.exists():
        manifest_paths = sorted(models_dir.rglob("*.json"))
    else:
        manifest_paths = []

    for manifest_path in manifest_paths:
        data, error = _read_json(manifest_path)
        suite, mode, model, kind, model_dir = _manifest_location(manifest_path, models_dir)
        rel_path = _rel(manifest_path, repo_root)
        if error is not None:
            invalid_json.append({"path": rel_path, "error": error})
            continue

        patterns, shape_stats = _manifest_patterns(data or {})
        if not patterns:
            continue

        key = (suite, mode)
        entry = by_suite_mode[key]
        entry["manifest_files"] += 1
        entry[f"{kind}_manifest_files"] += 1
        entry["total_pattern_references"] += len(patterns)
        entry["unique_pattern_references"].update(patterns)
        all_refs.extend(patterns)

        manifest = {
            "path": rel_path,
            "suite": suite,
            "mode": mode,
            "model": model,
            "kind": kind,
            "pattern_references": len(patterns),
            "unique_pattern_references": len(set(patterns)),
            "shape_key_stats": shape_stats,
        }
        manifests.append(manifest)

    unique_refs = set(all_refs)
    missing_canonical = sorted(unique_refs - canonical_hashes)
    unreferenced_canonical = sorted(canonical_hashes - unique_refs)

    by_suite_mode_rows = []
    for (suite, mode), values in sorted(by_suite_mode.items()):
        by_suite_mode_rows.append(
            {
                "suite": suite,
                "mode": mode,
                "manifest_files": values["manifest_files"],
                "directory_manifest_files": values["directory_manifest_files"],
                "flat_manifest_files": values["flat_manifest_files"],
                "total_pattern_references": values["total_pattern_references"],
                "unique_pattern_references": len(values["unique_pattern_references"]),
            }
        )

    manifest_summary = {
        "directory": _rel(models_dir, repo_root),
        "manifest_files": len(manifests),
        "directory_manifest_files": sum(1 for m in manifests if m["kind"] == "directory"),
        "flat_manifest_files": sum(1 for m in manifests if m["kind"] == "flat"),
        "total_pattern_references": len(all_refs),
        "unique_pattern_references": len(unique_refs),
        "references_missing_canonical": {
            "unique_count": len(missing_canonical),
            "examples": _limited(missing_canonical, max_examples),
        },
        "canonical_hashes_unreferenced_by_manifests": {
            "count": len(unreferenced_canonical),
            "examples": _limited(unreferenced_canonical, max_examples),
        },
        "invalid_json": {
            "count": len(invalid_json),
            "examples": _limited(invalid_json, max_examples),
        },
        "by_suite_mode": by_suite_mode_rows,
    }

    full_graph_summary = _summarize_full_graphs(models_dir, repo_root, max_examples)
    return manifest_summary, full_graph_summary, unique_refs


def _summarize_full_graphs(
    models_dir: Path,
    repo_root: Path,
    max_examples: int,
) -> dict[str, Any]:
    by_suite_mode: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {
            "model_dirs": 0,
            "with_full_graph": 0,
            "without_full_graph": 0,
            "full_graph_files": 0,
        }
    )
    source_dirs: dict[Path, list[Path]] = defaultdict(list)
    if models_dir.exists():
        for path in sorted(models_dir.rglob("full_graph_*.py")):
            source_dirs[path.parent].append(path)

    full_graph_files = sum(len(files) for files in source_dirs.values())

    for model_dir, files in sorted(source_dirs.items()):
        parts = model_dir.relative_to(models_dir).parts
        suite = parts[0] if parts else "-"
        if len(parts) >= 3 and parts[1] in KNOWN_MODES:
            mode = parts[1]
        else:
            mode = "-"
        key = (suite, mode)
        row = by_suite_mode[key]
        row["model_dirs"] += 1
        row["full_graph_files"] += len(files)
        row["with_full_graph"] += 1

    rows = []
    for (suite, mode), values in sorted(by_suite_mode.items()):
        rows.append({"suite": suite, "mode": mode, **values})

    return {
        "model_dirs": len(source_dirs),
        "with_full_graph": len(source_dirs),
        "without_full_graph": 0,
        "full_graph_files": full_graph_files,
        "missing_examples": [],
        "by_suite_mode": rows,
    }


def _is_old_or_incomplete(repro: dict[str, Any], current_version: int) -> bool:
    return (
        repro["version_error"] is not None
        or repro["version"] != current_version
        or not repro["has_shapes_config"]
    )


def _summarize_followup(
    repros: list[dict[str, Any]],
    manifest_refs: set[str],
    current_version: int,
    max_examples: int,
) -> dict[str, Any]:
    old_or_incomplete = [r for r in repros if _is_old_or_incomplete(r, current_version)]
    referenced = [r for r in old_or_incomplete if r["pattern_hash"] in manifest_refs]
    unreferenced = [r for r in old_or_incomplete if r["pattern_hash"] not in manifest_refs]
    return {
        "old_or_incomplete_canonical_repros": {
            "count": len(old_or_incomplete),
            "examples": _limited([r["path"] for r in old_or_incomplete], max_examples),
        },
        "referenced_old_or_incomplete_canonical_repros": {
            "count": len(referenced),
            "examples": _limited([r["path"] for r in referenced], max_examples),
        },
        "unreferenced_old_or_incomplete_canonical_repros": {
            "count": len(unreferenced),
            "examples": _limited([r["path"] for r in unreferenced], max_examples),
        },
        "safe_sequence": [
            "Use saved full_graph source directories for future recapture; generated per-graph manifest dirs are not expected to contain full_graph_*.py.",
            "Re-run this report and require zero manifest references missing canonical hashes before deleting any old canonical repros.",
            "Require zero manifest references missing canonical hashes before and after each reviewed change.",
            "Recapture or upgrade referenced old/incomplete canonical repros one at a time from the full graph source, then rerun focused tests.",
            "Delete only explicit, reviewed unreferenced canonical repro paths after recapture coverage is confirmed; do not use wildcards or bulk deletes.",
        ],
    }


def build_report(
    repo_root: Path,
    canonical_dir: Path | None = None,
    models_dir: Path | None = None,
    current_version: int | None = None,
    max_examples: int = 10,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    canonical_dir = (canonical_dir or repo_root / "repros" / "canonical").resolve()
    models_dir = (models_dir or repo_root / "repros" / "models").resolve()
    current_version = current_version or detect_current_repro_version(repo_root)

    canonical, repros, canonical_hashes = scan_canonical_repros(
        canonical_dir,
        repo_root,
        current_version,
        max_examples,
    )
    manifests, full_graph, manifest_refs = scan_model_manifests(
        models_dir,
        repo_root,
        canonical_hashes,
        max_examples,
    )
    followup = _summarize_followup(
        repros,
        manifest_refs,
        current_version,
        max_examples,
    )

    return {
        "repo_root": repo_root.as_posix(),
        "current_repro_version": current_version,
        "canonical": canonical,
        "model_manifest_references": manifests,
        "full_graph_availability": full_graph,
        "followup": followup,
    }


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    rendered = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rendered))
        for idx in range(len(headers))
    ]
    lines = ["  " + "  ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers)))]
    lines.append("  " + "  ".join("-" * width for width in widths))
    for row in rendered:
        lines.append("  " + "  ".join(row[idx].rjust(widths[idx]) for idx in range(len(headers))))
    return lines


def _example_lines(label: str, examples: list[Any]) -> list[str]:
    if not examples:
        return []
    lines = [f"  {label} examples:"]
    for example in examples:
        if isinstance(example, dict):
            path = example.get("path", example)
            error = example.get("error")
            lines.append(f"    {path}: {error}" if error else f"    {path}")
        else:
            lines.append(f"    {example}")
    return lines


def render_report(report: dict[str, Any]) -> str:
    canonical = report["canonical"]
    manifests = report["model_manifest_references"]
    full_graph = report["full_graph_availability"]
    followup = report["followup"]
    current_version = report["current_repro_version"]

    lines = [
        "Repro Corpus Report",
        f"Repo root: {report['repo_root']}",
        f"Current repro version: v{current_version}",
        "",
        "Canonical repros:",
        f"  directory: {canonical['directory']}",
        f"  total repros: {canonical['total_repros']}",
        f"  unique pattern hashes: {canonical['unique_pattern_hashes']}",
        f"  current-version repros (v{current_version}): {canonical['current_version']['count']}",
        f"  unversioned repros: {canonical['unversioned']['count']}",
        f"  old versioned repros: {canonical['old_versioned']['count']}",
        f"  newer versioned repros: {canonical['newer_versioned']['count']}",
        f"  invalid version markers: {canonical['invalid_version_markers']['count']}",
        f"  missing _shapes_config: {canonical['missing_shapes_config']['count']}",
        f"  hash collisions: {canonical['hash_collisions']['count']}",
    ]
    lines.extend(_example_lines("unversioned", canonical["unversioned"]["examples"]))
    lines.extend(
        _example_lines(
            "missing _shapes_config",
            canonical["missing_shapes_config"]["examples"],
        )
    )
    lines.extend(
        _example_lines(
            "invalid version marker",
            canonical["invalid_version_markers"]["examples"],
        )
    )

    lines.extend(
        [
            "",
            "Model manifest references:",
            f"  directory: {manifests['directory']}",
            f"  manifest files: {manifests['manifest_files']}",
            f"  directory manifest files: {manifests['directory_manifest_files']}",
            f"  flat manifest files: {manifests['flat_manifest_files']}",
            f"  total pattern references: {manifests['total_pattern_references']}",
            f"  unique pattern references: {manifests['unique_pattern_references']}",
            "  refs missing canonical: "
            f"{manifests['references_missing_canonical']['unique_count']}",
            "  canonical hashes unreferenced by manifests: "
            f"{manifests['canonical_hashes_unreferenced_by_manifests']['count']}",
            f"  invalid json files: {manifests['invalid_json']['count']}",
        ]
    )
    lines.extend(
        _example_lines(
            "refs missing canonical",
            manifests["references_missing_canonical"]["examples"],
        )
    )

    if manifests["by_suite_mode"]:
        lines.append("  by suite/mode:")
        lines.extend(
            _table(
                ["suite/mode", "files", "dir", "flat", "refs", "unique"],
                [
                    [
                        f"{row['suite']}/{row['mode']}",
                        row["manifest_files"],
                        row["directory_manifest_files"],
                        row["flat_manifest_files"],
                        row["total_pattern_references"],
                        row["unique_pattern_references"],
                    ]
                    for row in manifests["by_suite_mode"]
                ],
            )
        )

    lines.extend(
        [
            "",
            "Saved full graph sources:",
            f"  source dirs: {full_graph['model_dirs']}",
            f"  full_graph files: {full_graph['full_graph_files']}",
        ]
    )
    lines.extend(
        _example_lines(
            "missing full_graph",
            full_graph["missing_examples"],
        )
    )
    if full_graph["by_suite_mode"]:
        lines.append("  by suite/mode:")
        lines.extend(
            _table(
                ["suite/mode", "source dirs", "files"],
                [
                    [
                        f"{row['suite']}/{row['mode']}",
                        row["model_dirs"],
                        row["full_graph_files"],
                    ]
                    for row in full_graph["by_suite_mode"]
                ],
            )
        )

    lines.extend(
        [
            "",
            "Old/incomplete canonical repros:",
            "  old or incomplete: "
            f"{followup['old_or_incomplete_canonical_repros']['count']}",
            "  still referenced by manifests: "
            f"{followup['referenced_old_or_incomplete_canonical_repros']['count']}",
            "  unreferenced by manifests: "
            f"{followup['unreferenced_old_or_incomplete_canonical_repros']['count']}",
        ]
    )
    lines.extend(
        _example_lines(
            "unreferenced old/incomplete",
            followup["unreferenced_old_or_incomplete_canonical_repros"]["examples"],
        )
    )

    lines.extend(["", "Safe deletion/recapture sequence:"])
    for idx, step in enumerate(followup["safe_sequence"], start=1):
        lines.append(f"  {idx}. {step}")
    lines.append("  No deletion or recapture is performed by this report.")
    return "\n".join(lines)


def _resolve_arg_path(repo_root: Path, value: Path | None, default: Path) -> Path:
    if value is None:
        return default
    if value.is_absolute():
        return value
    return repo_root / value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--canonical-dir", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--current-version", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    canonical_dir = _resolve_arg_path(
        repo_root,
        args.canonical_dir,
        repo_root / "repros" / "canonical",
    )
    models_dir = _resolve_arg_path(
        repo_root,
        args.models_dir,
        repo_root / "repros" / "models",
    )
    report = build_report(
        repo_root=repo_root,
        canonical_dir=canonical_dir,
        models_dir=models_dir,
        current_version=args.current_version,
        max_examples=args.max_examples,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_report(report))


if __name__ == "__main__":
    main()
