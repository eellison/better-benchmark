#!/usr/bin/env python3
"""Generate reproducible model-occurrence sidecars for ``perf_ab_rollup.py``.

The generator traces the saved full-model graphs to count exact canonical
kernel shapes, benchmarks each unique non-fusible operation once, and writes
one sidecar per model.  Extern measurements are cached with a hardware and
software fingerprint so interrupted runs can resume without mixing results
from incompatible environments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from gpu_lock import device_kind_from_name

SCHEMA_VERSION = 1
BENCHMARK_VERSION = 1
N_WARMUP = 10
N_REP = 50


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical_json(value) + b"\n")
    os.replace(temporary, path)


def _git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _driver_version() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


def _benchmark_source_digest() -> str:
    paths = [
        Path(__file__),
        ROOT / "capture_hook.py",
        ROOT / "full_graph_harness.py",
        ROOT / "scripts" / "model_attribution.py",
        ROOT / "scripts" / "model_graph_accounting.py",
    ]
    return _digest(
        {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        }
    )


def environment_fingerprint(device: int = 0) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to benchmark external operations")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    device_name = torch.cuda.get_device_name(device)
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "benchmark_source_digest": _benchmark_source_digest(),
        "method": "cudagraph_replay_min",
        "warmup": N_WARMUP,
        "repetitions": N_REP,
        "device_name": device_name,
        "device_kind": device_kind_from_name(device_name),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "torch_version": torch.__version__,
        "torch_git_version": getattr(torch.version, "git_version", None),
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": _driver_version(),
    }


def discover_models(
    corpus_root: Path,
    *,
    suite: str | None = None,
    mode: str | None = None,
    models: set[str] | None = None,
) -> list[dict]:
    root = corpus_root / "models"
    discovered = {}
    for graph in sorted(root.glob("**/full_graph_*.py")):
        relative = graph.parent.relative_to(root)
        if len(relative.parts) < 3:
            continue
        graph_suite, graph_mode = relative.parts[:2]
        model = "/".join(relative.parts[2:])
        if suite is not None and graph_suite != suite:
            continue
        if mode is not None and graph_mode != mode:
            continue
        if models and model not in models:
            continue
        identity = f"{graph_suite}/{graph_mode}/{model}"
        discovered[identity] = {
            "identity": identity,
            "suite": graph_suite,
            "mode": graph_mode,
            "model": model,
            "directory": graph.parent,
        }
    return [discovered[name] for name in sorted(discovered)]


def sidecar_filename(identity: str) -> str:
    readable = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in identity
    ).strip("_")
    readable = readable[:100] or "model"
    return f"{readable}.{hashlib.sha256(identity.encode()).hexdigest()[:12]}.json"


def _source_digest(model_dir: Path, corpus_root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(model_dir.glob("full_graph_*.py"))
    paths += sorted(model_dir.glob("full_graph_*.meta.json"))
    for path in sorted(paths):
        digest.update(str(path.relative_to(corpus_root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _cache_key(target: str, signature: str) -> str:
    return hashlib.sha256(f"{target}\0{signature}".encode()).hexdigest()


def load_cache(path: Path, fingerprint: dict) -> dict:
    if not path.is_file():
        return {
            "schema_version": SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "fingerprint_digest": _digest(fingerprint),
            "entries": {},
        }
    cache = json.loads(path.read_text())
    if (
        cache.get("schema_version") != SCHEMA_VERSION
        or cache.get("fingerprint_digest") != _digest(fingerprint)
        or cache.get("fingerprint") != fingerprint
        or not isinstance(cache.get("entries"), dict)
    ):
        raise ValueError(
            f"{path}: extern cache belongs to a different environment; "
            "use a new cache path"
        )
    return cache


def _fusible_map(occurrences: Counter) -> dict:
    nested = defaultdict(dict)
    for (pattern_hash, shape_hash), count in sorted(occurrences.items()):
        nested[pattern_hash][shape_hash] = count
    return dict(nested)


def _metadata_validation_error(graph: Path, corpus_root: Path) -> dict | None:
    metadata_path = graph.with_suffix(".meta.json")
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return {
            "graph": str(graph.relative_to(corpus_root)),
            "error": f"invalid metadata: {type(error).__name__}: {error}",
        }
    roundtrip = metadata.get("roundtrip")
    if isinstance(roundtrip, str) and roundtrip.startswith("failed"):
        return {
            "graph": str(graph.relative_to(corpus_root)),
            "error": roundtrip,
        }
    return None


def inventory_model(spec: dict, corpus_root: Path) -> dict:
    from model_attribution import collect_extern_points
    from model_graph_accounting import analyze_graph, trace_full_graph

    fusible = Counter()
    extern = Counter()
    handles = {}
    trace_errors = []
    graphs = sorted(spec["directory"].glob("full_graph_*.py"))
    for graph in graphs:
        relative = str(graph.relative_to(corpus_root))
        metadata_error = _metadata_validation_error(graph, corpus_root)
        if metadata_error is not None:
            trace_errors.append(metadata_error)
            continue
        try:
            graph_module = trace_full_graph(graph)
            accounting = analyze_graph(graph_module, str(graph), graph.stem)
            fusible.update(
                (occurrence.pattern_hash, occurrence.shape_hash)
                for occurrence in accounting.occurrences
            )
            for key, point in collect_extern_points(
                graph_module, graph=graph
            ).items():
                extern[key] += point["count"]
                handles.setdefault(key, point["graph"])
        except Exception as error:  # noqa: BLE001 - one bad graph is recorded
            trace_errors.append(
                {
                    "graph": relative,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return {
        "graphs": graphs,
        "fusible": fusible,
        "extern": extern,
        "handles": handles,
        "trace_errors": trace_errors,
        "source_digest": _source_digest(spec["directory"], corpus_root),
    }


def price_externs(
    inventory: dict,
    cache: dict,
    cache_path: Path,
    *,
    retry_failures: bool,
) -> None:
    from model_attribution import _bench_extern_graph_isolated

    missing_by_graph = defaultdict(list)
    node_to_key = {}
    for key in sorted(inventory["extern"]):
        target, signature = key
        entry = cache["entries"].get(_cache_key(target, signature))
        if entry is not None and (entry.get("us") is not None or not retry_failures):
            continue
        handle = inventory["handles"].get(key)
        if handle is None:
            cache["entries"][_cache_key(target, signature)] = {
                "target": target,
                "signature": signature,
                "us": None,
                "error": "no reproducible graph/node handle",
            }
            _atomic_json(cache_path, cache)
            continue
        graph_path, node_name = handle
        missing_by_graph[graph_path].append(node_name)
        node_to_key[(graph_path, node_name)] = key

    for graph_path, node_names in sorted(missing_by_graph.items()):
        results: dict[str, float] = {}
        failures: dict[str, str] = {}
        _bench_extern_graph_isolated(graph_path, node_names, results, failures)
        for node_name in node_names:
            target, signature = node_to_key[(graph_path, node_name)]
            us = results.get(node_name)
            cache["entries"][_cache_key(target, signature)] = {
                "target": target,
                "signature": signature,
                "us": us,
                "error": None if us is not None else failures.get(
                    node_name, "benchmark produced no result"
                ),
            }
        _atomic_json(cache_path, cache)


def build_sidecar(spec: dict, inventory: dict, cache: dict) -> dict:
    extern = []
    for (target, signature), count in sorted(inventory["extern"].items()):
        key = _cache_key(target, signature)
        cached = cache["entries"].get(key, {})
        extern.append(
            {
                "target": target,
                "signature": signature,
                "signature_hash": key[:12],
                "count": count,
                "baseline_us": cached.get("us"),
                "benchmark_error": cached.get("error"),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "suite": spec["suite"],
        "mode": spec["mode"],
        "model": spec["model"],
        "n_graphs": len(inventory["graphs"]),
        "source_digest": inventory["source_digest"],
        "fusible": _fusible_map(inventory["fusible"]),
        "extern": extern,
        "trace_errors": inventory["trace_errors"],
    }


def generate(
    specs: list[dict],
    *,
    corpus_root: Path,
    output_dir: Path,
    cache_path: Path,
    fingerprint: dict,
    resume: bool,
    retry_failures: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        path for path in output_dir.glob("*.json") if not path.name.startswith("_")
    ]
    if existing and not resume:
        raise ValueError(
            f"{output_dir}: output already contains sidecars; pass --resume"
        )
    cache = load_cache(cache_path, fingerprint)
    expected = {
        spec["identity"]: sidecar_filename(spec["identity"]) for spec in specs
    }
    stale = sorted(path.name for path in existing if path.name not in set(expected.values()))
    if stale:
        raise ValueError(
            f"{output_dir}: stale sidecars are not part of this run: {stale}"
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "incomplete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "better_benchmark_commit": _git_revision(ROOT),
        "corpus_root": str(corpus_root),
        "corpus_digest": _digest(
            {
                spec["identity"]: _source_digest(spec["directory"], corpus_root)
                for spec in specs
            }
        ),
        "hardware": fingerprint,
        "hardware_key": fingerprint["device_kind"],
        "extern_cache": str(cache_path),
        "extern_cache_fingerprint": cache["fingerprint_digest"],
        "expected_sidecars": expected,
    }
    manifest_path = output_dir / "_metadata.json"
    _atomic_json(manifest_path, manifest)

    sidecar_digests = {}
    for index, spec in enumerate(specs, 1):
        print(f"[{index}/{len(specs)}] {spec['identity']}", flush=True)
        inventory = inventory_model(spec, corpus_root)
        price_externs(
            inventory,
            cache,
            cache_path,
            retry_failures=retry_failures,
        )
        sidecar = build_sidecar(spec, inventory, cache)
        sidecar_path = output_dir / expected[spec["identity"]]
        _atomic_json(sidecar_path, sidecar)
        sidecar_digests[spec["identity"]] = _digest(sidecar)

    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["sidecar_digests"] = sidecar_digests
    manifest["priced_extern_points"] = sum(
        entry.get("us") is not None for entry in cache["entries"].values()
    )
    manifest["failed_extern_points"] = sum(
        entry.get("us") is None for entry in cache["entries"].values()
    )
    _atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate model occurrence and extern-price sidecars"
    )
    parser.add_argument("--corpus-root", type=Path, default=ROOT / "repros")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extern-cache", type=Path, required=True)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--suite")
    parser.add_argument("--mode")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()

    if not args.all and not (args.suite or args.mode or args.model):
        parser.error("pass --all or at least one --suite/--mode/--model filter")
    if not (args.corpus_root / "models").is_dir():
        parser.error(f"model corpus not found: {args.corpus_root / 'models'}")

    specs = discover_models(
        args.corpus_root,
        suite=args.suite,
        mode=args.mode,
        models=set(args.model),
    )
    if not specs:
        parser.error("no models matched the requested filters")

    fingerprint = environment_fingerprint(args.device)
    try:
        manifest = generate(
            specs,
            corpus_root=args.corpus_root,
            output_dir=args.output_dir,
            cache_path=args.extern_cache,
            fingerprint=fingerprint,
            resume=args.resume,
            retry_failures=args.retry_failures,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(
        f"Wrote {len(specs)} sidecars to {args.output_dir}; "
        f"manifest={_digest(manifest)}"
    )


if __name__ == "__main__":
    main()
