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
import fcntl
import gc
import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
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


def _release_cuda_memory(device: int) -> None:
    """Release graph-tracing allocations before isolated extern benchmarks."""
    import torch

    torch._dynamo.reset()
    gc.collect()
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def _visible_device_selector(device: int) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return str(device)
    selectors = [item.strip() for item in visible.split(",") if item.strip()]
    if device >= len(selectors):
        raise ValueError(
            f"logical GPU {device} is outside CUDA_VISIBLE_DEVICES={visible!r}"
        )
    return selectors[device]


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(_canonical_json(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def _cache_lock(path: Path):
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _persist_cache_entries(path: Path, cache: dict, updates: dict) -> None:
    if not updates:
        return
    with _cache_lock(path):
        persisted = load_cache(path, cache["fingerprint"])
        for key, update in updates.items():
            existing = persisted["entries"].get(key)
            if existing is None or existing.get("us") is None:
                persisted["entries"][key] = update
            persisted["claims"].pop(key, None)
        _atomic_json(path, persisted)
    cache["entries"] = persisted["entries"]
    cache["claims"] = persisted["claims"]


def _clear_cache_claims(path: Path, fingerprint: dict) -> None:
    with _cache_lock(path):
        cache = load_cache(path, fingerprint)
        if cache["claims"]:
            cache["claims"] = {}
            _atomic_json(path, cache)


def _release_owner_claims(path: Path, fingerprint: dict, owner: str) -> None:
    with _cache_lock(path):
        cache = load_cache(path, fingerprint)
        remaining = {
            key: claim_owner
            for key, claim_owner in cache["claims"].items()
            if claim_owner != owner
        }
        if remaining != cache["claims"]:
            cache["claims"] = remaining
            _atomic_json(path, cache)


def _claim_cache_entries(
    path: Path,
    cache: dict,
    keys: set[str],
    *,
    retry_failures: bool,
    owner: str,
) -> set[str]:
    claimed = set()
    with _cache_lock(path):
        persisted = load_cache(path, cache["fingerprint"])
        for key in sorted(keys):
            entry = persisted["entries"].get(key)
            resolved = entry is not None and (
                entry.get("us") is not None or not retry_failures
            )
            if resolved or key in persisted["claims"]:
                continue
            persisted["claims"][key] = owner
            claimed.add(key)
        if claimed:
            _atomic_json(path, persisted)
    cache["entries"] = persisted["entries"]
    cache["claims"] = persisted["claims"]
    return claimed


def _wait_for_cache_entries(
    path: Path,
    cache: dict,
    keys: set[str],
    *,
    timeout_s: float = 1800,
    stop_event=None,
) -> None:
    deadline = time.monotonic() + timeout_s
    pending = set(keys)
    while pending:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("extern pricing cancelled after another shard failed")
        persisted = load_cache(path, cache["fingerprint"])
        pending = {
            key
            for key in pending
            if not (
                persisted["entries"].get(key) is not None
                and key not in persisted["claims"]
            )
        }
        if not pending:
            cache["entries"] = persisted["entries"]
            cache["claims"] = persisted["claims"]
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for {len(pending)} claimed extern cache entries"
            )
        time.sleep(0.1)


def _logical_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return path.name


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
            "claims": {},
        }
    cache = json.loads(path.read_text())
    if (
        cache.get("schema_version") != SCHEMA_VERSION
        or cache.get("fingerprint_digest") != _digest(fingerprint)
        or cache.get("fingerprint") != fingerprint
        or not isinstance(cache.get("entries"), dict)
        or not isinstance(cache.get("claims", {}), dict)
    ):
        raise ValueError(
            f"{path}: extern cache belongs to a different environment; "
            "use a new cache path"
        )
    cache.setdefault("claims", {})
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
    device: int | str = 0,
    stop_event=None,
) -> None:
    from model_attribution import _bench_extern_graph_isolated

    missing_by_graph = defaultdict(list)
    node_to_key = {}
    unresolved = {}
    for key in sorted(inventory["extern"]):
        target, signature = key
        entry = cache["entries"].get(_cache_key(target, signature))
        if entry is not None and (entry.get("us") is not None or not retry_failures):
            continue
        handle = inventory["handles"].get(key)
        if handle is None:
            unresolved[_cache_key(target, signature)] = {
                "target": target,
                "signature": signature,
                "us": None,
                "error": "no reproducible graph/node handle",
            }
            continue
        graph_path, node_name = handle
        missing_by_graph[graph_path].append(node_name)
        node_to_key[(graph_path, node_name)] = key

    _persist_cache_entries(cache_path, cache, unresolved)
    owner = f"{os.getpid()}:{device}"
    for graph_path, node_names in sorted(missing_by_graph.items()):
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("extern pricing cancelled after another shard failed")
        graph_keys = {
            _cache_key(*node_to_key[(graph_path, node_name)])
            for node_name in node_names
        }
        claimed = _claim_cache_entries(
            cache_path,
            cache,
            graph_keys,
            retry_failures=retry_failures,
            owner=owner,
        )
        node_names = [
            node_name
            for node_name in node_names
            if _cache_key(*node_to_key[(graph_path, node_name)]) in claimed
        ]
        if node_names:
            results: dict[str, float] = {}
            failures: dict[str, str] = {}
            _bench_extern_graph_isolated(
                graph_path, node_names, results, failures, device=device
            )
            updates = {}
            for node_name in node_names:
                target, signature = node_to_key[(graph_path, node_name)]
                us = results.get(node_name)
                updates[_cache_key(target, signature)] = {
                    "target": target,
                    "signature": signature,
                    "us": us,
                    "error": None if us is not None else failures.get(
                        node_name, "benchmark produced no result"
                    ),
                }
            _persist_cache_entries(cache_path, cache, updates)
        _wait_for_cache_entries(
            cache_path,
            cache,
            graph_keys,
            stop_event=stop_event,
        )


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


def _generate_shard(
    specs: list[dict],
    *,
    corpus_root: Path,
    cache_path: Path,
    fingerprint: dict,
    retry_failures: bool,
    device: int,
    verify_fingerprint: bool,
    stop_event=None,
) -> tuple[list[tuple[str, dict]], set[str]]:
    if verify_fingerprint:
        import torch

        torch.cuda.set_device(device)
        actual_fingerprint = environment_fingerprint(device)
        if actual_fingerprint != fingerprint:
            raise ValueError(
                f"GPU {device} environment does not match the generation fingerprint"
            )
    cache = load_cache(cache_path, fingerprint)
    sidecars = []
    extern_keys = set()
    owner = f"{os.getpid()}:{_visible_device_selector(device)}"
    try:
        for index, spec in enumerate(specs, 1):
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("generation cancelled after another shard failed")
            print(
                f"[GPU {device} {index}/{len(specs)}] {spec['identity']}",
                flush=True,
            )
            inventory = inventory_model(spec, corpus_root)
            _release_cuda_memory(device)
            extern_keys.update(
                _cache_key(target, signature)
                for target, signature in inventory["extern"]
            )
            price_externs(
                inventory,
                cache,
                cache_path,
                retry_failures=retry_failures,
                device=_visible_device_selector(device),
                stop_event=stop_event,
            )
            sidecars.append((spec["identity"], build_sidecar(spec, inventory, cache)))
            del inventory
            _release_cuda_memory(device)
    finally:
        _release_owner_claims(cache_path, fingerprint, owner)
    return sidecars, extern_keys


def generate(
    specs: list[dict],
    *,
    corpus_root: Path,
    output_dir: Path,
    cache_path: Path,
    fingerprint: dict,
    resume: bool,
    retry_failures: bool,
    device: int = 0,
    devices: list[int] | None = None,
) -> dict:
    devices = devices or [device]
    if not devices or len(set(devices)) != len(devices) or any(item < 0 for item in devices):
        raise ValueError("devices must be a non-empty list of unique non-negative indices")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        path for path in output_dir.glob("*.json") if not path.name.startswith("_")
    ]
    if existing and not resume:
        raise ValueError(
            f"{output_dir}: output already contains sidecars; pass --resume"
        )
    cache = load_cache(cache_path, fingerprint)
    _clear_cache_claims(cache_path, fingerprint)
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
        "corpus_root": _logical_path(corpus_root),
        "corpus_digest": _digest(
            {
                spec["identity"]: _source_digest(spec["directory"], corpus_root)
                for spec in specs
            }
        ),
        "hardware": fingerprint,
        "hardware_key": fingerprint["device_kind"],
        "extern_cache": _logical_path(cache_path),
        "extern_cache_fingerprint": cache["fingerprint_digest"],
        "expected_sidecars": expected,
    }
    manifest_path = output_dir / "_metadata.json"
    _atomic_json(manifest_path, manifest)

    shards = [specs[index :: len(devices)] for index in range(len(devices))]
    active_shards = [
        (shard, shard_device)
        for shard, shard_device in zip(shards, devices, strict=True)
        if shard
    ]
    if len(active_shards) == 1:
        shard_results = [
            _generate_shard(
                active_shards[0][0],
                corpus_root=corpus_root,
                cache_path=cache_path,
                fingerprint=fingerprint,
                retry_failures=retry_failures,
                device=active_shards[0][1],
                verify_fingerprint=False,
            )
        ]
    else:
        context = multiprocessing.get_context("spawn")
        with context.Manager() as manager:
            stop_event = manager.Event()
            with ProcessPoolExecutor(
                max_workers=len(active_shards),
                mp_context=context,
            ) as executor:
                futures = [
                    executor.submit(
                        _generate_shard,
                        shard,
                        corpus_root=corpus_root,
                        cache_path=cache_path,
                        fingerprint=fingerprint,
                        retry_failures=retry_failures,
                        device=shard_device,
                        verify_fingerprint=True,
                        stop_event=stop_event,
                    )
                    for shard, shard_device in active_shards
                ]
                try:
                    shard_results = [
                        future.result() for future in as_completed(futures)
                    ]
                except BaseException:
                    stop_event.set()
                    for future in futures:
                        future.cancel()
                    raise

    sidecars = {}
    run_extern_keys = set()
    for shard_sidecars, shard_extern_keys in shard_results:
        run_extern_keys.update(shard_extern_keys)
        for identity, sidecar in shard_sidecars:
            if identity in sidecars:
                raise ValueError(f"duplicate generated sidecar: {identity}")
            sidecars[identity] = sidecar
    if set(sidecars) != set(expected):
        raise ValueError("generated sidecar inventory does not match requested models")

    sidecar_digests = {}
    for identity, sidecar in sorted(sidecars.items()):
        sidecar_path = output_dir / expected[identity]
        _atomic_json(sidecar_path, sidecar)
        sidecar_digests[identity] = _digest(sidecar)

    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["sidecar_digests"] = sidecar_digests
    cache = load_cache(cache_path, fingerprint)
    manifest["priced_extern_points"] = sum(
        cache["entries"].get(key, {}).get("us") is not None
        for key in run_extern_keys
    )
    manifest["failed_extern_points"] = sum(
        cache["entries"].get(key, {}).get("us") is None
        for key in run_extern_keys
    )
    _atomic_json(manifest_path, manifest)
    return manifest


def require_complete_extern_pricing(manifest: dict) -> None:
    failed = manifest.get("failed_extern_points")
    if isinstance(failed, bool) or not isinstance(failed, int) or failed < 0:
        raise ValueError("manifest has an invalid failed_extern_points count")
    if failed:
        raise ValueError(
            f"{failed} external-operation points are unpriced; rerun with "
            "--resume --retry-failures after fixing the cause"
        )


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
    parser.add_argument(
        "--devices",
        help="Comma-separated GPU indices; models are sharded across these devices",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument(
        "--allow-unpriced-externs",
        action="store_true",
        help="Write a successful result even when extern pricing is incomplete",
    )
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

    try:
        devices = (
            [int(item.strip()) for item in args.devices.split(",")]
            if args.devices
            else [args.device]
        )
    except ValueError:
        parser.error("--devices must be a comma-separated list of GPU indices")
    if not devices or len(set(devices)) != len(devices) or any(item < 0 for item in devices):
        parser.error("--devices must contain unique non-negative GPU indices")

    fingerprint = environment_fingerprint(devices[0])
    try:
        manifest = generate(
            specs,
            corpus_root=args.corpus_root,
            output_dir=args.output_dir,
            cache_path=args.extern_cache,
            fingerprint=fingerprint,
            resume=args.resume,
            retry_failures=args.retry_failures,
            devices=devices,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if not args.allow_unpriced_externs:
        try:
            require_complete_extern_pricing(manifest)
        except ValueError as error:
            parser.error(str(error))
    print(
        f"Wrote {len(specs)} sidecars to {args.output_dir}; "
        f"manifest={_digest(manifest)}"
    )


if __name__ == "__main__":
    main()
