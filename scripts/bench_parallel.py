"""
Parallel benchmark runner — distributes repros across all available GPUs.

Each GPU gets one or more persistent worker processes. Workers stay alive across
repros to avoid per-repro torch startup overhead, and coordinate GPU timing with
per-GPU locks.

Usage:
    python scripts/bench_parallel.py repros/canonical/
    python scripts/bench_parallel.py repros/canonical/ --device-kind B200 --update-perf
    python scripts/bench_parallel.py repro1.py repro2.py repro3.py

# Architecture & performance notes
# ==================================
#
# Bottleneck analysis (measured on typical 1500-repro sweep):
#   - Per-repro time breakdown: ~20-40s compilation, ~2-5s timing (do_bench).
#   - GPU utilization during compilation: <5% (CPU-bound graph lowering + Triton
#     codegen). The GPU is idle while we hold the lock.
#   - Python+CUDA startup: ~3s per subprocess spawn (torch import + CUDA init).
#
# Current optimization: persistent worker subprocess per GPU (avoids the 3s
# startup per repro). Worker stays alive across repros, only respawns on
# CUDA errors.
#
# Phase-split optimization (compile-then-time):
# ---
# Ideally we would: (1) compile N repros in parallel on a CPU pool, then
# (2) acquire GPU lock around benchmark timing phases. Use --strict-gpu-lock
# to hold a shared setup lock and upgrade to exclusive for timing/autotune.
#
# Challenge: torch.compile uses lazy compilation — the actual Triton kernel
# generation and autotuning happen during the first few forward() calls, not
# at the torch.compile() call site. This means compilation and GPU execution
# are interleaved. Splitting them requires either:
#   a) Using torch._inductor.compile_fx directly to force AOT compilation
#      (fragile internal API, breaks across PT versions), or
#   b) Running warmup calls on GPU without the strict benchmark lock (risks
#      measurement interference from concurrent CUDA work on same GPU), or
#   c) A two-subprocess design: compile in one process, serialize the compiled
#      artifact, load+time in another (significant engineering, but doable
#      via torch._inductor's cache).
#
# Implemented optimizations:
# 1. Persistent worker subprocess (already done — avoids 3s Python+CUDA
#    startup per repro).
# 2. Pre-compilation pipeline (NEW): while one repro is being timed on the
#    GPU, the worker pre-imports the next repro module and prepares inputs
#    (overlaps CPU work with GPU timing of previous repro).
# 3. Inductor cache sharing (NEW): workers share a common inductor cache dir.
#    If the same repro was compiled before (e.g., on a previous run), the
#    cached kernels are reused, skipping compilation entirely.
# 4. Adaptive worker count (NEW): --max-workers can exceed GPU count. Extra
#    workers prepare repros speculatively; benchmark timing serializes on the
#    per-GPU lock. --strict-gpu-lock additionally serializes CUDA setup,
#    warmup, and capture against timing windows.
#
# Future work:
# - torch._inductor cache-based phase split: compile with TORCH_COMPILE_DEBUG
#   to populate cache, then time in a separate step using cached kernels.
# - Triton autotuning cache: pre-populate ~/.triton/cache so autotuning
#   doesn't need to re-run during timing.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gpu_lock import gpu_lock_for_kind, discover_gpus, matching_gpus


# Shared inductor cache — all workers read/write the same cache so repeated
# compilations of the same repro (e.g., across runs or retries) hit cache.
_SHARED_CACHE_DIR = os.environ.get(
    "TORCHINDUCTOR_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "bench_parallel_inductor_cache"),
)
_RESERVED_TOP_LEVEL_KEYS = {"_metadata", "__failures__", "__summary__"}
_RESERVED_RESULT_LABELS = {"__graph__"}

# Separator used to encode a (dir, shape_label) task key for --oracles
# --all-shapes sharding. The task unit becomes one (dir, shape) point so a
# many-shape dir spreads across all GPUs instead of pinning one worker. The
# parent regroups results back under their dir before aggregation, preserving
# the per-dir output contract that model_graph_accounting consumes.
_SHAPE_TASK_SEP = "::SHAPE::"
# Label used for the single point of a dir with no shape configs.
_DEFAULT_SHAPE_TOKEN = "__default__"


def _make_shape_task_key(dir_path: str, shape_label: str) -> str:
    """Encode a (dir, shape_label) sharded oracle task key."""
    return f"{dir_path}{_SHAPE_TASK_SEP}{shape_label}"


def _split_shape_task_key(task_key: str) -> tuple[str, str | None]:
    """Decode a sharded task key into (dir_path, shape_label).

    A plain dir path (no separator) returns (dir_path, None) so callers can
    treat un-sharded tasks transparently.
    """
    if _SHAPE_TASK_SEP in task_key:
        dir_path, label = task_key.rsplit(_SHAPE_TASK_SEP, 1)
        return dir_path, label
    return task_key, None


def _expand_oracle_shape_tasks(dirs: list[Path]) -> list[str]:
    """Expand oracle dirs into (dir, shape_label) task keys for sharding.

    Each dir's shapes.json points become individual tasks so the worker pool
    spreads them across all GPUs. A dir with no shape configs yields a single
    ``<dir>::SHAPE::__default__`` task. The numerics gate + fresh static
    compile (the per-shape dynamo reset, commit 4ca6d532b) still run once per
    task, so sharding does NOT reintroduce cross-shape dynamo reuse.
    """
    from repro_harness import load_shape_configs

    tasks: list[str] = []
    for d in dirs:
        repro_file = d / "repro.py"
        try:
            configs = load_shape_configs(str(repro_file))
        except Exception:
            configs = None
        if configs:
            for label in configs:
                tasks.append(_make_shape_task_key(str(d), label))
        else:
            tasks.append(_make_shape_task_key(str(d), _DEFAULT_SHAPE_TOKEN))
    return tasks


def _regroup_sharded_oracle_failures(sharded_failures: dict) -> dict:
    """Regroup per-(dir,shape) WORKER EXCEPTIONS back under their dir name.

    With per-shape sharding a worker exception fails ONE (dir,shape) task
    (keyed ``<dir>::SHAPE::<label>``) rather than the whole dir. A raised shape
    would otherwise vanish from the per-dir timings file. This returns
    ``{dir_name -> {failed_shapes:[...], n_failed_shapes, example_error}}`` so
    the oracle output can account for every queued shape. The CALLER decides
    whether each dir is still priced (then the record is an annotation) or
    has no valid floor (then it folds into ``__failures__``), using the
    aggregator's authoritative priced-dir set rather than guessing here.

    Note: numerics/CUDAGraph-warning/no-oracle statuses are RETURNED by
    bench_oracle (not raised), so they already appear as invalid-status points
    in all_results and are handled by the aggregator; only genuine Python
    exceptions reach sharded_failures and this function.
    """
    by_dir: dict[str, dict] = {}
    for task_key, failure in sharded_failures.items():
        dir_path, label = _split_shape_task_key(str(task_key))
        dir_name = _oracle_dir_name(dir_path)
        entry = by_dir.setdefault(dir_name, {
            "failed_shapes": [],
            "n_failed_shapes": 0,
            "example_error": None,
        })
        if label is not None:
            entry["failed_shapes"].append(label)
        entry["n_failed_shapes"] += 1
        if entry["example_error"] is None and isinstance(failure, dict):
            entry["example_error"] = failure.get("error") or failure.get("reason")
    return by_dir


def find_repros(paths: list[Path]) -> list[Path]:
    """Resolve paths to individual repro.py files."""
    repros = []
    for p in paths:
        if p.is_file() and p.name.endswith(".py"):
            repros.append(p)
        elif p.is_dir():
            repros.extend(sorted(p.rglob("repro.py")))
    return repros


def find_oracle_dirs(paths: list[Path]) -> list[Path]:
    """Resolve paths to canonical dirs that contain an oracle (oracle*.py).

    In --oracles mode the worker loads the oracle module from the canonical
    *directory* (via oracle_harness._load_oracle_module), so the task unit is
    the directory, not a repro.py file. We pass the directory path itself.
    """
    dirs: list[Path] = []
    seen: set[str] = set()

    def _has_oracle(d: Path) -> bool:
        return (d / "oracle.py").exists() or bool(list(d.glob("oracle_*.py")))

    for p in paths:
        if p.is_dir():
            # A canonical dir directly, or a parent containing many of them.
            if _has_oracle(p):
                candidates = [p]
            else:
                candidates = [c.parent for c in sorted(p.rglob("oracle.py"))]
                candidates += [
                    c.parent for c in sorted(p.rglob("oracle_*.py"))
                ]
        elif p.is_file() and p.name == "repro.py":
            candidates = [p.parent]
        else:
            continue
        for c in candidates:
            key = str(c.resolve())
            if key not in seen and _has_oracle(c):
                seen.add(key)
                dirs.append(c)
    return dirs


def find_full_graphs(paths: list[Path]) -> list[Path]:
    """Resolve paths to saved full_graph_*.py files."""
    graphs = []
    seen = set()
    for p in paths:
        if p.is_file():
            candidates = [p]
        elif p.is_dir():
            candidates = sorted(p.rglob("full_graph_*.py"))
        else:
            continue

        for candidate in candidates:
            if (
                candidate.is_file()
                and candidate.name.startswith("full_graph_")
                and candidate.suffix == ".py"
            ):
                resolved = candidate.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    graphs.append(candidate)
    return graphs


def _task_display_name(task_path: str) -> str:
    # Sharded oracle task keys carry a ::SHAPE::<label> suffix; show dir[label].
    dir_part, shape_label = _split_shape_task_key(str(task_path))
    if shape_label is not None:
        base = Path(dir_part).name
        if shape_label == _DEFAULT_SHAPE_TOKEN:
            return base
        return f"{base}[{shape_label}]"
    path = Path(task_path)
    if path.name == "repro.py":
        return path.parent.name
    if path.is_dir():
        return path.name
    if path.name.startswith("full_graph_") and path.suffix == ".py":
        try:
            from full_graph_harness import infer_full_graph_source

            source = infer_full_graph_source(path)
            parts = [
                source.get("kind"),
                source.get("suite"),
                source.get("mode"),
                source.get("model"),
                source.get("graph"),
            ]
            return "/".join(str(part) for part in parts if part)
        except Exception:
            pass
    return f"{path.parent.name}/{path.name}"


def _benchmark_entry_name(entry) -> str | None:
    if isinstance(entry, str):
        return entry
    if not isinstance(entry, dict):
        return None
    return entry.get("repro") or entry.get("name")


def load_benchmark_set(
    benchmark_set: Path,
    *,
    canonical_dir: Path = Path("repros/canonical"),
) -> tuple[list[Path], int]:
    """Load repro paths from either the old or current benchmark-set schema."""
    data = json.loads(benchmark_set.read_text())
    if isinstance(data, list):
        entries = data
    else:
        entries = data.get("patterns") or data.get("benchmarks") or []

    repros = []
    missing = []
    for entry in entries:
        name = _benchmark_entry_name(entry)
        if not name:
            continue
        repro_path = canonical_dir / name / "repro.py"
        if repro_path.exists():
            repros.append(repro_path)
        else:
            missing.append(name)
    if missing:
        preview = ", ".join(missing[:10])
        suffix = f" and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"{benchmark_set} references {len(missing)} missing canonical repros: "
            f"{preview}{suffix}"
        )
    return sorted(set(repros)), len(entries)


def _filter_gpus(gpus: list[dict[str, str]], selected: str | None) -> list[dict[str, str]]:
    if not selected:
        return gpus

    wanted = {item.strip() for item in selected.split(",") if item.strip()}
    if not wanted:
        return gpus

    return [gpu for gpu in gpus if gpu["index"] in wanted]


def _compute_worker_count(
    *,
    num_repros: int,
    num_gpus: int,
    max_workers: int | None,
    workers_per_gpu: int | None,
    workload_kind: str | None = None,
) -> tuple[int, int]:
    """Return (worker_count, effective_workers_per_gpu)."""
    if num_gpus <= 0:
        raise ValueError("No matching GPUs found")
    if max_workers is not None and max_workers < 1:
        raise ValueError("--max-workers must be >= 1")
    if workers_per_gpu is not None and workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be >= 1")
    if num_repros <= 0:
        return 0, workers_per_gpu or 1

    if workers_per_gpu is None:
        if max_workers is None:
            # Default: parallelize compilation across workers. Timing serializes
            # on the per-GPU lock, but COMPILING workers still allocate GPU memory
            # (tracing / autotune / intermediates), so the safe default depends on
            # per-worker memory:
            #   - repro/oracle kernels: tiny footprint -> pack several per GPU,
            #     bounded by CPU cores (the real ceiling on compile parallelism).
            #   - full model graphs: a single compile can hold large GPU memory;
            #     N concurrent model compiles can co-resident-OOM -> stay conservative.
            #     (A memory-aware packer that measures per-graph footprint is future work.)
            if workload_kind == "full_graph":
                effective_workers_per_gpu = 1
            else:
                try:
                    n_cores = len(os.sched_getaffinity(0))
                except AttributeError:
                    n_cores = os.cpu_count() or num_gpus
                cores_per_gpu = max(1, n_cores // num_gpus)
                effective_workers_per_gpu = max(1, min(4, cores_per_gpu))
        else:
            effective_workers_per_gpu = max(1, (max_workers + num_gpus - 1) // num_gpus)
    else:
        effective_workers_per_gpu = workers_per_gpu

    worker_capacity = num_gpus * effective_workers_per_gpu
    requested_workers = max_workers if max_workers is not None else worker_capacity
    return min(requested_workers, worker_capacity, num_repros), effective_workers_per_gpu


def _results_payload(
    all_results: dict,
    failures: dict | None = None,
    summary: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    """Return the JSON payload for --output.

    Top-level repro-path keys remain the success result schema consumed by
    bench_report.py. Failure/summary metadata lives under reserved keys so a
    partially failed sweep is resumable and diagnosable without breaking the
    existing happy-path consumers.
    """
    payload = {}
    if metadata:
        payload["_metadata"] = metadata
    payload.update(all_results)
    if failures:
        payload["__failures__"] = failures
    if summary:
        payload["__summary__"] = summary
    return payload


def _metric_result_items(results: dict):
    """Yield only per-shape/per-run metric entries from one result payload."""
    for label, result in results.items():
        if label in _RESERVED_RESULT_LABELS:
            continue
        if isinstance(result, dict):
            yield label, result


def _success_items(payload: dict):
    for key, value in payload.items():
        if key not in _RESERVED_TOP_LEVEL_KEYS:
            yield key, value


def _infer_workload_kind_from_payload(
    successes: dict,
    failures: dict | None = None,
) -> str | None:
    saw_full_graph = False
    saw_repro = False

    def record_path(path: str) -> None:
        nonlocal saw_full_graph, saw_repro
        name = Path(path).name
        if name.startswith("full_graph_") and name.endswith(".py"):
            saw_full_graph = True
        elif name == "repro.py":
            saw_repro = True

    for key, value in successes.items():
        record_path(str(key))
        if not isinstance(value, dict):
            continue
        if "__graph__" in value:
            saw_full_graph = True
        else:
            saw_repro = True

    for key, value in (failures or {}).items():
        record_path(str(key))
        if isinstance(value, dict):
            repro = value.get("repro")
            if repro:
                record_path(str(repro))

    if saw_full_graph and not saw_repro:
        return "full_graph"
    if saw_repro and not saw_full_graph:
        return "repro"
    if saw_full_graph and saw_repro:
        return "mixed"
    return None


def _run_metadata(*, workload_kind: str | None, n_results: int) -> dict:
    import subprocess

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.strip()
    metadata = {
        "schema_version": 1,
        "tool": "scripts/bench_parallel.py",
        "commit": commit,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_repros": n_results,
        "n_results": n_results,
    }
    if workload_kind:
        metadata["workload_kind"] = workload_kind
    return metadata


def _failure_status_counts(failures: dict | None) -> tuple[int, int]:
    """Return (failed, skipped) counts from a failure map."""
    failures = failures or {}
    skipped = sum(
        1
        for failure in failures.values()
        if isinstance(failure, dict) and failure.get("status") == "skipped"
    )
    return len(failures) - skipped, skipped


def _atomic_write_json(path: Path, payload: dict) -> None:
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        json.dump(payload, tmp, indent=2)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def _write_results_output(
    output_path: Path,
    all_results: dict,
    failures: dict,
    *,
    total: int,
    done: int,
    failed: int,
    elapsed: float,
    skipped: int = 0,
    workload_kind: str | None = None,
):
    _atomic_write_json(
        output_path,
        _results_payload(
            all_results,
            failures,
            {
                "total": total,
                "ok": done,
                "failed": failed,
                "skipped": skipped,
                "elapsed_s": elapsed,
            },
            _run_metadata(workload_kind=workload_kind, n_results=len(all_results)),
        ),
    )


def _oracle_dir_name(task_key: str) -> str:
    """Canonical dir basename for an oracle task key (a dir path).

    Tolerates a sharded ``::SHAPE::<label>`` suffix (results are regrouped to
    bare dir paths before aggregation, but strip defensively).
    """
    dir_part, _label = _split_shape_task_key(str(task_key))
    path = Path(dir_part)
    return path.parent.name if path.name == "repro.py" else path.name


def _median(values: list[float]) -> float:
    """Median of a non-empty numeric list."""
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _aggregate_oracle_timings(all_results: dict) -> dict:
    """Flatten per-dir oracle bench results into the model-tool timings schema.

    model_graph_accounting.py --timings consumes {dir_name -> {...}}. Each
    point is keyed by SHAPE HASH (the trailing underscore token of the result
    label, which is ``<model>_<shape_hash[:8]>``) so the model tool can match
    each occurrence to the oracle timing for ITS shape. A dir's shape points
    can span ~144x (e.g. 7us..1069us), so booking every occurrence at a single
    per-dir number badly mis-prices the roll-up.

    Emits per dir:
      - ``points_by_shape``: {shape_hash -> {oracle_us, compile_us, status,
        ratio, fallback}} — the authoritative per-shape data the roll-up uses.
      - ``points``: the original {label -> {...}} dict (back-compat).
      - ``oracle_us`` / ``compile_us``: representative MEDIAN across valid
        points (fallback only, used when an occurrence's shape_hash has no
        timed point; NOT the floor for matched occurrences).

    Floor validity excludes ``_INVALID_STATUSES`` AND ``BAD_ORACLE`` (an oracle
    slower than compile is not a valid floor). Dirs with no valid point are
    NOT silently dropped: they are recorded under the top-level ``__failures__``
    key with a reason, so the file accounts for every processed dir.
    """
    from oracle_harness import _INVALID_STATUSES

    # Local exclusion set; do not mutate the shared frozenset.
    floor_excluded = _INVALID_STATUSES | {"BAD_ORACLE"}

    # Sharding pre-pass: per-(dir,shape) tasks produce many result payloads
    # that map to the SAME dir. Merge their per-shape label entries into one
    # payload per dir BEFORE pricing, so each dir is aggregated once (a plain
    # ``flat[dir]=entry`` would otherwise let the last shape clobber the rest).
    # Bare dir keys (un-sharded) pass through unchanged: one key per dir.
    merged: dict[str, dict] = {}
    for task_key, results in all_results.items():
        dir_name = _oracle_dir_name(str(task_key))
        bucket = merged.setdefault(dir_name, {})
        if isinstance(results, dict):
            for label, point in results.items():
                # Copy only per-shape metric entries (dict-valued labels), the
                # same items _metric_result_items downstream will price.
                if isinstance(point, dict):
                    bucket[label] = point
    all_results = merged

    flat: dict[str, dict] = {}
    failures: dict[str, dict] = {}
    for task_key, results in all_results.items():
        dir_name = _oracle_dir_name(str(task_key))
        points: dict[str, dict] = {}
        points_by_shape: dict[str, dict] = {}
        valid_us: list[float] = []
        valid_compile: list[float] = []
        n_total_points = 0
        n_bad_oracle = 0
        for label, point in _metric_result_items(results):
            n_total_points += 1
            status = point.get("status")
            us = point.get("oracle_us")
            compile_us = point.get("compile_us")
            ratio = point.get("ratio")
            # dispatch.fallback marks a cross-hardware fallback (no matching
            # tuned impl for the running GPU); preserve it for visibility.
            fallback = None
            dispatch = point.get("dispatch")
            if isinstance(dispatch, dict):
                fallback = dispatch.get("fallback")
            points[label] = {
                "oracle_us": us,
                "compile_us": compile_us,
                "ratio": ratio,
                "status": status,
            }
            # Key per-shape data by the trailing shape-hash token of the label.
            shape_hash = label.rsplit("_", 1)[-1]
            shape_entry = {
                "oracle_us": us,
                "compile_us": compile_us,
                "ratio": ratio,
                "status": status,
                "fallback": fallback,
            }
            existing = points_by_shape.get(shape_hash)
            # If two labels collapse to the same shape_hash, keep the valid /
            # faster oracle point (prefer a usable floor over a bad one).
            if existing is None or _prefer_shape_point(shape_entry, existing,
                                                        floor_excluded):
                points_by_shape[shape_hash] = shape_entry
            if status == "BAD_ORACLE":
                n_bad_oracle += 1
            if status not in floor_excluded and isinstance(us, (int, float)):
                valid_us.append(us)
                if isinstance(compile_us, (int, float)):
                    valid_compile.append(compile_us)
        if not valid_us:
            # Account for the dir instead of dropping it.
            if n_total_points == 0:
                reason = "no_points"
            elif n_bad_oracle == n_total_points:
                reason = "all_bad_oracle"
            else:
                reason = "no_valid_point"
            failures[dir_name] = {
                "reason": reason,
                "n_points": n_total_points,
                "n_bad_oracle": n_bad_oracle,
                "points": points,
            }
            continue
        entry = {
            # Representative fallback value: MEDIAN (not min) of valid points.
            "oracle_us": round(_median(valid_us), 2),
            "n_points": len(valid_us),
            "points": points,
            "points_by_shape": points_by_shape,
        }
        if valid_compile:
            entry["compile_us"] = round(_median(valid_compile), 2)
        flat[dir_name] = entry
    if failures:
        flat["__failures__"] = failures
    return flat


def _prefer_shape_point(candidate: dict, existing: dict, floor_excluded) -> bool:
    """True if ``candidate`` is a better per-shape point than ``existing``.

    Prefer a floor-valid point over an invalid one; among two valid points
    prefer the faster oracle (the achievable floor for that shape).
    """
    cand_us = candidate.get("oracle_us")
    exist_us = existing.get("oracle_us")
    cand_valid = (candidate.get("status") not in floor_excluded
                  and isinstance(cand_us, (int, float)))
    exist_valid = (existing.get("status") not in floor_excluded
                   and isinstance(exist_us, (int, float)))
    if cand_valid != exist_valid:
        return cand_valid
    if cand_valid and exist_valid:
        return cand_us < exist_us
    # Both invalid: keep whichever has a numeric oracle_us, else keep existing.
    return isinstance(cand_us, (int, float)) and not isinstance(exist_us, (int, float))


def _write_oracle_timings_output(
    output_path: Path, all_results: dict, failures: dict | None = None
) -> dict:
    """Write flat {dir_name -> {oracle_us,...}} for model_graph_accounting.

    ``failures`` (per-shape worker exceptions, keyed by sharded task key) are
    regrouped per dir so the file accounts for every queued shape: dirs that
    lost some shapes keep their priced entry with a ``shape_failures`` note;
    dirs that lost ALL shapes are added to ``__failures__``.
    """
    timed = _aggregate_oracle_timings(all_results)
    if failures:
        shape_fail = _regroup_sharded_oracle_failures(failures)
        # Authoritative "still priced?" signal: a dir that produced a valid
        # floor is a top-level non-reserved key in the aggregator output. A dir
        # only present under __failures__ (no valid point) is NOT priced — its
        # raised shapes must still be accounted, never dropped.
        priced_dirs = {
            k for k in timed if k not in _RESERVED_TOP_LEVEL_KEYS
        }
        agg_failures = timed.setdefault("__failures__", {})
        for dir_name, entry in shape_fail.items():
            note = {
                "n_failed_shapes": entry["n_failed_shapes"],
                "failed_shapes": entry["failed_shapes"],
                "example_error": entry["example_error"],
            }
            if dir_name in priced_dirs:
                # Dir still priced from its surviving valid shapes — annotate.
                timed[dir_name]["shape_failures"] = note
            else:
                # Dir has no valid floor. FOLD the raised-shape accounting into
                # whatever failure entry exists (the aggregator may already
                # record no_valid_point/all_bad_oracle for invalid-status
                # points) so the count never vanishes; create one otherwise.
                existing = agg_failures.get(dir_name)
                if existing is None:
                    agg_failures[dir_name] = {
                        "reason": "all_shapes_failed",
                        **note,
                    }
                else:
                    existing.setdefault("shape_failures", note)
        if not agg_failures:
            timed.pop("__failures__", None)
    _atomic_write_json(output_path, timed)
    return timed


def _is_integer_non_bool_dtype_name(dtype_name: object) -> bool:
    text = str(dtype_name).removeprefix("torch.")
    return text != "bool" and (text.startswith("int") or text.startswith("uint"))


def _is_graph_inferred_integer_input(spec: dict) -> bool:
    if spec.get("exact") and "data" in spec:
        return True
    generator = spec.get("gen") or spec.get("generator") or {}
    return (
        generator.get("kind") in {"index", "permutation"}
        and spec.get("constraint_source") == "graph_inference"
    )


def _classify_full_graph_definition(
    definition,
    *,
    allow_unsafe: bool = False,
) -> list[dict[str, str]]:
    """Classify saved full graphs that are not safe to instantiate generically."""
    reasons: list[dict[str, str]] = []
    has_sidecar = bool((definition.metadata or {}).get("sidecar"))

    for attr_name, spec in (definition.tensor_attrs or {}).items():
        if spec.get("requires_exact") and not (spec.get("exact") and "data" in spec):
            reasons.append({
                "category": "requires_exact_attr",
                "reason": (
                    f"tensor attr {attr_name!r} is integer/bool state without "
                    "an exact payload"
                ),
                "hint": "Recapture this model so full_graph_NNN.meta.json contains exact tensor attr data.",
            })

    if allow_unsafe:
        return reasons

    for spec in definition.input_specs or []:
        name = str(spec.get("name", "<unnamed>"))
        kind = spec.get("kind")
        if not has_sidecar and kind == "symint":
            reasons.append({
                "category": "symbolic_dim",
                "reason": f"symbolic input {name!r} only has annotation defaults",
                "hint": "Recapture with metadata sidecars so symbolic dimensions have concrete values.",
            })
            continue
        if kind != "tensor":
            continue
        if not has_sidecar and spec.get("symbolic_dims"):
            reasons.append({
                "category": "symbolic_dim",
                "reason": f"tensor input {name!r} has symbolic dimensions from annotations",
                "hint": "Recapture with metadata sidecars so symbolic dimensions have concrete values.",
            })
        if _is_integer_non_bool_dtype_name(spec.get("dtype", "")) and not _is_graph_inferred_integer_input(spec):
            reasons.append({
                "category": "unsafe_integer_input",
                "reason": (
                    f"integer tensor input {name!r} lacks graph-inferred "
                    "index/permutation constraints"
                ),
                "hint": "Recapture with metadata sidecars or pass --allow-unsafe-full-graphs for exploratory runs.",
            })

    return reasons


def _full_graph_skip_failure(path: Path, reasons: list[dict[str, str]]) -> dict:
    reason_text = "; ".join(reason["reason"] for reason in reasons[:3])
    if len(reasons) > 3:
        reason_text += f"; {len(reasons) - 3} more"
    hints = []
    for reason in reasons:
        hint = reason.get("hint")
        if hint and hint not in hints:
            hints.append(hint)
    return {
        "status": "skipped",
        "category": reasons[0]["category"] if reasons else "unsupported_full_graph",
        "repro": str(path),
        "reason": reason_text,
        "hint": " ".join(hints),
    }


def _full_graph_preflight_error(path: Path, exc: Exception) -> dict:
    return {
        "status": "failed",
        "category": "preflight_error",
        "repro": str(path),
        "exception_type": type(exc).__name__,
        "error": str(exc)[:1000],
        "reason": "failed to parse saved full graph before queueing",
        "hint": "Inspect the graph module or recapture it with current exporter metadata.",
    }


def _preflight_full_graphs(
    repros: list[Path],
    *,
    allow_unsafe: bool = False,
) -> tuple[list[Path], dict[str, dict]]:
    from full_graph_harness import load_full_graph_definition

    queueable: list[Path] = []
    failures: dict[str, dict] = {}
    for repro_path in repros:
        try:
            definition = load_full_graph_definition(repro_path)
            reasons = _classify_full_graph_definition(
                definition,
                allow_unsafe=allow_unsafe,
            )
        except Exception as exc:
            failures[str(repro_path)] = _full_graph_preflight_error(repro_path, exc)
            continue
        if reasons:
            failures[str(repro_path)] = _full_graph_skip_failure(repro_path, reasons)
        else:
            queueable.append(repro_path)
    return queueable, failures


def _merge_into_baseline(
    baseline_path: Path,
    new_results: dict,
    new_failures: dict,
    *,
    workload_kind: str | None = None,
):
    """Merge new benchmark results into an existing baseline JSON file."""
    import fcntl

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = baseline_path.with_name(f"{baseline_path.name}.lock")
    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            return _merge_into_baseline_locked(
                baseline_path,
                new_results,
                new_failures,
                workload_kind=workload_kind,
            )
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _merge_into_baseline_locked(
    baseline_path: Path,
    new_results: dict,
    new_failures: dict,
    *,
    workload_kind: str | None = None,
):
    """Merge new benchmark results into an existing baseline JSON file."""
    import subprocess as _sp

    if not baseline_path.exists():
        print(f"[merge-into] {baseline_path} does not exist, writing fresh")
        commit = _sp.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        failed_count, skipped_count = _failure_status_counts(new_failures)
        payload = _results_payload(
            new_results, new_failures,
            {
                "total": len(new_results) + len(new_failures),
                "ok": len(new_results),
                "failed": failed_count,
                "skipped": skipped_count,
            },
            {
                "schema_version": 1,
                "tool": "scripts/bench_parallel.py",
                "commit": commit,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "n_repros": len(new_results),
                "n_results": len(new_results),
                **({"workload_kind": workload_kind} if workload_kind else {}),
            },
        )
        _atomic_write_json(baseline_path, payload)
        print(f"[merge-into] Wrote {baseline_path} ({len(new_results)} repros)")
        return

    existing = json.loads(baseline_path.read_text())
    old_failures = existing.pop("__failures__", {}) or {}
    old_summary = existing.pop("__summary__", {}) or {}
    old_meta = existing.pop("_metadata", {}) or {}
    metadata_kind = old_meta.get("workload_kind")
    content_kind = _infer_workload_kind_from_payload(
        existing,
        old_failures,
    )
    if metadata_kind and content_kind and metadata_kind != content_kind:
        raise ValueError(
            f"Baseline workload_kind metadata {metadata_kind!r} conflicts with "
            f"payload content {content_kind!r}"
        )
    existing_kind = metadata_kind or content_kind
    if (
        workload_kind
        and existing_kind
        and existing_kind != workload_kind
    ):
        raise ValueError(
            f"Cannot merge {workload_kind!r} results into {existing_kind!r} baseline"
        )

    updated = 0
    for repro_path, results in new_results.items():
        existing[repro_path] = results
        old_failures.pop(repro_path, None)
        updated += 1

    for repro_path, failure in new_failures.items():
        old_failures[repro_path] = failure
        existing.pop(repro_path, None)

    commit = _sp.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
    ).stdout.strip()
    old_meta["commit"] = commit
    old_meta["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    old_meta.setdefault("schema_version", 1)
    old_meta.setdefault("tool", "scripts/bench_parallel.py")
    if workload_kind:
        old_meta["workload_kind"] = workload_kind
    successes = dict(_success_items(existing))
    old_meta["n_repros"] = len(successes)
    old_meta["n_results"] = len(successes)
    old_meta["last_merge"] = f"updated {updated} repros"

    old_summary["total"] = len(successes) + len(old_failures)
    old_summary["ok"] = len(successes)
    failed_count, skipped_count = _failure_status_counts(old_failures)
    old_summary["failed"] = failed_count
    old_summary["skipped"] = skipped_count

    payload = _results_payload(successes, old_failures or None, old_summary, old_meta)
    _atomic_write_json(baseline_path, payload)
    print(f"[merge-into] Updated {updated} repros in {baseline_path} "
          f"(total: {len(successes)} ok, {failed_count} failed, {skipped_count} skipped)")


def worker(gpu_idx: str, task_queue: mp.Queue, result_queue: mp.Queue,
           args_dict: dict):
    """Worker process: holds GPU lock, benchmarks repros until queue is empty."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx

    import torch
    import torch._dynamo
    import torch._inductor.config as inductor_config
    from triton.testing import do_bench
    import importlib.util
    import math
    from byte_accounting import count_bytes_effective

    def load_and_bench(repro_path: str, all_shapes: bool, no_cd: bool,
                       n_warmup: int, n_rep: int) -> dict:
        spec = importlib.util.spec_from_file_location("repro", repro_path)
        mod = importlib.util.module_from_spec(spec)
        mod.device = torch.device
        mod.inf = math.inf
        mod.nan = math.nan
        spec.loader.exec_module(mod)

        repro_cls = mod.Repro
        make_inputs_fn = getattr(mod, "make_inputs", mod.make_inputs)

        # Determine shapes to run
        if all_shapes and hasattr(mod, "load_shape_configs"):
            configs = mod.load_shape_configs(repro_path)
            if configs:
                shape_names = list(configs.keys())
            else:
                shape_names = [None]
        else:
            shape_names = [None]

        results = {}
        for name in shape_names:
            label = name if name is not None else "default"
            if name is not None:
                from repro_harness import (
                    load_shape_configs,
                    make_inputs_from_config,
                    make_inputs_safely,
                )
                configs = load_shape_configs(repro_path)
                inputs = make_inputs_from_config(configs[name])
            else:
                inputs = make_inputs_safely(make_inputs_fn)

            instance = repro_cls()
            with torch.no_grad():
                eager_out = instance(*inputs)

            total_bytes = count_bytes_effective(instance, inputs)

            # SOL
            copy_elems = max(total_bytes // (2 * 4), 256)
            src = torch.empty(copy_elems, dtype=torch.float32, device="cuda")
            dst = torch.empty_like(src)
            sol_us = do_bench(
                lambda: torch.add(src, 1, out=dst),
                warmup=n_warmup,
                rep=n_rep,
                return_mode="min",
            ) * 1000
            del src, dst

            # Compiled
            torch._dynamo.reset()
            compiled = torch.compile(instance)
            with torch.no_grad():
                for _ in range(3):
                    compiled(*inputs)
                torch.cuda.synchronize()
            compiled_us = do_bench(
                lambda: compiled(*inputs),
                warmup=n_warmup,
                rep=n_rep,
                return_mode="min",
            ) * 1000

            # Coord descent
            cd_us = None
            if not no_cd:
                inductor_config.coordinate_descent_tuning = True
                torch._dynamo.reset()
                compiled_cd = torch.compile(instance)
                with torch.no_grad():
                    for _ in range(3):
                        compiled_cd(*inputs)
                    torch.cuda.synchronize()
                cd_us = do_bench(
                    lambda: compiled_cd(*inputs),
                    warmup=n_warmup,
                    rep=n_rep,
                    return_mode="min",
                ) * 1000
                inductor_config.coordinate_descent_tuning = False

            results[label] = {
                "compiled_us": compiled_us,
                "coord_descent_us": cd_us,
                "memcopy_sol_us": sol_us,
                "total_bytes": total_bytes,
                "gap_default": compiled_us / sol_us if sol_us > 0 else None,
                "gap_cd": cd_us / sol_us if (cd_us and sol_us > 0) else None,
            }

        return results

    # Main worker loop
    while True:
        try:
            repro_path = task_queue.get_nowait()
        except Exception:
            break

        start = time.time()
        try:
            results = load_and_bench(
                str(repro_path),
                all_shapes=args_dict["all_shapes"],
                no_cd=args_dict["no_cd"],
                n_warmup=args_dict["n_warmup"],
                n_rep=args_dict["n_rep"],
            )
            elapsed = time.time() - start
            result_queue.put({
                "repro": str(repro_path),
                "gpu": gpu_idx,
                "status": "ok",
                "elapsed": elapsed,
                "results": results,
            })
        except Exception as e:
            elapsed = time.time() - start
            result_queue.put({
                "repro": str(repro_path),
                "gpu": gpu_idx,
                "status": "failed",
                "elapsed": elapsed,
                "error": str(e)[:200],
            })


def main():
    parser = argparse.ArgumentParser(description="Parallel GPU benchmark runner")
    parser.add_argument("paths", nargs="*", type=Path,
                        help="repro.py files or directories to benchmark")
    parser.add_argument("--oracles", action="store_true",
                        help="Benchmark migrated v2 oracles (oracle.py per canonical "
                             "dir) instead of the torch.compile path. Mirrors "
                             "`python -m oracle_harness <dir> --bench`: runs the "
                             "numerics check gate, then bench_oracle per shape point. "
                             "With --output writes a flat {dir_name -> {oracle_us,...}} "
                             "mapping consumable by model_graph_accounting.py --timings.")
    parser.add_argument("--full-graphs", action="store_true",
                        help="Benchmark saved repros/models full_graph_*.py files "
                             "instead of partition repro.py files")
    parser.add_argument("--allow-unsafe-full-graphs", action="store_true",
                        help="Queue annotation-only integer/symbolic full graphs "
                             "instead of skipping them during preflight")
    parser.add_argument("--models-root", type=Path, default=Path("repros/models"),
                        help="Default graph search root for --full-graphs")
    parser.add_argument("--benchmark-set", type=Path, default=None,
                        help="Path to a frozen benchmark set JSON (e.g. benchmarks/v1.json)")
    parser.add_argument("--device-kind", default=None,
                        help="GPU kind to use (e.g. H100, B200). Default: all GPUs")
    parser.add_argument("--gpus", default=None,
                        help="Comma-separated physical GPU indices to use, e.g. '0' or '0,1'")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Max parallel workers (default: workers-per-gpu x matching GPUs)")
    parser.add_argument("--workers-per-gpu", type=int, default=None,
                        help="Max persistent worker subprocesses per matching GPU. "
                             "Default is workload-aware: min(4, cores/gpus) for "
                             "repro/oracle kernels (tiny footprint, compile-bound), "
                             "but 1 for --full-graphs (a model compile can hold large "
                             "GPU memory; concurrent compiles can OOM). Override "
                             "explicitly to pack more, or via --max-workers.")
    parser.add_argument("--all-shapes", action="store_true",
                        help="Benchmark all shapes from shapes.txt")
    parser.add_argument("--no-cd", action="store_true",
                        help="Skip coordinate descent tuning")
    parser.add_argument("--combo-kernels", action="store_true",
                        help="Enable inductor combo_kernels config")
    parser.add_argument("--multi-kernel", type=int, default=0,
                        help="Set inductor multi_kernel config (0=off, 1=on)")
    parser.add_argument("--update-perf", action="store_true",
                        help="Write results to each repro's perf.json")
    parser.add_argument("--hardware", default=None,
                        help="Hardware label for perf.json (auto-detected if not set)")
    parser.add_argument("--n-warmup", type=int, default=25)
    parser.add_argument("--n-rep", type=int, default=100)
    parser.add_argument("--share-cache", action="store_true", default=True,
                        help="Share inductor cache across workers (default: enabled)")
    parser.add_argument("--no-share-cache", dest="share_cache", action="store_false",
                        help="Disable shared inductor cache")
    parser.add_argument("--strict-gpu-lock", action="store_true",
                        help="Hold a shared GPU lock during setup/warmup/capture "
                             "and upgrade to exclusive for timing/autotune. "
                             "Slower, useful for variance validation.")
    parser.add_argument("--tag", default=None,
                        help="Tag for this run (e.g. 'baseline', 'my_fix'). Used to key results in perf.json for comparison.")
    parser.add_argument("--compare", type=str, nargs=2, metavar=("TAG_A", "TAG_B"),
                        help="Compare two tagged runs from perf.json (no benchmarking, just report)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write all results to a JSON file")
    parser.add_argument("--merge-into", type=Path, default=None,
                        help="Merge results into an existing baseline JSON (updates only the benchmarked repros)")
    args = parser.parse_args()

    # Compare mode: just read existing perf.json and diff
    if args.compare:
        if args.full_graphs:
            parser.error("--compare reads repro perf.json files and is not supported with --full-graphs")
        paths = args.paths or [Path("repros/canonical")]
        _run_compare(paths, args.compare[0], args.compare[1])
        return

    if args.full_graphs and args.benchmark_set:
        parser.error("--benchmark-set selects canonical repros and cannot be used with --full-graphs")
    if args.full_graphs and args.update_perf:
        parser.error("--update-perf writes repro perf.json files and is not supported with --full-graphs")
    if args.oracles and args.full_graphs:
        parser.error("--oracles and --full-graphs are mutually exclusive")
    if args.oracles and args.update_perf:
        parser.error("--update-perf writes repro perf.json files and is not supported with --oracles")
    if args.oracles and args.benchmark_set:
        parser.error("--benchmark-set is not supported with --oracles")

    # Load benchmark set if specified
    benchmark_entries = None
    if args.oracles:
        workload_kind = "oracle"
    elif args.full_graphs:
        workload_kind = "full_graph"
    else:
        workload_kind = "repro"
    oracle_n_dirs = None
    if args.oracles:
        oracle_paths = args.paths or [Path("repros/canonical")]
        oracle_dirs = find_oracle_dirs(oracle_paths)
        oracle_n_dirs = len(oracle_dirs)
        # Oracle bench is per shape point internally; --all-shapes is implied.
        args.all_shapes = True
        # SHARDING: make the task unit a (dir, shape) point rather than a whole
        # dir, so a big multi-shape dir (e.g. 469 shapes) spreads across all
        # GPUs instead of pinning ONE worker that compiles them serially while
        # peers idle. Order biggest-dirs-first so any residual tail is small.
        # Each task still runs the numerics gate + fresh static compile per
        # shape (commit 4ca6d532b); sharding is purely a scheduling change.
        repros = _expand_oracle_shape_tasks(oracle_dirs)
        # Group a dir's shape tasks contiguously, but place dirs with the most
        # shapes first. The worker pool pulls round-robin, so emitting the big
        # dirs' points up front lets all workers chew them in parallel from the
        # start; the small single-shape dirs drain at the end (cheap, no tail).
        from collections import Counter as _Counter
        _dir_task_count = _Counter(_split_shape_task_key(t)[0] for t in repros)
        repros.sort(
            key=lambda k: (
                -_dir_task_count[_split_shape_task_key(k)[0]],
                _split_shape_task_key(k)[0],
            )
        )
    elif args.full_graphs:
        graph_paths = args.paths or [args.models_root]
        repros = find_full_graphs(graph_paths)
    elif args.benchmark_set:
        repros, benchmark_entries = load_benchmark_set(args.benchmark_set)
        # Enable --all-shapes for benchmark set runs (repro_harness now merges
        # shape params from _default_make_inputs when shapes.json doesn't have them)
        args.all_shapes = True
        print(f"Benchmark set: {args.benchmark_set.name} "
              f"({benchmark_entries} points, {len(repros)} unique repros)")
    elif args.paths:
        repros = find_repros(args.paths)
    else:
        repros = find_repros([Path("repros/canonical")])

    if not repros:
        if args.oracles:
            print("No oracle.py dirs found.")
        elif args.full_graphs:
            print("No full_graph_*.py files found.")
        else:
            print("No repro.py files found.")
        return

    total_requested = len(repros)
    preflight_failures: dict[str, dict] = {}
    preflight_failed = 0
    skipped = 0
    if args.full_graphs:
        repros, preflight_failures = _preflight_full_graphs(
            repros,
            allow_unsafe=args.allow_unsafe_full_graphs,
        )
        preflight_failed, skipped = _failure_status_counts(preflight_failures)
        if preflight_failures:
            print(
                "Full-graph preflight: "
                f"{len(repros)} runnable, {skipped} skipped, {preflight_failed} failed"
            )
        if not repros:
            elapsed_total = 0.0
            print(
                f"\nDone: 0 ok, {preflight_failed} failed, {skipped} skipped "
                f"in {elapsed_total:.1f}s"
            )
            if args.output:
                _write_results_output(
                    args.output,
                    {},
                    preflight_failures,
                    total=total_requested,
                    done=0,
                    failed=preflight_failed,
                    skipped=skipped,
                    elapsed=elapsed_total,
                    workload_kind=workload_kind,
                )
                print(f"[output] Wrote {args.output}")
            if args.merge_into:
                _merge_into_baseline(
                    args.merge_into,
                    {},
                    preflight_failures,
                    workload_kind=workload_kind,
                )
            return

    gpus = _filter_gpus(matching_gpus(args.device_kind), args.gpus)
    try:
        n_workers, workers_per_gpu = _compute_worker_count(
            num_repros=len(repros),
            num_gpus=len(gpus),
            max_workers=args.max_workers,
            workers_per_gpu=args.workers_per_gpu,
            workload_kind=workload_kind,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.oracles:
        task_label, task_unit = "oracle shape-points", "point"
    elif args.full_graphs:
        task_label, task_unit = "full graphs", "graph"
    else:
        task_label, task_unit = "repros", "repro"
    if args.oracles:
        print(f"Benchmarking {len(repros)} {task_label} from {oracle_n_dirs} dirs "
              f"across {n_workers} workers on {len(gpus)} GPUs (sharded per shape)")
    else:
        print(f"Benchmarking {len(repros)} {task_label} across {n_workers} workers on {len(gpus)} GPUs")
    gpu_labels = [f"{g['index']}:{g['kind']}" for g in gpus]
    print(f"  GPUs: {', '.join(gpu_labels)}")
    print(f"  Workers per GPU cap: {workers_per_gpu}")
    if args.share_cache:
        print(f"  Shared inductor cache: {_SHARED_CACHE_DIR}")
    print(f"  Prefetch: enabled (overlaps module loading with GPU timing)")
    print()

    # Fill task queue (use regular queue since workers are threads)
    import queue
    task_queue = queue.Queue()
    for r in repros:
        task_queue.put(r)
    result_queue = queue.Queue()
    root = str(Path(__file__).resolve().parents[1])

    args_dict = {
        "root": root,
        "all_shapes": args.all_shapes,
        "no_cd": args.no_cd,
        "n_warmup": args.n_warmup,
        "n_rep": args.n_rep,
        "share_cache": args.share_cache,
        "strict_gpu_lock": args.strict_gpu_lock,
        "n_workers": n_workers,
        "combo_kernels": args.combo_kernels,
        "multi_kernel": args.multi_kernel,
        "workload_kind": workload_kind,
    }

    if args.oracles:
        # Reuse all the existing orchestration (task queue, GPU round-robin,
        # workers-per-gpu, INDUCTOR_GPU_BENCH_LOCK) — only swap the per-worker
        # subprocess script for the oracle check+bench loop.
        args_dict["_persistent_worker_script_factory"] = _oracle_worker_script

    # Use threads — the actual GPU work is in subprocess.Popen children,
    # so GIL isn't a bottleneck (threads just do pipe I/O).
    import threading
    workers = []
    for i in range(n_workers):
        gpu = gpus[i % len(gpus)]  # round-robin GPUs across workers
        t = threading.Thread(
            target=_locked_worker,
            args=(gpu, task_queue, result_queue, args_dict),
            daemon=True,
        )
        t.start()
        workers.append(t)

    # Collect results
    all_results = {}
    failures = dict(preflight_failures)
    done = 0
    failed = 0
    start_time = time.time()
    last_progress_time = time.time()

    while done + failed < len(repros):
        try:
            result = result_queue.get(timeout=30)
        except Exception:
            # No result in 30s — check worker health
            elapsed = time.time() - start_time
            alive = sum(1 for p in workers if p.is_alive())
            rate = (done + failed) / max(elapsed, 1) * 60
            remaining = len(repros) - done - failed
            eta = remaining / max(rate, 0.01)

            if alive == 0:
                print(f"\n[WARN] All workers dead! {done} ok, {failed} failed, "
                      f"{remaining} remaining. Aborting.")
                break

            # Periodic progress update every 30s of silence
            print(f"  [{done+failed}/{len(repros)}] ... {alive} workers alive, "
                  f"{rate:.1f} {task_unit}s/min, ETA {eta:.0f}min, elapsed {elapsed/60:.1f}min",
                  flush=True)
            continue

        repro_name = _task_display_name(result["repro"])
        if result["status"] == "ok":
            done += 1
            if args.oracles:
                from oracle_harness import _INVALID_STATUSES
                best_us = None
                for label, r in _metric_result_items(result["results"]):
                    us = r.get("oracle_us")
                    if (r.get("status") not in _INVALID_STATUSES
                            and isinstance(us, (int, float))
                            and (best_us is None or us < best_us)):
                        best_us = us
                metric_str = f"oracle={best_us:.1f}us" if best_us else "no-valid-point"
            else:
                best_gap = None
                for label, r in _metric_result_items(result["results"]):
                    gap = r.get("gap_cd") or r.get("gap_default")
                    if gap and (best_gap is None or gap > best_gap):
                        best_gap = gap
                metric_str = f"gap={best_gap:.2f}x" if best_gap else "gap=?"
            print(f"  [{done+failed}/{len(repros)}] OK  gpu={result['gpu']}  "
                  f"{result['elapsed']:.1f}s  {metric_str}  {repro_name}", flush=True)
            all_results[result["repro"]] = result["results"]
        else:
            failed += 1
            failure = {
                "status": "failed",
                "gpu": result.get("gpu"),
                "elapsed": result.get("elapsed"),
                "error": result.get("error", ""),
                "category": result.get("category", "runtime_error"),
                "reason": result.get("reason") or result.get("error", ""),
                "hint": result.get("hint") or "Inspect worker stderr or rerun this workload directly.",
            }
            if result.get("exception_type"):
                failure["exception_type"] = result["exception_type"]
            failures[result["repro"]] = failure
            print(f"  [{done+failed}/{len(repros)}] FAIL gpu={result['gpu']}  "
                  f"{result['elapsed']:.1f}s  {repro_name}: {result['error'][:80]}", flush=True)

        # Incremental save every 50 results
        if args.output and args.oracles and (done + failed) % 50 == 0:
            _write_oracle_timings_output(args.output, all_results, failures)
        elif args.output and (done + failed) % 50 == 0:
            _write_results_output(
                args.output,
                all_results,
                failures,
                total=total_requested,
                done=done,
                failed=failed + preflight_failed,
                skipped=skipped,
                elapsed=time.time() - start_time,
                workload_kind=workload_kind,
            )

    # Wait for workers
    for p in workers:
        p.join(timeout=10)

    # Check for unfinished work
    remaining = len(repros) - done - failed
    if remaining > 0:
        print(f"\n[WARN] {remaining} {task_unit}s were not completed (workers died)")
        if not args.full_graphs:
            print(f"  Re-run with same --tag to fill gaps.")

    elapsed_total = time.time() - start_time
    total_failed = failed + preflight_failed
    print(f"\nDone: {done} ok, {total_failed} failed, {skipped} skipped in {elapsed_total:.1f}s "
          f"({elapsed_total/max(done+failed,1):.1f}s/{task_unit} effective)")

    # Save perf.json per repro
    if args.update_perf and all_results:
        hardware = args.hardware or _detect_hw()
        tag = args.tag or "latest"
        for repro_path, results in all_results.items():
            perf_path = Path(repro_path).parent / "perf.json"
            if perf_path.exists():
                perf = json.loads(perf_path.read_text())
            else:
                perf = {}
            if hardware not in perf:
                perf[hardware] = {}
            if tag not in perf[hardware]:
                perf[hardware][tag] = {}
            for shape_label, r in _metric_result_items(results):
                entry = {k: v for k, v in r.items()}
                entry["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                perf[hardware][tag][shape_label] = entry
            perf_path.write_text(json.dumps(perf, indent=2))
        print(f"[perf] Updated perf.json for {len(all_results)} repros (hardware={hardware}, tag={tag})")

    # Summary report
    if all_results:
        gaps = []
        for repro_path, results in all_results.items():
            for label, r in _metric_result_items(results):
                gap = r.get("gap_cd") or r.get("gap_default")
                if gap is not None:
                    gaps.append((gap, Path(repro_path).parent.name, label,
                                 r.get("total_bytes", r.get("naive_io_bytes", 0))))
        if gaps:
            gaps.sort(reverse=True)
            print(f"\n{'='*70}")
            print(f"Top gaps (worst SOL ratio):")
            for gap, name, label, nbytes in gaps[:15]:
                size_str = f"{nbytes/1024:.0f}KB" if nbytes < 1e6 else f"{nbytes/1e6:.1f}MB"
                print(f"  {gap:5.2f}x  {size_str:>8s}  {name}  [{label}]")
            avg_gap = sum(g for g, *_ in gaps) / len(gaps)
            at_sol = sum(1 for g, *_ in gaps if g <= 1.1)
            print(f"\n  Median gap: {sorted(g for g,*_ in gaps)[len(gaps)//2]:.2f}x")
            print(f"  Mean gap:   {avg_gap:.2f}x")
            print(f"  At SOL (<=1.1x): {at_sol}/{len(gaps)} ({at_sol*100//len(gaps)}%)")

    # Optional JSON output — include metadata for staleness detection
    if args.output and args.oracles:
        timed = _write_oracle_timings_output(args.output, all_results, failures)
        n_priced = sum(1 for k in timed if not k.startswith("_"))
        n_failed = len(timed.get("__failures__", {}))
        print(f"[output] Wrote {args.output} "
              f"({n_priced} oracle dirs with valid oracle_us, "
              f"{n_failed} unpriced)")
    elif args.output:
        _write_results_output(
            args.output,
            all_results,
            failures,
            total=total_requested,
            done=done,
            failed=total_failed,
            skipped=skipped,
            elapsed=elapsed_total,
            workload_kind=workload_kind,
        )
        print(f"[output] Wrote {args.output}")

    # Merge into existing baseline
    if args.merge_into and args.oracles:
        print("[merge-into] not supported in --oracles mode "
              "(flat timings schema differs from the path-keyed baseline); "
              "skipped.")
    elif args.merge_into:
        _merge_into_baseline(
            args.merge_into,
            all_results,
            failures,
            workload_kind=workload_kind,
        )


def _locked_worker(gpu: dict, task_queue, result_queue, args_dict):
    """Run persistent worker subprocess on a GPU, respawn on CUDA error.

    No process-level batch GPU lock. Workers serialize timing with
    INDUCTOR_GPU_BENCH_LOCK, while CPU-only module loading can overlap. With
    --strict-gpu-lock, workers also serialize CUDA setup/warmup/capture against
    timing windows via shared/exclusive lock upgrades.
    """
    import collections
    import subprocess
    import threading

    if True:  # was: with gpu_lock(...) — now using inductor-level lock instead
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu["index"]
        # Enable the per-GPU exclusive lock used by inductor benchmark calls
        # and the direct harness CUDA setup/timing sections below.
        env["INDUCTOR_GPU_BENCH_LOCK"] = "1"
        # Share inductor cache across workers to avoid redundant compilation
        if args_dict.get("share_cache", True):
            env["TORCHINDUCTOR_CACHE_DIR"] = _SHARED_CACHE_DIR

        proc = None
        stderr_tail = collections.deque(maxlen=20)

        def _stderr_summary():
            lines = [line for line in stderr_tail if line]
            if not lines:
                return ""
            return "stderr: " + " | ".join(lines[-3:])[:300]

        def _spawn_worker():
            nonlocal stderr_tail
            stderr_tail = collections.deque(maxlen=20)
            script_factory = args_dict.get("_persistent_worker_script_factory")
            if script_factory is None:
                script = _persistent_worker_script(gpu["index"], args_dict)
            else:
                script = script_factory(gpu["index"], args_dict)
            p = subprocess.Popen(
                [sys.executable, "-u", "-c", script],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1,
                env=env, cwd=args_dict["root"],
            )

            def _drain_stderr():
                if p.stderr is None:
                    return
                for stderr_line in p.stderr:
                    stderr_line = stderr_line.rstrip()
                    if stderr_line:
                        stderr_tail.append(stderr_line)

            threading.Thread(
                target=_drain_stderr,
                name=f"bench-worker-stderr-gpu-{gpu['index']}",
                daemon=True,
            ).start()
            return p

        def _kill_worker(p):
            if p is None:
                return
            if p.poll() is None:
                try:
                    if p.stdin is not None:
                        p.stdin.write("EXIT\n")
                        p.stdin.flush()
                except Exception:
                    pass
                try:
                    p.wait(timeout=2)
                except Exception:
                    p.terminate()
                    try:
                        p.wait(timeout=5)
                    except Exception:
                        p.kill()
                        try:
                            p.wait(timeout=5)
                        except Exception:
                            pass
            for stream in (p.stdin, p.stdout, p.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass

        # Prefetch: keep at most one next repro local so we can send PREFETCH
        # hints while the worker is timing the current repro without starving
        # peer workers at the start of small runs.
        # The worker will pre-import the next module (CPU-bound) while the
        # current do_bench is running (GPU-bound), overlapping the two phases.
        pending_repros = []

        def _grab_next_batch(max_prefetch=2, reserve_for_peers=0):
            """Pull up to max_prefetch items from the queue into pending_repros."""
            while len(pending_repros) < max_prefetch:
                if reserve_for_peers and task_queue.qsize() <= reserve_for_peers:
                    break
                try:
                    pending_repros.append(task_queue.get_nowait())
                except Exception:
                    break

        while True:
            if not pending_repros:
                _grab_next_batch(max_prefetch=1)
                if not pending_repros:
                    break

            repro_path = pending_repros.pop(0)
            _grab_next_batch(
                max_prefetch=2,
                reserve_for_peers=max(0, int(args_dict.get("n_workers", 1)) - 1),
            )

            # Ensure we have a live worker
            if proc is None or proc.poll() is not None:
                _kill_worker(proc)
                proc = _spawn_worker()

            start = time.time()
            try:
                # Send prefetch hint for NEXT repro (worker will pre-import it
                # in a background thread while timing the current one)
                if pending_repros:
                    proc.stdin.write(f"PREFETCH:{pending_repros[0]}\n")
                    proc.stdin.flush()

                # Send actual benchmark request
                proc.stdin.write(str(repro_path) + "\n")
                proc.stdin.flush()

                line = proc.stdout.readline()
                elapsed = time.time() - start

                if not line:
                    # Worker died
                    returncode = proc.poll()
                    error = f"worker exited without a result (returncode={returncode})"
                    stderr = _stderr_summary()
                    if stderr:
                        error = f"{error}; {stderr}"
                    _kill_worker(proc)
                    proc = None
                    result_queue.put({
                        "repro": str(repro_path),
                        "gpu": gpu["index"],
                        "status": "failed",
                        "elapsed": elapsed,
                        "error": error[:1000],
                    })
                    continue

                line = line.strip()
                if line.startswith("{"):
                    results = json.loads(line)
                    # Validate result-repro alignment: the worker embeds the
                    # repro path it actually benchmarked in "_repro". If this
                    # doesn't match what we sent, the stdout stream is misaligned
                    # (e.g., due to stray prints from module loading polluting
                    # the result pipe before our stdout-redirect fix took effect).
                    result_repro = results.pop("_repro", None)
                    expected = str(repro_path)
                    if result_repro is not None and result_repro != expected:
                        # MISALIGNMENT DETECTED: kill worker to reset the pipe.
                        # Report as failure — we cannot trust this result.
                        error = (
                            f"result misalignment: expected repro "
                            f"'{_task_display_name(expected)}' but worker returned "
                            f"result for '{_task_display_name(result_repro)}'. "
                            f"Killed worker to reset stdout pipe."
                        )
                        _kill_worker(proc)
                        proc = None
                        result_queue.put({
                            "repro": str(repro_path),
                            "gpu": gpu["index"],
                            "status": "failed",
                            "elapsed": elapsed,
                            "error": error[:1000],
                        })
                        continue
                    error_payload = results.pop("__error__", None)
                    if error_payload is not None:
                        result_queue.put({
                            "repro": str(repro_path),
                            "gpu": gpu["index"],
                            "status": "failed",
                            "elapsed": elapsed,
                            "error": str(error_payload.get("error", ""))[:1000],
                            "category": error_payload.get("category", "runtime_error"),
                            "reason": error_payload.get("reason"),
                            "hint": error_payload.get("hint"),
                            "exception_type": error_payload.get("exception_type"),
                        })
                        continue
                    result_queue.put({
                        "repro": str(repro_path),
                        "gpu": gpu["index"],
                        "status": "ok",
                        "elapsed": elapsed,
                        "results": results,
                    })
                elif "CUDA_ERROR" in line:
                    # Worker will exit, respawn on next iteration
                    error = line[:200]
                    stderr = _stderr_summary()
                    if stderr:
                        error = f"{error}; {stderr}"
                    _kill_worker(proc)
                    proc = None
                    result_queue.put({
                        "repro": str(repro_path),
                        "gpu": gpu["index"],
                        "status": "failed",
                        "elapsed": elapsed,
                        "error": error[:1000],
                    })
                else:
                    error = line[:200]
                    stderr = _stderr_summary()
                    if stderr:
                        error = f"{error}; {stderr}"
                    result_queue.put({
                        "repro": str(repro_path),
                        "gpu": gpu["index"],
                        "status": "failed",
                        "elapsed": elapsed,
                        "error": error[:1000],
                    })
            except Exception as e:
                _kill_worker(proc)
                proc = None
                result_queue.put({
                    "repro": str(repro_path),
                    "gpu": gpu["index"],
                    "status": "failed",
                    "elapsed": time.time() - start,
                    "error": str(e)[:1000],
                    "category": "worker_parent_error",
                    "exception_type": type(e).__name__,
                })

        _kill_worker(proc)


def _oracle_worker_script(gpu_idx: str, args_dict: dict) -> str:
    """Persistent worker that benchmarks v2 oracles, one canonical dir per line.

    Mirrors the `python -m oracle_harness <dir> --bench` CLI loop in
    oracle_harness._runner_main (check first, then bench_oracle per shape
    point, gated on the per-point numerics check). Reuses bench_oracle, which
    already honors INDUCTOR_GPU_BENCH_LOCK via _gpu_exclusive_lock — the env
    var is set by _locked_worker, so the per-GPU exclusive timing lock is
    respected with no extra wiring here.

    The result JSON per dir is:
        {
          "<label>": {"oracle_us": float, "compile_us": float,
                      "ratio": float, "status": str},
          ...,
          "_repro": "<dir path>",   # parent-side alignment check
        }
    Invalid/failing points carry a "status" in oracle_harness._INVALID_STATUSES
    and no oracle_us, so the aggregator excludes them.

    Uses the same fd-isolation as _persistent_worker_script: bench_oracle and
    check_oracle_all_shapes print progress/JSON to stdout; we redirect fd 1 to
    stderr so only our result JSON reaches the parent's result pipe.
    """
    skip_check = bool(args_dict.get("oracle_skip_check", False))
    return f'''
import sys, json, os, io
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_idx}"

import torch  # noqa: F401  (init CUDA in this subprocess)
from pathlib import Path
from oracle_harness import (
    _load_oracle_module,
    check_oracle_all_shapes,
    bench_oracle,
    get_inputs,
    get_repro_instance,
    _INVALID_STATUSES,
)
from repro_harness import load_shape_configs, make_inputs_from_config

WARMUP = {args_dict["n_warmup"]}
REP = {args_dict["n_rep"]}
SKIP_CHECK = {skip_check}
SHAPE_SEP = {_SHAPE_TASK_SEP!r}
DEFAULT_SHAPE_TOKEN = {_DEFAULT_SHAPE_TOKEN!r}

# Isolate the result pipe from any stdout pollution (bench_oracle /
# check_oracle_all_shapes print JSON + progress to fd 1). Keep a private dup of
# the original stdout for results; point fd 1 (and python stdout) at stderr.
_result_fd = os.dup(1)
_result_file = os.fdopen(_result_fd, "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr

def bench_oracle_point(canonical_dir, shape_label):
    """Bench ONE (dir, shape) point — the sharded task unit.

    Runs the numerics gate for just this shape (check_oracle_all_shapes
    point=...) then bench_oracle for it. bench_oracle does its own per-shape
    torch._dynamo.reset() + fresh static compile (commit 4ca6d532b), so a
    single-shape task is identical in measurement to that shape's iteration of
    the per-dir loop — sharding never reuses dynamo state across shapes.

    Returns {{label: <point result>}} so the parent can regroup all of a dir's
    shape points into one per-dir payload identical to the un-sharded path.
    """
    mod, fn, d = _load_oracle_module(canonical_dir)
    repro_id = d.name
    repro_file = d / "repro.py"
    configs = load_shape_configs(str(repro_file))

    if shape_label == DEFAULT_SHAPE_TOKEN or not configs:
        # Dir with no shape configs: single default point.
        check = {{}}
        if not SKIP_CHECK:
            check = check_oracle_all_shapes(fn, d, repro_id, skip_stochastic=True)
        passed = True
        if not SKIP_CHECK:
            passed = all(v not in ("fail", False) for v in check.values())
        if passed:
            inputs = get_inputs(d)
            instance = get_repro_instance(d)
            return {{"default": bench_oracle(
                fn, instance, inputs, repro_id, warmup=WARMUP, rep=REP)}}
        return {{"default": {{"repro_id": repro_id,
                             "status": "UNVERIFIED_NUMERICS"}}}}

    if shape_label not in configs:
        return {{shape_label: {{"repro_id": repro_id + "_" + shape_label,
                               "status": "NO_ORACLE_FOR_SHAPE"}}}}

    config = configs[shape_label]
    point_id = repro_id + "_" + shape_label
    passed = True
    if not SKIP_CHECK:
        # Narrow the numerics gate to just this shape point.
        check = check_oracle_all_shapes(
            fn, d, repro_id, skip_stochastic=True, point=shape_label)
        passed = all(v not in ("fail", False) for v in check.values())
    if passed:
        inputs = make_inputs_from_config(config)
        instance = get_repro_instance(d)
        return {{shape_label: bench_oracle(
            fn, instance, inputs, point_id, warmup=WARMUP, rep=REP)}}
    return {{shape_label: {{"repro_id": point_id,
                           "status": "UNVERIFIED_NUMERICS"}}}}

def bench_oracle_dir(canonical_dir):
    mod, fn, d = _load_oracle_module(canonical_dir)
    repro_id = d.name
    repro_file = d / "repro.py"
    configs = load_shape_configs(str(repro_file))

    # Numerics check gate (mirror CLI: --bench implies --check).
    check = {{}}
    if not SKIP_CHECK:
        check = check_oracle_all_shapes(fn, d, repro_id, skip_stochastic=True)

    results = {{}}
    if not configs:
        passed = True
        if not SKIP_CHECK:
            passed = all(v not in ("fail", False) for v in check.values())
        if passed:
            inputs = get_inputs(d)
            instance = get_repro_instance(d)
            results["default"] = bench_oracle(
                fn, instance, inputs, repro_id, warmup=WARMUP, rep=REP)
        else:
            results["default"] = {{"repro_id": repro_id,
                                   "status": "UNVERIFIED_NUMERICS"}}
        return results

    for label, config in configs.items():
        point_id = f"{{repro_id}}_{{label}}"
        passed = True
        if not SKIP_CHECK:
            passed = check.get(label) not in ("fail", False)
        if passed:
            inputs = make_inputs_from_config(config)
            instance = get_repro_instance(d)
            results[label] = bench_oracle(
                fn, instance, inputs, point_id, warmup=WARMUP, rep=REP)
        else:
            results[label] = {{"repro_id": point_id,
                               "status": "UNVERIFIED_NUMERICS"}}
    return results

for line in sys.stdin:
    line = line.strip()
    if not line or line == "EXIT":
        break
    if line.startswith("PREFETCH:"):
        # No-op for oracles: per-dir module load is cheap relative to bench.
        continue
    try:
        # Sharded task key: "<dir>::SHAPE::<label>" -> bench ONE shape point.
        # A bare dir path (no separator) falls back to the whole-dir loop.
        if SHAPE_SEP in line:
            canonical_dir, shape_label = line.rsplit(SHAPE_SEP, 1)
            results = bench_oracle_point(canonical_dir, shape_label)
        else:
            results = bench_oracle_dir(line)
        results["_repro"] = line
        _result_file.write(json.dumps(results, default=str) + "\\n")
        _result_file.flush()
    except Exception as e:
        if "CUDA" in str(e) or "device-side assert" in str(e):
            _result_file.write(f"CUDA_ERROR: {{str(e)[:1000]}}\\n")
            _result_file.flush()
            sys.exit(1)
        _result_file.write(json.dumps({{
            "_repro": line,
            "__error__": {{
                "status": "failed",
                "category": "runtime_error",
                "exception_type": type(e).__name__,
                "error": str(e)[:1000],
                "reason": "oracle worker failed while benchmarking dir/point",
                "hint": "Run `python -m oracle_harness <dir> --bench` directly.",
            }},
        }}) + "\\n")
        _result_file.flush()
'''


def _persistent_worker_script(gpu_idx: str, args_dict: dict) -> str:
    """Script for a persistent worker that reads repro paths from stdin.

    Supports PREFETCH:<path> commands to pre-import the next repro module
    in a background thread, overlapping CPU-bound module loading with
    GPU-bound timing of the current repro.

    PROTOCOL FIX: each JSON result now includes a "_repro" key identifying
    which repro produced it. The parent validates this matches expectations
    to detect and recover from stdout pipeline misalignment (e.g., if a
    prefetch thread or module import prints to stdout, shifting the result
    stream).
    """
    return f'''
import builtins, contextlib, fcntl, io, re, sys, json, os, tempfile, threading, time
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_idx}"

import torch, torch._dynamo
import torch._inductor.config as inductor_config
import torch._inductor.metrics as inductor_metrics
from triton.testing import do_bench
import importlib.util, math
from repro_harness import load_shape_configs, make_inputs_from_config, make_inputs_safely
from byte_accounting import count_bytes_effective
from full_graph_harness import load_full_graph_definition, load_full_graph, result_metadata, tensor_bytes

STRICT_GPU_LOCK = {args_dict["strict_gpu_lock"]}
WORKLOAD_KIND = {args_dict.get("workload_kind", "repro")!r}

# Extra inductor config knobs
if {args_dict.get("combo_kernels", False)}:
    inductor_config.combo_kernels = True
    inductor_config.combo_kernel_per_subkernel_blocks = True
if {args_dict.get("multi_kernel", 0)}:
    inductor_config.triton.multi_kernel = {args_dict.get("multi_kernel", 0)}

# --- Prefetch infrastructure ---
# Pre-imports the next repro module while the current one is being timed.
# Module loading (file I/O, AST parse, exec) is CPU-bound and can overlap
# with the GPU-bound do_bench calls of the previous repro.
_prefetch_cache = {{}}  # path -> repro tuple or FullGraphDefinition
_prefetch_lock = threading.Lock()
_GPU_BENCH_LOCK_STATE_NAME = "_torchinductor_gpu_benchmark_lock_state"
_GPU_BENCH_LOCK_MODES = {{"shared", "exclusive"}}

def _env_flag_enabled(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")

def _safe_lock_component(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()) or "unknown"

def _gpu_bench_lock_state():
    state = getattr(builtins, _GPU_BENCH_LOCK_STATE_NAME, None)
    if state is None:
        state = {{
            "mutex": threading.RLock(),
            "local": threading.local(),
        }}
        setattr(builtins, _GPU_BENCH_LOCK_STATE_NAME, state)
    return state

def _gpu_bench_flock_op(mode):
    if mode == "shared":
        return fcntl.LOCK_SH
    if mode == "exclusive":
        return fcntl.LOCK_EX
    raise ValueError(f"Unsupported GPU benchmark lock mode: {{mode!r}}")

def _write_gpu_lock_metadata(fd, gpu, mode, label):
    try:
        os.ftruncate(fd, 0)
        os.write(fd, (
            f"pid={{os.getpid()}}\\n"
            f"gpu={{gpu}}\\n"
            f"mode={{mode}}\\n"
            f"label={{label}}\\n"
            f"acquired_unix={{time.time():.0f}}\\n"
        ).encode())
        os.fsync(fd)
    except OSError:
        pass

def _release_fd(fd):
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)

@contextlib.contextmanager
def gpu_bench_lock(mode="exclusive"):
    if mode not in _GPU_BENCH_LOCK_MODES:
        raise ValueError(f"Unsupported GPU benchmark lock mode: {{mode!r}}")
    if not (
        _env_flag_enabled("INDUCTOR_GPU_BENCH_LOCK")
        or _env_flag_enabled("TORCHINDUCTOR_GPU_BENCH_LOCK")
    ):
        yield
        return
    lock_dir = (
        os.environ.get("INDUCTOR_GPU_BENCH_LOCK_DIR")
        or os.environ.get("TORCHINDUCTOR_GPU_BENCH_LOCK_DIR")
        or os.environ.get("COMPILE_UTILS_GPU_LOCK_DIR")
        or os.path.join(tempfile.gettempdir(), "compile_utils_gpu_locks")
    )
    os.makedirs(lock_dir, exist_ok=True)
    visible = [d.strip() for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    gpu = _safe_lock_component(visible[0] if visible else "0")
    lock_path = os.path.join(lock_dir, f"gpu_{{gpu}}.lock")
    gate_path = os.path.join(lock_dir, f"gpu_{{gpu}}.gate")
    state = _gpu_bench_lock_state()
    mutex = state["mutex"]
    local = state["local"]
    depth = getattr(local, "depth", 0)
    if depth > 0:
        current_mode = getattr(local, "mode", None)
        fd = getattr(local, "fd", None)
        if current_mode == "exclusive" or current_mode == mode:
            local.depth = depth + 1
            try:
                yield
            finally:
                local.depth -= 1
            return
        if current_mode == "shared" and mode == "exclusive" and fd is not None:
            gate_fd = os.open(gate_path, os.O_CREAT | os.O_RDWR, 0o666)
            fcntl.flock(fd, fcntl.LOCK_UN)
            fcntl.flock(gate_fd, fcntl.LOCK_EX)
            fcntl.flock(fd, _gpu_bench_flock_op("exclusive"))
            _write_gpu_lock_metadata(fd, gpu, "exclusive", "bench_parallel_upgrade")
            local.mode = "exclusive"
            local.gate_fd = gate_fd
            local.depth = depth + 1
            try:
                yield
            finally:
                local.depth -= 1
                fcntl.flock(fd, _gpu_bench_flock_op("shared"))
                _write_gpu_lock_metadata(fd, gpu, "shared", "bench_parallel_setup")
                local.mode = "shared"
                local.gate_fd = None
                _release_fd(gate_fd)
            return
        local.depth = depth + 1
        try:
            yield
        finally:
            local.depth -= 1
        return
    with mutex:
        fd = None
        gate_fd = None
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
            gate_fd = os.open(gate_path, os.O_CREAT | os.O_RDWR, 0o666)
            if mode == "shared":
                # Writer-preferring turnstile: an exclusive waiter holds the
                # gate exclusively while waiting for existing shared holders to
                # drain, which prevents new setup work from entering shared.
                fcntl.flock(gate_fd, fcntl.LOCK_SH)
                fcntl.flock(fd, fcntl.LOCK_SH)
                _release_fd(gate_fd)
                gate_fd = None
            else:
                fcntl.flock(gate_fd, fcntl.LOCK_EX)
                fcntl.flock(fd, fcntl.LOCK_EX)
            label = "bench_parallel_setup" if mode == "shared" else "bench_parallel_do_bench"
            _write_gpu_lock_metadata(fd, gpu, mode, label)
            local.depth = 1
            local.mode = mode
            local.fd = fd
            local.gate_fd = gate_fd
            try:
                yield
            finally:
                local.depth = 0
                local.mode = None
                local.fd = None
                local.gate_fd = None
        finally:
            _release_fd(fd)
            _release_fd(gate_fd)

def _install_inductor_gpu_lock_hook():
    """Register this worker's lock with Inductor internal benchmark calls."""
    if not STRICT_GPU_LOCK:
        return
    try:
        import torch._inductor.runtime.benchmarking as inductor_benchmarking
    except Exception as exc:
        raise RuntimeError("--strict-gpu-lock requires Inductor benchmarking hooks") from exc

    set_context = getattr(inductor_benchmarking, "set_gpu_benchmark_lock_context", None)
    if set_context is None:
        raise RuntimeError(
            "--strict-gpu-lock requires torch._inductor.runtime.benchmarking."
            "set_gpu_benchmark_lock_context"
        )

    @contextlib.contextmanager
    def context_factory():
        with gpu_bench_lock("exclusive"):
            yield

    set_context(context_factory)

_install_inductor_gpu_lock_hook()

@contextlib.contextmanager
def gpu_setup_lock():
    if STRICT_GPU_LOCK:
        with gpu_bench_lock("shared"):
            yield
    else:
        yield

def _prefetch_module(repro_path):
    """Load a repro module into the cache (called from background thread).

    IMPORTANT: redirects stdout to /dev/null during module loading to prevent
    any print statements in the loaded module from polluting the JSON result
    stream on stdout (which would cause result misalignment in the parent).
    """
    try:
        # Redirect stdout in this thread to prevent pollution of the result pipe.
        # stderr is fine (parent drains it separately).
        _real_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            if WORKLOAD_KIND == "full_graph":
                definition = load_full_graph_definition(repro_path)
                with _prefetch_lock:
                    _prefetch_cache[repro_path] = definition
                return
            spec = importlib.util.spec_from_file_location("repro_prefetch", repro_path)
            mod = importlib.util.module_from_spec(spec)
            mod.device = torch.device
            mod.inf = math.inf
            mod.nan = math.nan
            spec.loader.exec_module(mod)
            instance = mod.Repro()
            configs = load_shape_configs(repro_path)
            with _prefetch_lock:
                _prefetch_cache[repro_path] = (mod, instance, configs)
        finally:
            sys.stdout = _real_stdout
    except Exception:
        pass  # prefetch failure is non-fatal; bench_one will load normally

def _get_or_load_module(repro_path):
    """Get module from prefetch cache, or load synchronously."""
    with _prefetch_lock:
        cached = _prefetch_cache.pop(repro_path, None)
    if cached is not None:
        return cached
    # Synchronous fallback
    spec = importlib.util.spec_from_file_location("repro", repro_path)
    mod = importlib.util.module_from_spec(spec)
    mod.device = torch.device
    mod.inf = math.inf
    mod.nan = math.nan
    spec.loader.exec_module(mod)
    instance = mod.Repro()
    configs = load_shape_configs(repro_path)
    return (mod, instance, configs)

def _get_or_load_full_graph(repro_path):
    with _prefetch_lock:
        cached = _prefetch_cache.pop(repro_path, None)
    definition = cached if cached is not None else load_full_graph_definition(repro_path)
    instance, inputs, definition = load_full_graph(definition, default_device="cuda")
    return instance, inputs, definition

N_ROUNDS = 5  # interleaved timing rounds for min-of-mins

def _capture_cudagraph(fn, inps):
    """Compile warmup and capture a CUDAGraph. Returns (graph, is_graph).

    If CUDAGraph capture fails (some ops unsupported), returns (fn, False)
    as a fallback callable.
    """
    with torch.no_grad():
        for _ in range(3):
            fn(*inps)
        torch.cuda.synchronize()
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn(*inps)
            torch.cuda.synchronize()
            return g, True
        except Exception:
            return fn, False

def _make_bench_callable(graph_or_fn, is_graph, inps):
    """Return a zero-arg callable suitable for do_bench."""
    if is_graph:
        return lambda: graph_or_fn.replay()
    else:
        return lambda: graph_or_fn(*inps)

def bench_full_graph_one(repro_path):
    instance, inputs, _definition = _get_or_load_full_graph(repro_path)

    with gpu_setup_lock():
        with torch.no_grad():
            eager_out = instance(*inputs)
        torch.cuda.synchronize()
        input_bytes = tensor_bytes(inputs)
        output_bytes = tensor_bytes(eager_out)
        del eager_out

    # Compile default
    inductor_metrics.reset()
    torch._dynamo.reset()
    compiled = torch.compile(instance)
    with gpu_setup_lock():
        with torch.no_grad():
            graph_default, default_is_graph = _capture_cudagraph(compiled, inputs)
    n_kernels = inductor_metrics.generated_kernel_count

    # Compile coordinate descent
    do_cd = not {args_dict["no_cd"]}
    graph_cd = None
    cd_is_graph = False
    if do_cd:
        inductor_config.coordinate_descent_tuning = True
        try:
            torch._dynamo.reset()
            compiled_cd = torch.compile(instance)
            with gpu_setup_lock():
                with torch.no_grad():
                    graph_cd, cd_is_graph = _capture_cudagraph(compiled_cd, inputs)
        finally:
            inductor_config.coordinate_descent_tuning = False

    bench_default = _make_bench_callable(graph_default, default_is_graph, inputs)
    bench_cd = _make_bench_callable(graph_cd, cd_is_graph, inputs) if do_cd else None

    with gpu_bench_lock():
        for _ in range(25):
            bench_default()
            if bench_cd:
                bench_cd()
        torch.cuda.synchronize()

        default_times = []
        cd_times = []
        for _ in range(N_ROUNDS):
            t = do_bench(
                bench_default,
                warmup={args_dict["n_warmup"]},
                rep={args_dict["n_rep"]},
                return_mode="min",
            ) * 1000
            default_times.append(t)
            if bench_cd:
                t = do_bench(
                    bench_cd,
                    warmup={args_dict["n_warmup"]},
                    rep={args_dict["n_rep"]},
                    return_mode="min",
                ) * 1000
                cd_times.append(t)

    compiled_us = min(default_times)
    cd_us = min(cd_times) if cd_times else None

    return {{
        "__graph__": result_metadata(_definition),
        "default": {{
            "compiled_us": compiled_us,
            "coord_descent_us": cd_us,
            "memcopy_sol_us": None,
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "naive_io_bytes": input_bytes + output_bytes,
            "n_kernels": n_kernels,
            "num_inputs": len(inputs),
            "gap_default": None,
            "gap_cd": None,
        }}
    }}

def bench_one(repro_path):
    if WORKLOAD_KIND == "full_graph":
        return bench_full_graph_one(repro_path)

    mod, instance, configs = _get_or_load_module(repro_path)

    all_shapes = {args_dict["all_shapes"]}
    if all_shapes and configs:
        shape_items = list(configs.items())
    else:
        shape_items = [(None, None)]

    all_results = {{}}
    for shape_name, shape_config in shape_items:
        with gpu_setup_lock():
            if shape_config is not None:
                inputs = make_inputs_from_config(shape_config)
                label = shape_name
            else:
                make_inputs_fn = mod.make_inputs if hasattr(mod, "make_inputs") else mod._default_make_inputs
                inputs = make_inputs_safely(make_inputs_fn)
                label = "default"

            with torch.no_grad():
                eager_out = instance(*inputs)
            torch.cuda.synchronize()
            total_bytes = count_bytes_effective(instance, inputs)
            torch.cuda.synchronize()

        # --- Phase 1: Compile all configs and capture CUDAGraphs (no exclusive lock) ---
        copy_elems = max(total_bytes // 8, 256)

        # Compile default
        inductor_metrics.reset()
        torch._dynamo.reset()
        compiled = torch.compile(instance)
        with gpu_setup_lock():
            with torch.no_grad():
                graph_default, default_is_graph = _capture_cudagraph(compiled, inputs)
        n_kernels = inductor_metrics.generated_kernel_count

        # Compile coordinate descent
        do_cd = not {args_dict["no_cd"]}
        graph_cd = None
        cd_is_graph = False
        if do_cd:
            inductor_config.coordinate_descent_tuning = True
            torch._dynamo.reset()
            compiled_cd = torch.compile(instance)
            with gpu_setup_lock():
                with torch.no_grad():
                    graph_cd, cd_is_graph = _capture_cudagraph(compiled_cd, inputs)
            inductor_config.coordinate_descent_tuning = False

        # Build callables for timing
        bench_default = _make_bench_callable(graph_default, default_is_graph, inputs)
        bench_cd = _make_bench_callable(graph_cd, cd_is_graph, inputs) if do_cd else None

        # --- Phase 2: Time ALL configs under a single exclusive lock hold ---
        with gpu_bench_lock():
            # Memcopy speed-of-light measurement
            src = torch.empty(copy_elems, dtype=torch.float32, device="cuda")
            dst = torch.empty_like(src)
            sol_us = do_bench(
                lambda: torch.add(src, 1, out=dst),
                warmup={args_dict["n_warmup"]},
                rep={args_dict["n_rep"]},
                return_mode="min",
            ) * 1000
            torch.cuda.synchronize()
            del src, dst

            # Warm up all graphs
            for _ in range(25):
                bench_default()
                if bench_cd:
                    bench_cd()
            torch.cuda.synchronize()

            # Interleaved timing rounds
            default_times = []
            cd_times = []
            for _ in range(N_ROUNDS):
                t = do_bench(bench_default, warmup={args_dict["n_warmup"]}, rep={args_dict["n_rep"]}, return_mode="min") * 1000
                default_times.append(t)
                if bench_cd:
                    t = do_bench(bench_cd, warmup={args_dict["n_warmup"]}, rep={args_dict["n_rep"]}, return_mode="min") * 1000
                    cd_times.append(t)

            compiled_us = min(default_times)
            cd_us = min(cd_times) if cd_times else None

        all_results[label] = {{
            "compiled_us": compiled_us,
            "coord_descent_us": cd_us,
            "memcopy_sol_us": sol_us,
            "total_bytes": total_bytes,
            "n_kernels": n_kernels,
            "gap_default": compiled_us / sol_us if sol_us > 0 else None,
            "gap_cd": cd_us / sol_us if (cd_us and sol_us > 0) else None,
        }}

    return all_results

# Main loop: read repro paths from stdin, write JSON results to stdout.
# Supports PREFETCH:<path> to pre-import the next module in background.
#
# PROTOCOL: each JSON result includes "_repro" key so the parent can verify
# the result belongs to the expected repro path. This prevents misattribution
# when stdout gets polluted by stray prints from module loading or torch internals.
#
# Dedicated result fd: results are the ONLY thing written to the parent's
# result pipe. We preserve the original stdout pipe as a private fd, then point
# fd 1 itself at stderr (os.dup2). This isolates the result pipe from ALL stdout
# pollution -- including native C/C++ writes to fd 1 from torch/triton/CUDA,
# which a Python-only ``sys.stdout`` redirect cannot catch. Any such writes now
# land on stderr (drained separately by the parent), never the result stream.
_result_fd = os.dup(1)  # private dup of the original stdout pipe (for results)
_result_file = os.fdopen(_result_fd, "w", buffering=1)  # line-buffered
os.dup2(2, 1)  # fd 1 -> stderr: native + stray fd-1 writes go to stderr, not results
sys.stdout = sys.stderr  # python-level prints -> stderr too

for line in sys.stdin:
    line = line.strip()
    if not line or line == "EXIT":
        break
    if line.startswith("PREFETCH:"):
        # Start background thread to pre-import the next repro module.
        # This overlaps CPU-bound module loading with the GPU-bound timing
        # of the current repro (which follows immediately after).
        prefetch_path = line[len("PREFETCH:"):]
        t = threading.Thread(target=_prefetch_module, args=(prefetch_path,), daemon=True)
        t.start()
        continue
    try:
        results = bench_one(line)
        # Include the repro path in the result for parent-side validation
        results["_repro"] = line
        _result_file.write(json.dumps(results) + "\\n")
        _result_file.flush()
    except Exception as e:
        if "CUDA" in str(e) or "device-side assert" in str(e):
            _result_file.write(f"CUDA_ERROR: {{str(e)[:1000]}}\\n")
            _result_file.flush()
            sys.exit(1)  # die so parent respawns
        _result_file.write(json.dumps({{
            "_repro": line,
            "__error__": {{
                "status": "failed",
                "category": "runtime_error",
                "exception_type": type(e).__name__,
                "error": str(e)[:1000],
                "reason": "worker failed while benchmarking workload",
                "hint": "Inspect worker stderr or rerun this workload directly.",
            }},
        }}) + "\\n")
        _result_file.flush()
'''


def _worker_script(repro_path: str, gpu_idx: str, args_dict: dict) -> str:
    """Generate a self-contained benchmark script for one repro."""
    return f'''
import sys, json, os
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_idx}"

import torch, torch._dynamo
import torch._inductor.config as inductor_config
import torch._inductor.metrics as inductor_metrics
from triton.testing import do_bench
import importlib.util, math

# Extra inductor config knobs
if {args_dict.get("combo_kernels", False)}:
    inductor_config.combo_kernels = True
    inductor_config.combo_kernel_per_subkernel_blocks = True
if {args_dict.get("multi_kernel", 0)}:
    inductor_config.triton.multi_kernel = {args_dict.get("multi_kernel", 0)}

spec = importlib.util.spec_from_file_location("repro", "{repro_path}")
mod = importlib.util.module_from_spec(spec)
mod.device = torch.device
mod.inf = math.inf
mod.nan = math.nan
spec.loader.exec_module(mod)

from repro_harness import load_shape_configs, make_inputs_from_config, make_inputs_safely
from byte_accounting import count_bytes_effective

instance = mod.Repro()

# Determine which shapes to run
all_shapes = {args_dict["all_shapes"]}
configs = load_shape_configs("{repro_path}")
if all_shapes and configs:
    shape_items = list(configs.items())
else:
    shape_items = [(None, None)]

N_ROUNDS = 5  # interleaved timing rounds for min-of-mins

def _capture_cudagraph(fn, inps):
    with torch.no_grad():
        for _ in range(3):
            fn(*inps)
        torch.cuda.synchronize()
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn(*inps)
            torch.cuda.synchronize()
            return g, True
        except Exception:
            return fn, False

def _make_bench_callable(graph_or_fn, is_graph, inps):
    if is_graph:
        return lambda: graph_or_fn.replay()
    else:
        return lambda: graph_or_fn(*inps)

all_results = {{}}
for shape_name, shape_config in shape_items:
    if shape_config is not None:
        inputs = make_inputs_from_config(shape_config)
        label = shape_name
    else:
        make_inputs_fn = mod.make_inputs if hasattr(mod, "make_inputs") else mod._default_make_inputs
        inputs = make_inputs_safely(make_inputs_fn)
        label = "default"

    with torch.no_grad():
        eager_out = instance(*inputs)

    total_bytes = count_bytes_effective(instance, inputs)

    # --- Phase 1: Compile all configs and capture CUDAGraphs ---
    copy_elems = max(total_bytes // 8, 256)

    # Compile default
    inductor_metrics.reset()
    torch._dynamo.reset()
    compiled = torch.compile(instance)
    with torch.no_grad():
        graph_default, default_is_graph = _capture_cudagraph(compiled, inputs)
    n_kernels = inductor_metrics.generated_kernel_count

    # Compile coordinate descent
    do_cd = not {args_dict["no_cd"]}
    graph_cd = None
    cd_is_graph = False
    if do_cd:
        inductor_config.coordinate_descent_tuning = True
        torch._dynamo.reset()
        compiled_cd = torch.compile(instance)
        with torch.no_grad():
            graph_cd, cd_is_graph = _capture_cudagraph(compiled_cd, inputs)
        inductor_config.coordinate_descent_tuning = False

    # Build callables for timing
    bench_default = _make_bench_callable(graph_default, default_is_graph, inputs)
    bench_cd = _make_bench_callable(graph_cd, cd_is_graph, inputs) if do_cd else None

    # --- Phase 2: Time ALL configs together ---
    # Memcopy speed-of-light
    src = torch.empty(copy_elems, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    sol_us = do_bench(
        lambda: dst.copy_(src),
        warmup={args_dict["n_warmup"]},
        rep={args_dict["n_rep"]},
        return_mode="min",
    ) * 1000
    del src, dst

    # Warm up all graphs
    for _ in range(25):
        bench_default()
        if bench_cd:
            bench_cd()
    torch.cuda.synchronize()

    # Interleaved timing rounds
    default_times = []
    cd_times = []
    for _ in range(N_ROUNDS):
        t = do_bench(bench_default, warmup=5, rep={args_dict["n_rep"]}, return_mode="min") * 1000
        default_times.append(t)
        if bench_cd:
            t = do_bench(bench_cd, warmup=5, rep={args_dict["n_rep"]}, return_mode="min") * 1000
            cd_times.append(t)

    compiled_us = min(default_times)
    cd_us = min(cd_times) if cd_times else None

    all_results[label] = {{
        "compiled_us": compiled_us,
        "coord_descent_us": cd_us,
        "memcopy_sol_us": sol_us,
        "total_bytes": total_bytes,
        "n_kernels": n_kernels,
        "gap_default": compiled_us / sol_us if sol_us > 0 else None,
        "gap_cd": cd_us / sol_us if (cd_us and sol_us > 0) else None,
    }}

print(json.dumps(all_results))
'''


def _detect_hw() -> str:
    gpus = discover_gpus()
    return gpus[0]["kind"] if gpus else "unknown"


def _run_compare(paths: list[Path], tag_a: str, tag_b: str):
    """Compare two tagged runs across all repros."""
    repros = find_repros(paths)
    if not repros:
        print("No repro.py files found.")
        return

    diffs = []
    for repro in repros:
        perf_path = repro.parent / "perf.json"
        if not perf_path.exists():
            continue
        perf = json.loads(perf_path.read_text())
        name = repro.parent.name

        for hardware, hw_data in perf.items():
            a_data = hw_data.get(tag_a, {})
            b_data = hw_data.get(tag_b, {})

            for shape in set(list(a_data.keys()) + list(b_data.keys())):
                a = a_data.get(shape, {})
                b = b_data.get(shape, {})

                a_gap = a.get("gap_cd") or a.get("gap_default")
                b_gap = b.get("gap_cd") or b.get("gap_default")
                a_us = a.get("coord_descent_us") or a.get("compiled_us")
                b_us = b.get("coord_descent_us") or b.get("compiled_us")

                if a_us and b_us:
                    speedup = a_us / b_us
                    diffs.append({
                        "name": name,
                        "shape": shape,
                        "hardware": hardware,
                        "a_us": a_us,
                        "b_us": b_us,
                        "speedup": speedup,
                        "a_gap": a_gap,
                        "b_gap": b_gap,
                    })

    if not diffs:
        print(f"No comparable results found for tags '{tag_a}' and '{tag_b}'")
        return

    diffs.sort(key=lambda d: d["speedup"])

    print(f"Comparing '{tag_a}' vs '{tag_b}' ({len(diffs)} shapes)")
    print(f"{'='*70}")

    # Regressions (b slower than a)
    regressions = [d for d in diffs if d["speedup"] > 1.05]
    improvements = [d for d in diffs if d["speedup"] < 0.95]
    unchanged = [d for d in diffs if 0.95 <= d["speedup"] <= 1.05]

    if improvements:
        print(f"\nImprovements ({len(improvements)}):")
        for d in improvements[:15]:
            print(f"  {d['speedup']:.2f}x  {d['a_us']:7.1f} -> {d['b_us']:7.1f} us  "
                  f"gap {d['a_gap']:.2f}x -> {d['b_gap']:.2f}x  {d['name']}[{d['shape']}]")

    if regressions:
        print(f"\nRegressions ({len(regressions)}):")
        for d in reversed(regressions[-15:]):
            print(f"  {d['speedup']:.2f}x  {d['a_us']:7.1f} -> {d['b_us']:7.1f} us  "
                  f"gap {d['a_gap']:.2f}x -> {d['b_gap']:.2f}x  {d['name']}[{d['shape']}]")

    print(f"\nSummary: {len(improvements)} improved, {len(regressions)} regressed, "
          f"{len(unchanged)} unchanged")
    geomean = 1.0
    for d in diffs:
        geomean *= d["speedup"]
    geomean = geomean ** (1.0 / len(diffs))
    print(f"Geometric mean speedup ({tag_a} -> {tag_b}): {geomean:.3f}x")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
