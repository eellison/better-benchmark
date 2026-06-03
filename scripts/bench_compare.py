"""
Interleaved N-config benchmark comparison — eliminates cross-run variance.

For each repro, in the SAME worker process, on the SAME GPU, back-to-back:
  1. Compile with each config (N configs)
  2. Capture CUDAGraphs for each config
  3. Acquire exclusive GPU lock
  4. Time all configs in interleaved rounds under the SAME lock
  5. Release lock
  6. Report all timings in a single JSON entry

This eliminates thermal drift, L2 cache state differences, and GPU clock state
variance between measurements.

Usage:
    python scripts/bench_compare.py repros/canonical/ \
        --config "baseline" --label "default" \
        --config "combo_kernels=True,combo_kernel_per_subkernel_blocks=True" --label "combo" \
        --output comparison.json

    python scripts/bench_compare.py repros/canonical/sum_403da67893be/repro.py \
        --config "baseline" --label "default" \
        --config "coordinate_descent_tuning=True" --label "cdt" \
        --rounds 5

Config spec syntax:
    "baseline"                    -> no config changes (default inductor settings)
    "key=True,key2=False"         -> set inductor_config.key = True, etc.
    "combo_kernels=True"          -> inductor_config.combo_kernels = True
    "triton.multi_kernel=1"       -> inductor_config.triton.multi_kernel = 1

Legacy A/B mode (backwards compatible):
    --config-a "baseline" --config-b "combo_kernels=True"
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_lock import discover_gpus, matching_gpus


_SHARED_CACHE_DIR = os.environ.get(
    "TORCHINDUCTOR_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "bench_compare_inductor_cache"),
)


def find_repros(paths: list[Path]) -> list[Path]:
    """Resolve paths to individual repro.py files."""
    repros = []
    for p in paths:
        if p.is_file() and p.name.endswith(".py"):
            repros.append(p)
        elif p.is_dir():
            repros.extend(sorted(p.rglob("repro.py")))
    return repros


def _filter_gpus(gpus: list[dict], selected: str | None) -> list[dict]:
    if not selected:
        return gpus
    wanted = {item.strip() for item in selected.split(",") if item.strip()}
    if not wanted:
        return gpus
    return [gpu for gpu in gpus if gpu["index"] in wanted]


def _parse_config_spec(spec: str) -> dict[str, str]:
    """Parse a config spec string into key=value pairs.

    Returns empty dict for "baseline" (meaning no changes).
    """
    spec = spec.strip()
    if spec.lower() in ("baseline", "default", "none", ""):
        return {}
    pairs = {}
    for item in spec.split(","):
        item = item.strip()
        if "=" not in item:
            # Treat bare names as =True
            pairs[item] = "True"
        else:
            key, val = item.split("=", 1)
            pairs[key.strip()] = val.strip()
    return pairs


def _config_label(spec: str) -> str:
    """Human-readable label for a config spec."""
    spec = spec.strip()
    if spec.lower() in ("baseline", "default", "none", ""):
        return "baseline"
    # Shorten: take key names only, joined with +
    pairs = _parse_config_spec(spec)
    parts = []
    for k, v in pairs.items():
        if v.lower() in ("true", "1"):
            parts.append(k.split(".")[-1])
        elif v.lower() in ("false", "0"):
            parts.append(f"no_{k.split('.')[-1]}")
        else:
            parts.append(f"{k.split('.')[-1]}={v}")
    return "+".join(parts) if parts else "baseline"


def _persistent_compare_worker_script(gpu_idx: str, args_dict: dict) -> str:
    """Generate the worker subprocess script for N-config comparison."""
    configs_list = [_parse_config_spec(spec) for spec in args_dict["configs"]]
    configs_json = json.dumps(configs_list)
    labels_json = json.dumps(args_dict["labels"])

    return f'''
import builtins, contextlib, fcntl, io, re, sys, json, os, tempfile, threading, time
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_idx}"
sys.path.insert(0, "{args_dict["root"]}")

import torch, torch._dynamo
import torch._inductor.config as inductor_config
import torch._inductor.metrics as inductor_metrics
from triton.testing import do_bench
import importlib.util, math
from repro_harness import load_shape_configs, make_inputs_from_config, make_inputs_safely
from byte_accounting import count_bytes_effective

N_WARMUP = {args_dict["n_warmup"]}
N_REP = {args_dict["n_rep"]}
N_ROUNDS = {args_dict["rounds"]}
CONFIGS = {configs_json}
CONFIG_LABELS = {labels_json}
N_CONFIGS = len(CONFIGS)

# --- GPU lock infrastructure (same as bench_parallel) ---
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
        if current_mode == "exclusive" or current_mode == mode:
            local.depth = depth + 1
            try:
                yield
            finally:
                local.depth -= 1
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
            fcntl.flock(gate_fd, fcntl.LOCK_EX)
            fcntl.flock(fd, fcntl.LOCK_EX)
            label = "bench_compare_timing"
            _write_gpu_lock_metadata(fd, gpu, mode, label)
            local.depth = 1
            local.mode = mode
            local.fd = fd
            try:
                yield
            finally:
                local.depth = 0
                local.mode = None
                local.fd = None
        finally:
            _release_fd(fd)
            _release_fd(gate_fd)

# --- Config application ---
def _apply_config(pairs):
    """Apply a dict of config key=value pairs to inductor_config."""
    for key, val in pairs.items():
        # Parse value
        if val.lower() == "true":
            v = True
        elif val.lower() == "false":
            v = False
        elif val.isdigit():
            v = int(val)
        else:
            try:
                v = float(val)
            except ValueError:
                v = val

        # Support dotted keys like "triton.multi_kernel"
        parts = key.split(".")
        obj = inductor_config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], v)

def _save_config_state(pairs):
    """Save current values for keys we are about to modify."""
    saved = {{}}
    for key in pairs:
        parts = key.split(".")
        obj = inductor_config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        saved[key] = getattr(obj, parts[-1])
    return saved

def _restore_config_state(saved):
    """Restore previously saved config values."""
    for key, val in saved.items():
        parts = key.split(".")
        obj = inductor_config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], val)

# --- Benchmark logic ---
def _capture_cudagraph(fn, inputs):
    """Compile, warm up, and capture a CUDAGraph. Returns (graph, fallback_fn).

    If CUDAGraph capture fails, returns (None, fn) for direct-call fallback.
    """
    with torch.no_grad():
        for _ in range(3):
            fn(*inputs)
        torch.cuda.synchronize()
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn(*inputs)
            torch.cuda.synchronize()
            return g, None
        except Exception:
            return None, fn

def _time_graph(graph, fallback_fn, inputs, warmup, rep):
    """Time a captured graph (or fallback fn) using do_bench."""
    if graph is not None:
        return do_bench(lambda: graph.replay(), warmup=warmup, rep=rep, return_mode="min") * 1000
    else:
        return do_bench(lambda: fallback_fn(*inputs), warmup=warmup, rep=rep, return_mode="min") * 1000

def _adaptive_reps(estimate_us):
    """Choose rep count based on estimated kernel time."""
    if estimate_us < 50:
        return 500
    elif estimate_us < 200:
        return 300
    else:
        return 200

def compare_one(repro_path):
    """Run N-config comparison for a single repro. Returns result dict."""
    # Load module
    spec = importlib.util.spec_from_file_location("repro", repro_path)
    mod = importlib.util.module_from_spec(spec)
    mod.device = torch.device
    mod.inf = math.inf
    mod.nan = math.nan
    spec.loader.exec_module(mod)

    instance = mod.Repro()
    make_inputs_fn = mod.make_inputs if hasattr(mod, "make_inputs") else mod._default_make_inputs
    inputs = make_inputs_safely(make_inputs_fn)

    with torch.no_grad():
        eager_out = instance(*inputs)
    total_bytes = count_bytes_effective(instance, inputs)

    # --- Compile and capture CUDAGraph for each config ---
    graphs = []  # list of (graph, fallback_fn) tuples
    for config_pairs in CONFIGS:
        saved = _save_config_state(config_pairs)
        _apply_config(config_pairs)
        torch._dynamo.reset()
        compiled = torch.compile(instance)
        graph, fallback = _capture_cudagraph(compiled, inputs)
        _restore_config_state(saved)
        graphs.append((graph, fallback))

    # --- Quick estimate to choose rep count ---
    quick_times = []
    for graph, fallback in graphs:
        t = _time_graph(graph, fallback, inputs, warmup=5, rep=10)
        quick_times.append(t)
    est_us = min(quick_times)
    rep = _adaptive_reps(est_us)

    # --- Timing phase: HOLD EXCLUSIVE LOCK for ALL configs ---
    all_times = [[] for _ in range(N_CONFIGS)]  # all_times[config_idx] = list of round timings

    with gpu_bench_lock("exclusive"):
        # Warm all graphs under the lock
        for _ in range(N_WARMUP):
            for graph, fallback in graphs:
                if graph is not None:
                    graph.replay()
                else:
                    with torch.no_grad():
                        fallback(*inputs)
        torch.cuda.synchronize()

        # Interleaved timing rounds: config_0, config_1, ..., config_N-1, repeat
        for round_idx in range(N_ROUNDS):
            for cfg_idx, (graph, fallback) in enumerate(graphs):
                t = _time_graph(graph, fallback, inputs, warmup=N_WARMUP, rep=rep)
                all_times[cfg_idx].append(t)

    # Build result dict
    configs_result = {{}}
    for cfg_idx, label in enumerate(CONFIG_LABELS):
        times = all_times[cfg_idx]
        configs_result[label] = {{
            "us": min(times),
            "all_rounds": times,
        }}

    return {{
        "configs": configs_result,
        "total_bytes": total_bytes,
        "rounds": N_ROUNDS,
        "rep_per_round": rep,
    }}

# --- Main loop: read repro paths from stdin, write JSON results to stdout ---
_result_fd = os.dup(1)
_result_file = os.fdopen(_result_fd, "w", buffering=1)
sys.stdout = sys.stderr  # redirect stray prints to stderr

for line in sys.stdin:
    line = line.strip()
    if not line or line == "EXIT":
        break
    try:
        results = compare_one(line)
        results["_repro"] = line
        _result_file.write(json.dumps(results) + "\\n")
        _result_file.flush()
    except Exception as e:
        if "CUDA" in str(e) or "device-side assert" in str(e):
            _result_file.write(f"CUDA_ERROR: {{str(e)[:150]}}\\n")
            _result_file.flush()
            sys.exit(1)
        _result_file.write(f"ERROR: {{str(e)[:150]}}\\n")
        _result_file.flush()
'''


def _locked_worker(gpu: dict, task_queue, result_queue, args_dict):
    """Run persistent worker subprocess on a GPU for N-config comparison."""
    import collections

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu["index"]
    env["INDUCTOR_GPU_BENCH_LOCK"] = "1"
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
        script = _persistent_compare_worker_script(gpu["index"], args_dict)
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
            name=f"compare-worker-stderr-gpu-{gpu['index']}",
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

    while True:
        try:
            repro_path = task_queue.get_nowait()
        except Exception:
            break

        # Ensure live worker
        if proc is None or proc.poll() is not None:
            _kill_worker(proc)
            proc = _spawn_worker()

        start = time.time()
        try:
            proc.stdin.write(str(repro_path) + "\n")
            proc.stdin.flush()

            line = proc.stdout.readline()
            elapsed = time.time() - start

            if not line:
                returncode = proc.poll()
                error = f"worker exited without result (rc={returncode})"
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
                    "error": error[:500],
                })
                continue

            line = line.strip()
            if line.startswith("{"):
                results = json.loads(line)
                result_repro = results.pop("_repro", None)
                expected = str(repro_path)
                if result_repro is not None and result_repro != expected:
                    error = (
                        f"result misalignment: expected "
                        f"'{Path(expected).parent.name}' but got "
                        f"'{Path(result_repro).parent.name}'"
                    )
                    _kill_worker(proc)
                    proc = None
                    result_queue.put({
                        "repro": str(repro_path),
                        "gpu": gpu["index"],
                        "status": "failed",
                        "elapsed": elapsed,
                        "error": error[:500],
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
                    "error": error[:500],
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
                    "error": error[:500],
                })
        except Exception as e:
            _kill_worker(proc)
            proc = None
            result_queue.put({
                "repro": str(repro_path),
                "gpu": gpu["index"],
                "status": "failed",
                "elapsed": time.time() - start,
                "error": str(e)[:200],
            })

    _kill_worker(proc)


def main():
    parser = argparse.ArgumentParser(
        description="Interleaved N-config benchmark comparison (eliminates cross-run variance)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare 3 configs across all canonical repros
  python scripts/bench_compare.py repros/canonical/ \\
      --config "baseline" --label "default" \\
      --config "combo_kernels=True" --label "combo" \\
      --config "coordinate_descent_tuning=True" --label "cdt" \\
      --output comparison.json

  # Legacy A/B mode (still supported)
  python scripts/bench_compare.py repros/canonical/ \\
      --config-a "baseline" \\
      --config-b "combo_kernels=True" \\
      --output comparison.json
""",
    )
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("repros/canonical")],
                        help="repro.py files or directories to benchmark")
    # New N-config interface
    parser.add_argument("--config", action="append", default=None,
                        help='Config spec (repeatable). E.g., --config "baseline" --config "combo_kernels=True"')
    parser.add_argument("--label", action="append", default=None,
                        help='Label for config (repeatable, same count as --config)')
    # Legacy A/B interface (backwards compatible)
    parser.add_argument("--config-a", default=None,
                        help='[Legacy] Config A spec')
    parser.add_argument("--config-b", default=None,
                        help='[Legacy] Config B spec')
    parser.add_argument("--device-kind", default=None,
                        help="GPU kind filter (e.g., H100, B200)")
    parser.add_argument("--gpus", default=None,
                        help="Comma-separated GPU indices to use")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Max parallel workers (default: one per GPU)")
    parser.add_argument("--n-warmup", type=int, default=25,
                        help="Warmup iterations per do_bench call")
    parser.add_argument("--n-rep", type=int, default=200,
                        help="Base rep count (auto-adjusted for fast kernels)")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Number of interleaved timing rounds")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write comparison results to JSON file")
    parser.add_argument("--share-cache", action="store_true", default=True,
                        help="Share inductor cache across workers (default: enabled)")
    parser.add_argument("--no-share-cache", dest="share_cache", action="store_false")
    args = parser.parse_args()

    # Resolve configs: either --config (new) or --config-a/--config-b (legacy)
    if args.config:
        configs = args.config
        labels = args.label if args.label else [_config_label(c) for c in configs]
        if len(labels) != len(configs):
            parser.error(f"--label count ({len(labels)}) must match --config count ({len(configs)})")
    elif args.config_a is not None and args.config_b is not None:
        configs = [args.config_a, args.config_b]
        labels = [_config_label(args.config_a), _config_label(args.config_b)]
    else:
        parser.error("Must provide either --config (repeatable) or --config-a and --config-b")

    if len(configs) < 2:
        parser.error("Need at least 2 configs for comparison")

    repros = find_repros(args.paths)
    if not repros:
        print("No repro.py files found.")
        return

    gpus = _filter_gpus(matching_gpus(args.device_kind), args.gpus)
    if not gpus:
        print("No matching GPUs found.")
        return

    n_workers = args.max_workers or len(gpus)
    n_workers = min(n_workers, len(gpus), len(repros))

    print(f"N-Config Comparison ({len(configs)} configs):")
    for i, (cfg, lbl) in enumerate(zip(configs, labels)):
        print(f"  Config {i} [{lbl}]: {cfg}")
    print(f"  Repros: {len(repros)}")
    print(f"  Workers: {n_workers} across {len(gpus)} GPUs")
    print(f"  Rounds: {args.rounds} interleaved per repro")
    print(f"  Base reps: {args.n_rep} (adaptive for fast kernels)")
    print()

    # Fill task queue
    task_queue = queue.Queue()
    for r in repros:
        task_queue.put(r)
    result_queue = queue.Queue()
    root = str(Path(__file__).resolve().parents[1])

    args_dict = {
        "root": root,
        "configs": configs,
        "labels": labels,
        "n_warmup": args.n_warmup,
        "n_rep": args.n_rep,
        "rounds": args.rounds,
        "share_cache": args.share_cache,
    }

    # Launch workers
    workers = []
    for i in range(n_workers):
        gpu = gpus[i % len(gpus)]
        t = threading.Thread(
            target=_locked_worker,
            args=(gpu, task_queue, result_queue, args_dict),
            daemon=True,
        )
        t.start()
        workers.append(t)

    # Collect results
    all_results = {}
    failures = {}
    done = 0
    failed = 0
    start_time = time.time()

    while done + failed < len(repros):
        try:
            result = result_queue.get(timeout=120)
        except Exception:
            elapsed = time.time() - start_time
            alive = sum(1 for t in workers if t.is_alive())
            if alive == 0:
                print(f"\n[WARN] All workers dead! {done} ok, {failed} failed. Aborting.")
                break
            remaining = len(repros) - done - failed
            print(f"  [{done+failed}/{len(repros)}] ... {alive} workers alive, "
                  f"{remaining} remaining, elapsed {elapsed/60:.1f}min", flush=True)
            continue

        repro_name = Path(result["repro"]).parent.name
        if result["status"] == "ok":
            done += 1
            r = result["results"]
            configs_data = r["configs"]
            # Show timing summary
            timing_parts = []
            for lbl in labels:
                if lbl in configs_data:
                    timing_parts.append(f"{lbl}={configs_data[lbl]['us']:.1f}us")
            timing_str = "  ".join(timing_parts)
            # Find fastest
            min_us = min(configs_data[lbl]["us"] for lbl in labels if lbl in configs_data)
            fastest = [lbl for lbl in labels if lbl in configs_data and configs_data[lbl]["us"] == min_us][0]
            print(f"  [{done+failed}/{len(repros)}] OK  gpu={result['gpu']}  "
                  f"{result['elapsed']:.1f}s  {timing_str}  "
                  f"fastest={fastest}  {repro_name}", flush=True)
            all_results[result["repro"]] = {
                "configs": configs_data,
                "total_bytes": r.get("total_bytes"),
                "rounds": r.get("rounds"),
                "rep_per_round": r.get("rep_per_round"),
            }
        else:
            failed += 1
            failures[result["repro"]] = {
                "error": result.get("error", ""),
                "gpu": result.get("gpu"),
                "elapsed": result.get("elapsed"),
            }
            print(f"  [{done+failed}/{len(repros)}] FAIL gpu={result['gpu']}  "
                  f"{result['elapsed']:.1f}s  {repro_name}: {result['error'][:80]}", flush=True)

        # Incremental save every 25 results
        if args.output and (done + failed) % 25 == 0:
            _write_output(args.output, all_results, failures, configs, labels,
                          len(repros), done, failed, time.time() - start_time)

    # Wait for workers
    for t in workers:
        t.join(timeout=10)

    elapsed_total = time.time() - start_time
    print(f"\nDone: {done} ok, {failed} failed in {elapsed_total:.1f}s")

    # Print summary
    if all_results:
        _print_summary(all_results, labels)

    # Write output
    if args.output:
        _write_output(args.output, all_results, failures, configs, labels,
                      len(repros), done, failed, elapsed_total)
        print(f"\n[output] Wrote {args.output}")


def _write_output(output_path: Path, all_results: dict, failures: dict,
                  configs: list[str], labels: list[str], total: int, done: int,
                  failed: int, elapsed: float):
    """Write comparison results to JSON."""
    commit = ""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        pass

    payload = {
        "_metadata": {
            "tool": "bench_compare.py",
            "n_configs": len(configs),
            "configs": dict(zip(labels, configs)),
            "labels": labels,
            "commit": commit,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total": total,
            "ok": done,
            "failed": failed,
            "elapsed_s": elapsed,
        },
    }
    payload.update(all_results)
    if failures:
        payload["__failures__"] = failures

    output_path.write_text(json.dumps(payload, indent=2))


def _print_summary(all_results: dict, labels: list[str]):
    """Print a summary of the N-config comparison."""
    import math

    # Collect per-config timings
    config_timings = {lbl: [] for lbl in labels}
    win_counts = {lbl: 0 for lbl in labels}
    n_repros = 0

    for repro_path, data in all_results.items():
        configs_data = data.get("configs", {})
        timings = {}
        for lbl in labels:
            if lbl in configs_data:
                timings[lbl] = configs_data[lbl]["us"]
        if len(timings) == len(labels):
            n_repros += 1
            for lbl, t in timings.items():
                config_timings[lbl].append(t)
            fastest_lbl = min(timings, key=timings.get)
            win_counts[fastest_lbl] += 1

    if n_repros == 0:
        return

    print(f"\n{'='*70}")
    print(f"N-Config Summary ({len(labels)} configs, {n_repros} repros)")
    print(f"{'='*70}")

    # Geometric mean timing for each config
    print(f"\nGeometric mean timing (us):")
    geomeans = {}
    for lbl in labels:
        log_sum = sum(math.log(t) for t in config_timings[lbl])
        geomeans[lbl] = math.exp(log_sum / n_repros)
        print(f"  {lbl:30s}: {geomeans[lbl]:8.2f} us")

    # Win counts
    print(f"\nFastest config wins:")
    for lbl in labels:
        pct = win_counts[lbl] * 100 / n_repros
        print(f"  {lbl:30s}: {win_counts[lbl]:4d} ({pct:5.1f}%)")

    # Speedup ratios vs first config (baseline)
    baseline_lbl = labels[0]
    print(f"\nSpeedup vs '{baseline_lbl}' (geomean ratio, >1 = faster than baseline):")
    for lbl in labels[1:]:
        # ratio = baseline_geomean / this_geomean
        ratios = []
        for i in range(n_repros):
            r = config_timings[baseline_lbl][i] / config_timings[lbl][i]
            ratios.append(r)
        log_sum = sum(math.log(r) for r in ratios)
        geomean_ratio = math.exp(log_sum / n_repros)
        if geomean_ratio >= 1:
            print(f"  {lbl:30s}: {geomean_ratio:.4f}x faster")
        else:
            print(f"  {lbl:30s}: {1/geomean_ratio:.4f}x slower")

    # Top regressions and improvements for each non-baseline config
    for lbl in labels[1:]:
        ratios_with_names = []
        for repro_path in all_results:
            data = all_results[repro_path]
            configs_data = data.get("configs", {})
            if baseline_lbl in configs_data and lbl in configs_data:
                baseline_us = configs_data[baseline_lbl]["us"]
                this_us = configs_data[lbl]["us"]
                ratio = baseline_us / this_us
                ratios_with_names.append((ratio, Path(repro_path).parent.name, baseline_us, this_us))

        if not ratios_with_names:
            continue

        ratios_with_names.sort(key=lambda x: x[0], reverse=True)
        print(f"\n  Top improvements for '{lbl}' vs '{baseline_lbl}':")
        for ratio, name, base_us, this_us in ratios_with_names[:10]:
            print(f"    {ratio:5.3f}x  base={base_us:7.1f}us  {lbl}={this_us:7.1f}us  {name}")

        print(f"\n  Top regressions for '{lbl}' vs '{baseline_lbl}':")
        for ratio, name, base_us, this_us in ratios_with_names[-10:]:
            print(f"    {ratio:5.3f}x  base={base_us:7.1f}us  {lbl}={this_us:7.1f}us  {name}")


if __name__ == "__main__":
    main()
