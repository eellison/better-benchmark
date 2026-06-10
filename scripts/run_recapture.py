#!/usr/bin/env python3
"""
Wave-1 mass recapture script.

Runs models from timm/hf/torchbench suites through torch.compile with the
capture hook installed, producing canonical repros under --corpus-root.

DTYPE POLICY (settled):
  - mode=infer: model + inputs in bf16
  - mode=train: autocast bf16 + loss.backward() (no optimizer step)

DISTRIBUTED POLICY (settled):
  - On process-group/collective errors or known-distributed models, skip
    with a recorded reason in <corpus-root>/skipped.json.

RESUMABILITY:
  - On restart with the same --corpus-root, models with status=ok in
    run_log.json are skipped (--force to redo).
  - run_log entries written atomically (write temp + rename) per model.

Usage:
  # Single model test:
  python scripts/run_recapture.py --suite timm --mode infer \\
      --models mobilenetv2_100 --corpus-root /tmp/recapture_test/repros

  # Full suite:
  python scripts/run_recapture.py --suite timm --mode infer \\
      --models all --corpus-root /tmp/wave1_corpus/repros

  # Multi-GPU workers:
  python scripts/run_recapture.py --suite timm --mode both \\
      --models all --corpus-root /tmp/wave1_corpus/repros --gpus 0,1
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PYTORCH_DYNAMO_DIR = Path("/tmp/pytorch-work/benchmarks/dynamo")
TORCHBENCH_DIR = Path("/tmp/pytorch-work/torchbenchmark")

# Known-distributed models that require process groups and cannot be captured
# on a single GPU. Add to this list as new ones are discovered.
KNOWN_DISTRIBUTED_MODELS = {
    "simple_gpt_tp_manual",
    "moco",
    "dlrm",  # uses dist.all_reduce in some configs
}

# Errors that indicate a distributed requirement
DISTRIBUTED_ERROR_PATTERNS = [
    "ProcessGroup",
    "process_group",
    "c10d",
    "distributed",
    "NCCL",
    "all_reduce",
    "all_gather",
    "broadcast",
    "DistBackendError",
    "Backend.NCCL",
    "init_process_group",
]


# ============================================================================
# Model list loading
# ============================================================================

def _parse_model_list_line(line: str) -> tuple[str, int] | None:
    """Parse a model list line (supports both comma and space separators)."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    # Try comma first, then whitespace
    if "," in line:
        parts = line.split(",")
    else:
        parts = line.split()
    if len(parts) >= 2:
        name = parts[0].strip()
        try:
            bs = int(parts[1].strip())
            return name, bs
        except ValueError:
            return None
    return None


def load_timm_models() -> dict[str, int]:
    """Load timm model names and batch sizes from the dynamo runner list."""
    path = PYTORCH_DYNAMO_DIR / "timm_models_list.txt"
    models = {}
    for line in path.read_text().splitlines():
        parsed = _parse_model_list_line(line)
        if parsed:
            models[parsed[0]] = parsed[1]
    return models


def load_hf_models() -> dict[str, int]:
    """Load HuggingFace model names and batch sizes."""
    path = PYTORCH_DYNAMO_DIR / "huggingface_models_list.txt"
    # Models that need HF auth or special downloads
    skip = {
        "meta-llama/Llama-3.2-1B", "google/gemma-2-2b", "google/gemma-3-4b-it",
        "openai/whisper-tiny", "Qwen/Qwen3-0.6B",
        "mistralai/Mistral-7B-Instruct-v0.3", "openai/gpt-oss-20b",
    }
    models = {}
    for line in path.read_text().splitlines():
        parsed = _parse_model_list_line(line)
        if parsed and parsed[0] not in skip:
            models[parsed[0]] = parsed[1]
    return models


def load_torchbench_models() -> dict[str, int]:
    """Load torchbench model names and batch sizes."""
    path = PYTORCH_DYNAMO_DIR / "all_torchbench_models_list.txt"
    # Fall back to CI list if all-models list doesn't exist
    if not path.exists():
        path = PYTORCH_DYNAMO_DIR / "torchbench_models_list.txt"
    skip = {
        "Background_Matting", "llama_v2_7b_16h", "simple_gpt_tp_manual",
        "mobilenet_v2_quantized_qat", "resnet50_quantized_qat",
        "microbench_unbacked_tolist_sum", "cm3leon_generate",
        "hf_T5_generate", "sam", "timm_vision_transformer_large",
        "stable_diffusion_unet",
    }
    models = {}
    for line in path.read_text().splitlines():
        parsed = _parse_model_list_line(line)
        if parsed and parsed[0] not in skip:
            models[parsed[0]] = parsed[1]
    return models


def get_suite_models(suite: str) -> dict[str, int]:
    """Get model name -> batch_size mapping for a suite."""
    if suite == "timm":
        return load_timm_models()
    elif suite == "hf":
        return load_hf_models()
    elif suite == "torchbench":
        return load_torchbench_models()
    else:
        raise ValueError(f"Unknown suite: {suite}")


# ============================================================================
# Run log (atomic, resumable)
# ============================================================================

def load_run_log(corpus_root: Path) -> dict:
    """Load the run log. Returns {model_key: entry} dict."""
    log_path = corpus_root / "run_log.json"
    if not log_path.exists():
        return {}
    with open(log_path) as f:
        entries = json.load(f)
    # Index by model key
    return {e["model_key"]: e for e in entries}


def write_run_log_entry(corpus_root: Path, entry: dict):
    """Atomically append/update a run log entry.

    Reads the full log, updates the entry for the model_key, writes to a
    temp file, then renames (atomic on same filesystem).
    """
    log_path = corpus_root / "run_log.json"
    if log_path.exists():
        with open(log_path) as f:
            entries = json.load(f)
    else:
        entries = []

    # Update or append
    model_key = entry["model_key"]
    found = False
    for i, e in enumerate(entries):
        if e["model_key"] == model_key:
            entries[i] = entry
            found = True
            break
    if not found:
        entries.append(entry)

    # Atomic write: temp file + rename
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(corpus_root), prefix=".run_log_", suffix=".json"
    )
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(entries, f, indent=2)
        os.replace(tmp_path, str(log_path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def write_skipped_entry(corpus_root: Path, model: str, suite: str, mode: str, reason: str):
    """Record a skipped model in <corpus-root>/skipped.json."""
    skip_path = corpus_root / "skipped.json"
    if skip_path.exists():
        with open(skip_path) as f:
            skipped = json.load(f)
    else:
        skipped = []

    skipped.append({
        "model": model,
        "suite": suite,
        "mode": mode,
        "reason": reason,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(corpus_root), prefix=".skipped_", suffix=".json"
    )
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(skipped, f, indent=2)
        os.replace(tmp_path, str(skip_path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ============================================================================
# Single-model capture (runs in a subprocess for GPU isolation)
# ============================================================================

def _is_distributed_error(error_str: str) -> bool:
    """Check if an error string indicates a distributed requirement."""
    error_lower = error_str.lower()
    for pattern in DISTRIBUTED_ERROR_PATTERNS:
        if pattern.lower() in error_lower:
            return True
    return False


def capture_single_model(
    suite: str,
    model_name: str,
    mode: str,
    batch_size: int,
    corpus_root: Path,
    timeout: int = 300,
) -> dict:
    """Capture a single model. Returns a run_log entry dict.

    This function is designed to be called in the worker process (via
    --_worker mode), not in the orchestrator.
    """
    import torch
    import torch._dynamo
    from torch._inductor.utils import fresh_inductor_cache
    from capture_hook import install_capture_hook, uninstall_capture_hook
    from merge_captures import temporary_capture_for_merge

    model_key = f"{suite}/{mode}/{model_name}"
    t0 = time.time()
    entry = {
        "model_key": model_key,
        "suite": suite,
        "model": model_name,
        "mode": mode,
        "status": "failed",
        "wall_time_s": 0,
        "region_count": 0,
        "roundtrip_summary": {"ok": 0, "failed": 0},
        "error_tail": None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Check known-distributed
    if model_name in KNOWN_DISTRIBUTED_MODELS:
        entry["status"] = "skipped"
        entry["error_tail"] = f"Known distributed model: {model_name}"
        entry["wall_time_s"] = time.time() - t0
        return entry

    canonical_dir = corpus_root
    model_dir = corpus_root / "models" / suite / mode / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    model_obj = None
    inputs = None

    try:
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        gc.collect()

        with temporary_capture_for_merge(
            canonical_dir,
            model_name,
            suite=suite,
            mode=mode,
            prefix=f"recapture_{suite}_",
        ) as capture:
            cap_dir = capture.capture_dir

            # --- Load model per suite ---
            if suite == "timm":
                model_obj, inputs = _load_timm(model_name, batch_size, mode)
            elif suite == "hf":
                model_obj, inputs = _load_hf(model_name, batch_size, mode)
            elif suite == "torchbench":
                model_obj, inputs = _load_torchbench(model_name, batch_size, mode)
            else:
                raise ValueError(f"Unknown suite: {suite}")

            # Install capture hook
            label = f"{suite}_{model_name}_{mode}"
            install_capture_hook(
                str(cap_dir), label=label, graph_dir=str(model_dir), validate=True
            )

            # Compile and run
            with fresh_inductor_cache():
                compiled = torch.compile(model_obj)

                if mode == "train":
                    # DTYPE POLICY: autocast bf16 + backward
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        if isinstance(inputs, dict):
                            out = compiled(**inputs)
                        elif isinstance(inputs, (list, tuple)):
                            out = compiled(*inputs)
                        else:
                            out = compiled(inputs)

                    # Compute loss and backward
                    loss = _reduce_output_to_loss(out)
                    loss.backward()
                else:
                    # DTYPE POLICY: model + inputs bf16 for inference
                    with torch.no_grad():
                        if isinstance(inputs, dict):
                            compiled(**inputs)
                        elif isinstance(inputs, (list, tuple)):
                            compiled(*inputs)
                        else:
                            compiled(inputs)

                torch.cuda.synchronize()

            uninstall_capture_hook()

            # Merge into canonical set
            n_regions = capture.merge()
            elapsed = time.time() - t0

            # Count roundtrip results from full_graph meta sidecars
            rt_ok, rt_failed = _count_roundtrips(model_dir)

            entry["status"] = "ok"
            entry["region_count"] = n_regions
            entry["wall_time_s"] = elapsed
            entry["roundtrip_summary"] = {"ok": rt_ok, "failed": rt_failed}

    except Exception as e:
        elapsed = time.time() - t0
        error_str = str(e)
        tb_str = ""
        try:
            import traceback
            tb_str = traceback.format_exc()
        except Exception:
            pass

        # Check if this is a distributed error -> skip
        if _is_distributed_error(error_str) or _is_distributed_error(tb_str):
            entry["status"] = "skipped"
            entry["error_tail"] = f"Distributed requirement: {error_str[:500]}"
        else:
            entry["status"] = "failed"
            # Keep last 1000 chars of error for the log
            full_error = tb_str if tb_str else error_str
            entry["error_tail"] = full_error[-1000:]

        entry["wall_time_s"] = elapsed

        try:
            uninstall_capture_hook()
        except Exception:
            pass

    finally:
        del model_obj, inputs
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    return entry


def _reduce_output_to_loss(out):
    """Reduce model output to a scalar loss for backward."""
    import torch
    if hasattr(out, "loss") and out.loss is not None:
        return out.loss
    if isinstance(out, torch.Tensor):
        return out.sum()
    if isinstance(out, (tuple, list)):
        tensors = [o for o in out if isinstance(o, torch.Tensor) and o.requires_grad]
        if not tensors:
            tensors = [o for o in out if isinstance(o, torch.Tensor)]
        if tensors:
            return sum(t.sum() for t in tensors)
    if isinstance(out, dict):
        tensors = [v for v in out.values() if isinstance(v, torch.Tensor)]
        if tensors:
            return sum(t.sum() for t in tensors)
    raise RuntimeError(f"Cannot reduce output of type {type(out)} to loss")


def _count_roundtrips(model_dir: Path) -> tuple[int, int]:
    """Count roundtrip:ok vs roundtrip:failed in full_graph meta sidecars.

    The roundtrip stamp is written by the write-gate into
    <model_dir>/full_graph_NNN.meta.json files.
    """
    ok = 0
    failed = 0
    model_path = Path(model_dir)
    for meta_file in model_path.glob("full_graph_*.meta.json"):
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            rt = meta.get("roundtrip", "")
            if rt == "ok":
                ok += 1
            elif rt and rt != "ok":
                failed += 1
            else:
                # No roundtrip stamp at all — count as neither (write-gate
                # may not be available or may have errored)
                pass
        except Exception:
            failed += 1
    return ok, failed


# ============================================================================
# Model loading helpers (per suite) — reusing patterns from recapture_*.py
# ============================================================================

def _load_timm(model_name: str, batch_size: int, mode: str):
    """Load a timm model with bf16 dtype policy."""
    import timm
    import torch

    model = timm.create_model(model_name, pretrained=False).cuda()
    model = model.to(memory_format=torch.channels_last)

    # DTYPE POLICY: infer = bf16 model + inputs; train = fp32 model (autocast wraps)
    if mode == "infer":
        model = model.to(dtype=torch.bfloat16)

    data_config = timm.data.resolve_model_data_config(model)
    input_size = data_config.get("input_size", (3, 224, 224))

    if mode == "infer":
        inp = torch.randn(batch_size, *input_size, device="cuda", dtype=torch.bfloat16)
    else:
        inp = torch.randn(batch_size, *input_size, device="cuda")
    inp = inp.to(memory_format=torch.channels_last)

    if mode == "train":
        model.train()
    else:
        model.eval()

    return model, [inp]


def _load_hf(model_name: str, batch_size: int, mode: str):
    """Load a HuggingFace model with bf16 dtype policy."""
    import torch

    sys.path.insert(0, str(PYTORCH_DYNAMO_DIR))
    import common  # noqa: F401
    from huggingface import HuggingfaceRunner

    # Set up minimal args for the runner
    sys.argv = [
        "capture", "--performance",
        "--training" if mode == "train" else "--inference",
        "--inductor", "--devices", "cuda",
        "--batch-size", str(batch_size), "--only", model_name,
    ]
    runner = HuggingfaceRunner()
    args = common.parse_args(sys.argv[1:])
    runner.args = args
    runner.model_iter_fn = (
        runner.forward_and_backward_pass if mode == "train" else runner.forward_pass
    )

    _, _, model, example_inputs, _ = runner.load_model("cuda", model_name, batch_size)

    if mode == "train":
        model.train()
    else:
        model.eval()
        # DTYPE POLICY: infer = bf16
        model = model.to(dtype=torch.bfloat16)
        if isinstance(example_inputs, dict):
            example_inputs = {
                k: v.to(dtype=torch.bfloat16) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                for k, v in example_inputs.items()
            }
        elif isinstance(example_inputs, (list, tuple)):
            example_inputs = type(example_inputs)(
                v.to(dtype=torch.bfloat16) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                for v in example_inputs
            )

    return model, example_inputs


def _load_torchbench(model_name: str, batch_size: int, mode: str):
    """Load a TorchBench model with bf16 dtype policy."""
    import importlib
    import torch

    sys.path.insert(0, str(TORCHBENCH_DIR))

    original_dir = os.getcwd()
    try:
        os.chdir(str(TORCHBENCH_DIR))
        module = importlib.import_module(f"torchbenchmark.models.{model_name}")
        benchmark_cls = module.Model

        test_mode = "train" if mode == "train" else "eval"
        kwargs = {"test": test_mode, "device": "cuda"}

        allow_custom = getattr(benchmark_cls, "ALLOW_CUSTOMIZE_BSIZE", True)
        if allow_custom and batch_size is not None:
            kwargs["batch_size"] = batch_size

        benchmark = benchmark_cls(**kwargs)
        model, example_inputs = benchmark.get_module()

        if mode == "train":
            model.train()
        else:
            model.eval()
            # DTYPE POLICY: infer = bf16
            model = model.to(dtype=torch.bfloat16)
            if isinstance(example_inputs, dict):
                example_inputs = {
                    k: v.to(dtype=torch.bfloat16) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                    for k, v in example_inputs.items()
                }
            elif isinstance(example_inputs, (list, tuple)):
                example_inputs = type(example_inputs)(
                    v.to(dtype=torch.bfloat16) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                    for v in example_inputs
                )

        return model, example_inputs
    finally:
        os.chdir(original_dir)


# ============================================================================
# Worker subprocess entry point
# ============================================================================

def run_worker(suite: str, model_name: str, mode: str, batch_size: int,
               corpus_root: Path, timeout: int = 300):
    """Run capture for a single model in a subprocess. Returns the run_log entry."""
    env = os.environ.copy()
    # CUDA_VISIBLE_DEVICES is set by the orchestrator per worker

    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--_worker",
        "--suite", suite,
        "--mode", mode,
        "--models", model_name,
        "--corpus-root", str(corpus_root),
        "--timeout", str(timeout),
    ]
    if batch_size is not None:
        cmd += ["--batch-size", str(batch_size)]

    model_key = f"{suite}/{mode}/{model_name}"

    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=timeout + 60,  # extra margin for subprocess overhead
            cwd=str(PROJECT_ROOT),
        )

        # Parse the JSON result from stdout (last line starting with RESULT_JSON:)
        for line in reversed(result.stdout.splitlines()):
            if line.startswith("RESULT_JSON:"):
                entry = json.loads(line[len("RESULT_JSON:"):])
                return entry

        # No result line — process failed
        stderr_tail = result.stderr[-1000:] if result.stderr else ""
        stdout_tail = result.stdout[-500:] if result.stdout else ""
        error_tail = f"exit={result.returncode}\nstderr: {stderr_tail}\nstdout: {stdout_tail}"

        # Check if distributed error
        if _is_distributed_error(error_tail):
            return {
                "model_key": model_key,
                "suite": suite, "model": model_name, "mode": mode,
                "status": "skipped",
                "wall_time_s": 0, "region_count": 0,
                "roundtrip_summary": {"ok": 0, "failed": 0},
                "error_tail": f"Distributed requirement (subprocess): {error_tail[:500]}",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

        return {
            "model_key": model_key,
            "suite": suite, "model": model_name, "mode": mode,
            "status": "failed",
            "wall_time_s": 0, "region_count": 0,
            "roundtrip_summary": {"ok": 0, "failed": 0},
            "error_tail": error_tail[-1000:],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    except subprocess.TimeoutExpired:
        return {
            "model_key": model_key,
            "suite": suite, "model": model_name, "mode": mode,
            "status": "failed",
            "wall_time_s": timeout,
            "region_count": 0,
            "roundtrip_summary": {"ok": 0, "failed": 0},
            "error_tail": f"Subprocess timeout after {timeout + 60}s",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    except Exception as e:
        return {
            "model_key": model_key,
            "suite": suite, "model": model_name, "mode": mode,
            "status": "failed",
            "wall_time_s": 0, "region_count": 0,
            "roundtrip_summary": {"ok": 0, "failed": 0},
            "error_tail": f"Subprocess launch error: {str(e)[:500]}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }


# ============================================================================
# Orchestrator
# ============================================================================

def run_orchestrator(args):
    """Main orchestrator: iterates models, dispatches workers, manages log."""
    corpus_root = Path(args.corpus_root)
    corpus_root.mkdir(parents=True, exist_ok=True)

    # Load suite models
    all_models = get_suite_models(args.suite)

    # Resolve model list (accept space-separated nargs AND comma-joined)
    if args.models == ["all"]:
        models = list(all_models.keys())
    else:
        models = [m for arg in args.models for m in arg.split(",") if m]

    # Resolve modes
    if args.mode == "both":
        modes = ["infer", "train"]
    else:
        modes = [args.mode]

    # Build work list: (model_name, mode, batch_size)
    work = []
    for model_name in models:
        bs = all_models.get(model_name, 128)  # default batch size
        for mode in modes:
            work.append((model_name, mode, bs))

    # Load existing run log for resume
    run_log = load_run_log(corpus_root)

    # Filter already-done models (resume)
    if not args.force:
        pending = []
        for model_name, mode, bs in work:
            model_key = f"{args.suite}/{mode}/{model_name}"
            existing = run_log.get(model_key)
            if existing and existing.get("status") == "ok":
                continue
            pending.append((model_name, mode, bs))
    else:
        pending = work

    if args.dry_run:
        print(f"DRY RUN: {len(pending)} models to capture ({len(work) - len(pending)} already done)")
        print(f"Suite: {args.suite}, Modes: {modes}")
        print(f"Corpus root: {corpus_root}")
        print()
        for model_name, mode, bs in pending:
            print(f"  {args.suite}/{mode}/{model_name} (batch={bs})")
        print(f"\nSkipped (already ok): {len(work) - len(pending)}")
        return

    print(f"{'='*70}")
    print(f"  RECAPTURE: {args.suite} | modes={modes} | {len(pending)} pending / {len(work)} total")
    print(f"  Corpus: {corpus_root}")
    print(f"{'='*70}")

    # Determine GPU assignment
    if args.gpus:
        gpu_list = [g.strip() for g in args.gpus.split(",")]
    else:
        gpu_list = [os.environ.get("CUDA_VISIBLE_DEVICES", "0")]

    # Sequential execution (one model at a time per GPU)
    # For multi-GPU, use a simple round-robin with concurrent.futures
    if len(gpu_list) > 1:
        _run_multi_gpu(args, pending, gpu_list, corpus_root)
    else:
        _run_sequential(args, pending, gpu_list[0], corpus_root)


def _run_sequential(args, pending, gpu_id: str, corpus_root: Path):
    """Run models sequentially on a single GPU."""
    total = len(pending)
    for i, (model_name, mode, bs) in enumerate(pending):
        model_key = f"{args.suite}/{mode}/{model_name}"
        print(f"\n[{i+1}/{total}] {model_key} (batch={bs})")

        # Set GPU for the subprocess
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

        entry = run_worker(
            suite=args.suite,
            model_name=model_name,
            mode=mode,
            batch_size=bs,
            corpus_root=corpus_root,
            timeout=args.timeout,
        )

        # Write atomically
        write_run_log_entry(corpus_root, entry)

        status = entry["status"]
        regions = entry.get("region_count", 0)
        wall = entry.get("wall_time_s", 0)
        rt = entry.get("roundtrip_summary", {})

        if status == "skipped":
            write_skipped_entry(corpus_root, model_name, args.suite, mode,
                                entry.get("error_tail", "unknown"))
            print(f"  SKIPPED: {entry.get('error_tail', '')[:100]}")
        elif status == "ok":
            print(f"  OK: {regions} regions, {wall:.1f}s, roundtrip: {rt.get('ok',0)} ok / {rt.get('failed',0)} failed")
        else:
            print(f"  FAILED: {entry.get('error_tail', '')[:200]}")

    _print_summary(corpus_root, args.suite)


def _run_multi_gpu(args, pending, gpu_list: list[str], corpus_root: Path):
    """Run models across multiple GPUs with a process pool."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    total = len(pending)
    n_gpus = len(gpu_list)

    # We'll use futures but control GPU assignment via env var in the subprocess
    # Since run_worker spawns a subprocess, we just need to set CUDA_VISIBLE_DEVICES
    # in the env for each call. Use a thread pool (not process) to dispatch.
    from concurrent.futures import ThreadPoolExecutor

    completed = 0

    def _worker_wrapper(idx, model_name, mode, bs, gpu_id):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        return run_worker(
            suite=args.suite,
            model_name=model_name,
            mode=mode,
            batch_size=bs,
            corpus_root=corpus_root,
            timeout=args.timeout,
        )

    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = {}
        for i, (model_name, mode, bs) in enumerate(pending):
            gpu_id = gpu_list[i % n_gpus]
            fut = executor.submit(_worker_wrapper, i, model_name, mode, bs, gpu_id)
            futures[fut] = (model_name, mode, bs, i)

        for fut in as_completed(futures):
            model_name, mode, bs, idx = futures[fut]
            model_key = f"{args.suite}/{mode}/{model_name}"
            completed += 1

            try:
                entry = fut.result()
            except Exception as e:
                entry = {
                    "model_key": model_key,
                    "suite": args.suite, "model": model_name, "mode": mode,
                    "status": "failed",
                    "wall_time_s": 0, "region_count": 0,
                    "roundtrip_summary": {"ok": 0, "failed": 0},
                    "error_tail": f"Worker exception: {str(e)[:500]}",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }

            write_run_log_entry(corpus_root, entry)

            status = entry["status"]
            regions = entry.get("region_count", 0)
            wall = entry.get("wall_time_s", 0)

            if status == "skipped":
                write_skipped_entry(corpus_root, model_name, args.suite, mode,
                                    entry.get("error_tail", "unknown"))
                print(f"  [{completed}/{total}] {model_key}: SKIPPED")
            elif status == "ok":
                print(f"  [{completed}/{total}] {model_key}: OK ({regions} regions, {wall:.1f}s)")
            else:
                print(f"  [{completed}/{total}] {model_key}: FAILED")

    _print_summary(corpus_root, args.suite)


def _print_summary(corpus_root: Path, suite: str):
    """Print final summary from run_log."""
    run_log = load_run_log(corpus_root)
    suite_entries = [e for e in run_log.values() if e.get("suite") == suite]

    ok = [e for e in suite_entries if e["status"] == "ok"]
    skipped = [e for e in suite_entries if e["status"] == "skipped"]
    failed = [e for e in suite_entries if e["status"] == "failed"]

    total_regions = sum(e.get("region_count", 0) for e in ok)
    total_time = sum(e.get("wall_time_s", 0) for e in suite_entries)

    print(f"\n{'='*70}")
    print(f"  SUMMARY ({suite})")
    print(f"  OK: {len(ok)} | Skipped: {len(skipped)} | Failed: {len(failed)}")
    print(f"  Total regions: {total_regions}")
    print(f"  Total wall time: {total_time:.0f}s")
    if failed:
        print(f"  Failed models:")
        for e in failed[:10]:
            print(f"    {e['model_key']}: {(e.get('error_tail') or '')[:80]}")
        if len(failed) > 10:
            print(f"    ... and {len(failed) - 10} more")
    print(f"{'='*70}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Wave-1 mass recapture script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--suite", required=True,
                        choices=["timm", "hf", "torchbench"],
                        help="Model suite to capture")
    parser.add_argument("--mode", required=True,
                        choices=["infer", "train", "both"],
                        help="Capture mode(s)")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model name(s) or 'all' for full suite list")
    parser.add_argument("--corpus-root", required=True, type=str,
                        help="Root directory for captured corpus (NEVER defaults to repros/)")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs for workers (e.g. '0,1')")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-model timeout in seconds (default: 300)")
    parser.add_argument("--force", action="store_true",
                        help="Force recapture even if model is already status=ok in run_log")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would run without actually capturing")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size for the model (worker mode)")

    # Internal: subprocess worker mode
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Safety: corpus-root must never be the existing repros/ directory
    corpus_root = Path(args.corpus_root).resolve()
    project_repros = (PROJECT_ROOT / "repros").resolve()
    if corpus_root == project_repros:
        print("ERROR: --corpus-root must NOT be the existing repros/ directory.")
        print("       Use a new directory for recapture (e.g. /tmp/wave1_corpus/repros).")
        sys.exit(1)

    if args._worker:
        # Worker mode: capture a single model in-process, print result as JSON
        assert len(args.models) == 1, "Worker mode expects exactly one model"
        model_name = args.models[0]
        mode = args.mode if args.mode != "both" else "infer"
        suite_models = get_suite_models(args.suite)
        bs = args.batch_size or suite_models.get(model_name, 128)

        entry = capture_single_model(
            suite=args.suite,
            model_name=model_name,
            mode=mode,
            batch_size=bs,
            corpus_root=corpus_root,
            timeout=args.timeout,
        )
        # Write to run_log from worker too (atomic)
        write_run_log_entry(corpus_root, entry)
        # Print result for the orchestrator to parse
        print(f"RESULT_JSON:{json.dumps(entry)}")
        return

    # Orchestrator mode
    run_orchestrator(args)


if __name__ == "__main__":
    main()
