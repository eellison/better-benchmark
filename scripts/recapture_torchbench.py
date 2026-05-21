"""
Recapture TorchBench models with the fixed capture hook.

Supports both CI-list models (17) and --all mode for all loadable models (100+).
Directly imports torchbenchmark models for maximum reliability.
"""
import gc
import importlib
import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Torchbench needs its directory on the path
TORCHBENCH_DIR = "/tmp/pytorch-work/torchbenchmark"
sys.path.insert(0, TORCHBENCH_DIR)

import torch
import torch._dynamo
from torch._dynamo.testing import reduce_to_scalar_loss
from torch._inductor.utils import fresh_inductor_cache
from capture_hook import install_capture_hook, uninstall_capture_hook
from merge_captures import merge_one_capture

# Models that are known to be problematic (require FB-internal data, hang, or OOM)
SKIP_MODELS = {
    "Background_Matting",  # requires FB-internal manifold download
    "llama_v2_7b_16h",    # too large for single GPU capture
    "simple_gpt_tp_manual",  # requires multi-GPU tensor parallel
    "mobilenet_v2_quantized_qat",  # quantization not supported in dynamo capture
    "resnet50_quantized_qat",  # quantization not supported in dynamo capture
    "microbench_unbacked_tolist_sum",  # microbenchmark, not a real model
    "cm3leon_generate",  # generate-only model, not useful for fusion benchmarking
    "hf_T5_generate",  # generate-only model
    "sam",  # very large, often OOMs
    "timm_vision_transformer_large",  # very large, often OOMs
    "stable_diffusion_unet",  # very large, often OOMs during compilation
}

# Per-model timeout in seconds (compilation can hang)
MODEL_TIMEOUT = 120

OUTPUT_DIR = Path("/tmp/scratch_space/better_benchmark/repros")
TORCHBENCH_LIST = Path("/tmp/pytorch-work/benchmarks/dynamo/torchbench_models_list.txt")


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Model capture timed out")


def load_torchbench_ci_models() -> dict[str, int]:
    """Load model names and batch sizes from torchbench_models_list.txt (CI subset)."""
    sizes = {}
    for line in TORCHBENCH_LIST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) == 2:
            name, bs = parts[0].strip(), int(parts[1].strip())
            if name not in SKIP_MODELS:
                sizes[name] = bs
    return sizes


def discover_all_torchbench_models() -> list[str]:
    """Discover all loadable models from the torchbench install directory."""
    models_dir = Path(TORCHBENCH_DIR) / "torchbenchmark" / "models"
    available = sorted(
        d.name for d in models_dir.iterdir()
        if d.is_dir() and (d / "__init__.py").exists()
    )
    # Filter out skip list
    return [m for m in available if m not in SKIP_MODELS]


def capture_torchbench_model(model_name: str, batch_size: int | None, mode: str, timeout: int = MODEL_TIMEOUT):
    """Capture one TorchBench model. Returns (n_regions, time_s).

    Args:
        model_name: Name of the torchbench model
        batch_size: Batch size to use, or None to use model default
        mode: "infer" or "train"
        timeout: Per-model timeout in seconds
    """
    label = f"torchbench_{model_name}_{mode}"
    cap_dir = Path(tempfile.mkdtemp())

    torch._dynamo.reset()
    torch.cuda.empty_cache()
    gc.collect()

    model = None
    example_inputs = None

    # Set up timeout (only on Unix)
    old_handler = None
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

    try:
        # Change to torchbench directory (some models need this)
        original_dir = os.getcwd()
        os.chdir(TORCHBENCH_DIR)

        # Import the model module directly
        module = importlib.import_module(f"torchbenchmark.models.{model_name}")
        benchmark_cls = module.Model

        # Create benchmark instance
        test_mode = "train" if mode == "train" else "eval"
        kwargs = {"test": test_mode, "device": "cuda"}

        # Only pass batch_size if we have one and the model allows it
        if batch_size is not None:
            allow_custom = getattr(benchmark_cls, "ALLOW_CUSTOMIZE_BSIZE", True)
            if allow_custom:
                kwargs["batch_size"] = batch_size

        benchmark = benchmark_cls(**kwargs)
        model, example_inputs = benchmark.get_module()

        if mode == "train":
            model.train()
        else:
            model.eval()

        # Set up capture directories
        model_dir = OUTPUT_DIR / "models" / "torchbench" / mode / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        # Skip eager validation during bulk capture - it can trigger CUDA asserts
        # that poison the GPU context and kill the entire process
        install_capture_hook(str(cap_dir), label=label, graph_dir=str(model_dir), validate=False)

        t0 = time.time()
        with fresh_inductor_cache():
            compiled = torch.compile(model)
            if mode == "train":
                if isinstance(example_inputs, dict):
                    pred = compiled(**example_inputs)
                elif isinstance(example_inputs, (list, tuple)):
                    pred = compiled(*example_inputs)
                else:
                    pred = compiled(example_inputs)
                loss = reduce_to_scalar_loss(pred)
                loss.backward()
            else:
                with torch.no_grad():
                    if isinstance(example_inputs, dict):
                        compiled(**example_inputs)
                    elif isinstance(example_inputs, (list, tuple)):
                        compiled(*example_inputs)
                    else:
                        compiled(example_inputs)
            torch.cuda.synchronize()
        elapsed = time.time() - t0

        uninstall_capture_hook()

        # Merge into canonical set
        n = merge_one_capture(cap_dir, OUTPUT_DIR, model_name, suite="torchbench", mode=mode)

        os.chdir(original_dir)
        return n, elapsed

    except TimeoutError:
        print(f"  TIMEOUT after {timeout}s")
        try:
            uninstall_capture_hook()
        except Exception:
            pass
        try:
            os.chdir(original_dir)
        except Exception:
            pass
        return 0, 0.0
    except Exception as e:
        import traceback
        print(f"  FAILED: {e}")
        traceback.print_exc()
        try:
            uninstall_capture_hook()
        except Exception:
            pass
        try:
            os.chdir(original_dir)
        except Exception:
            pass
        return 0, 0.0
    finally:
        # Cancel alarm
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        del model, example_inputs
        torch.cuda.empty_cache()
        gc.collect()


def capture_one_subprocess(model_name: str, batch_size: int | None, mode: str, timeout: int = MODEL_TIMEOUT):
    """Run capture for a single model in a subprocess for GPU isolation.

    Returns (n_regions, time_s) parsed from subprocess output.
    """
    import subprocess

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")

    # Build the command to run a single model
    cmd = [
        sys.executable, __file__,
        "--models", model_name,
        "--mode", mode,
        "--timeout", str(timeout),
        "--_single",  # internal flag: run in-process (already in subprocess)
    ]
    if batch_size is not None:
        cmd += ["--batch-size", str(batch_size)]

    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=timeout + 30,  # extra 30s for subprocess overhead
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        # Parse regions from output - look for "RESULT: <n_regions> <time>"
        for line in result.stdout.splitlines():
            if line.startswith("RESULT:"):
                parts = line.split()
                return int(parts[1]), float(parts[2])
        # If no RESULT line, check for error
        if result.returncode != 0:
            # Print last few lines of stderr for debugging
            stderr_lines = result.stderr.strip().splitlines()
            for line in stderr_lines[-5:]:
                if "CUDA" not in line and "Assertion" not in line:
                    print(f"    {line}")
            stdout_lines = result.stdout.strip().splitlines()
            for line in stdout_lines[-3:]:
                print(f"    {line}")
        return 0, 0.0
    except subprocess.TimeoutExpired:
        print(f"  SUBPROCESS TIMEOUT after {timeout + 30}s")
        return 0, 0.0
    except Exception as e:
        print(f"  SUBPROCESS ERROR: {e}")
        return 0, 0.0


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Capture FX graphs from TorchBench models"
    )
    parser.add_argument("--models", nargs="*", default=None,
                        help="Specific models to capture (default: all from CI list)")
    parser.add_argument("--all", action="store_true",
                        help="Capture ALL loadable torchbench models (not just CI list)")
    parser.add_argument("--mode", choices=["infer", "train", "both"], default="both",
                        help="Which mode(s) to capture")
    parser.add_argument("--start-from", type=str, default=None,
                        help="Start from this model (skip earlier ones)")
    parser.add_argument("--slice", type=str, default=None,
                        help="Slice of model list to capture, e.g. '0:35' or '35:70'")
    parser.add_argument("--timeout", type=int, default=MODEL_TIMEOUT,
                        help=f"Per-model timeout in seconds (default: {MODEL_TIMEOUT})")
    parser.add_argument("--subprocess", action="store_true", default=True,
                        help="Run each model in a subprocess for GPU isolation (default)")
    parser.add_argument("--no-subprocess", action="store_true",
                        help="Run all models in-process (faster but CUDA errors can propagate)")
    parser.add_argument("--_single", action="store_true",
                        help=argparse.SUPPRESS)  # Internal: single-model in-process mode
    parser.add_argument("--batch-size", type=int, default=None,
                        help=argparse.SUPPRESS)  # Internal: batch size for single model
    args = parser.parse_args()

    # Internal single-model mode (called from subprocess)
    if args._single:
        if not args.models or len(args.models) != 1:
            print("RESULT: 0 0.0")
            return
        model_name = args.models[0]
        mode = args.mode if args.mode != "both" else "infer"
        batch_size = args.batch_size
        n, elapsed = capture_torchbench_model(model_name, batch_size, mode, timeout=args.timeout)
        print(f"RESULT: {n} {elapsed:.2f}")
        return

    use_subprocess = not args.no_subprocess

    # Load CI batch sizes (used when available)
    ci_models = load_torchbench_ci_models()

    # Determine model list
    if args.models:
        models = args.models
    elif getattr(args, 'all'):
        models = discover_all_torchbench_models()
        print(f"Discovered {len(models)} loadable torchbench models")
    else:
        models = list(ci_models.keys())

    if args.start_from:
        try:
            idx = models.index(args.start_from)
            models = models[idx:]
        except ValueError:
            print(f"Warning: --start-from model '{args.start_from}' not found in list")

    if args.slice:
        parts = args.slice.split(":")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else len(models)
        models = models[start:end]
        print(f"Sliced to models[{start}:{end}] ({len(models)} models)")

    modes = ["infer", "train"] if args.mode == "both" else [args.mode]

    total_regions = 0
    total_time = 0
    results = []
    failures = []
    successes = []

    for i, model_name in enumerate(models):
        # Use CI batch size if available, otherwise None (model default)
        batch_size = ci_models.get(model_name, None)
        for mode in modes:
            label = f"torchbench_{model_name}_{mode}"
            bs_str = f"batch={batch_size}" if batch_size else "batch=default"
            print(f"\n[{i+1}/{len(models)}] {'='*54}")
            print(f"  {label} ({bs_str})")
            print(f"{'='*60}")

            if use_subprocess:
                n, elapsed = capture_one_subprocess(model_name, batch_size, mode, timeout=args.timeout)
            else:
                n, elapsed = capture_torchbench_model(model_name, batch_size, mode, timeout=args.timeout)
            total_regions += n
            total_time += elapsed
            results.append({"model": model_name, "mode": mode, "regions": n, "time": elapsed})
            if n == 0:
                failures.append(f"{model_name}/{mode}")
            else:
                successes.append(f"{model_name}/{mode}")
            print(f"  => {n} regions in {elapsed:.1f}s")

    print(f"\n{'='*60}")
    print(f"DONE: {total_regions} total regions from {len(results)} runs in {total_time:.0f}s")
    print(f"SUCCEEDED: {len(successes)}/{len(results)}")
    if failures:
        print(f"FAILURES ({len(failures)}): {', '.join(failures)}")
    print(f"{'='*60}")

    summary_path = OUTPUT_DIR / "capture_summary_torchbench.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
