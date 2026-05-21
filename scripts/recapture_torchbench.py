"""
Recapture TorchBench CI models with the fixed capture hook.

Directly imports torchbenchmark models for maximum reliability.
All 17 CI models, infer + train modes.
"""
import gc
import importlib
import json
import os
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

# Models that are known to be problematic (require FB-internal data or hang)
SKIP_MODELS = {
    "Background_Matting",  # requires FB-internal manifold download
}

OUTPUT_DIR = Path("/tmp/scratch_space/better_benchmark/repros")
TORCHBENCH_LIST = Path("/tmp/pytorch-work/benchmarks/dynamo/torchbench_models_list.txt")


def load_torchbench_models() -> dict[str, int]:
    """Load model names and batch sizes from torchbench_models_list.txt."""
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


def capture_torchbench_model(model_name: str, batch_size: int, mode: str):
    """Capture one TorchBench model. Returns (n_regions, time_s)."""
    label = f"torchbench_{model_name}_{mode}"
    cap_dir = Path(tempfile.mkdtemp())

    torch._dynamo.reset()
    torch.cuda.empty_cache()
    gc.collect()

    model = None
    example_inputs = None

    try:
        # Change to torchbench directory (some models need this)
        original_dir = os.getcwd()
        os.chdir(TORCHBENCH_DIR)

        # Import the model module directly
        module = importlib.import_module(f"torchbenchmark.models.{model_name}")
        benchmark_cls = module.Model

        # Create benchmark instance
        test_mode = "train" if mode == "train" else "eval"
        # Some models don't allow custom batch sizes
        allow_custom = getattr(benchmark_cls, "ALLOW_CUSTOMIZE_BSIZE", True)
        kwargs = {"test": test_mode, "device": "cuda"}
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
        install_capture_hook(str(cap_dir), label=label, graph_dir=str(model_dir))

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
        del model, example_inputs
        torch.cuda.empty_cache()
        gc.collect()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Capture FX graphs from TorchBench CI models"
    )
    parser.add_argument("--models", nargs="*", default=None,
                        help="Specific models to capture (default: all from CI list)")
    parser.add_argument("--mode", choices=["infer", "train", "both"], default="both",
                        help="Which mode(s) to capture")
    parser.add_argument("--start-from", type=str, default=None,
                        help="Start from this model (skip earlier ones)")
    args = parser.parse_args()

    all_models = load_torchbench_models()
    models = args.models or list(all_models.keys())

    if args.start_from:
        try:
            idx = models.index(args.start_from)
            models = models[idx:]
        except ValueError:
            print(f"Warning: --start-from model '{args.start_from}' not found in list")

    modes = ["infer", "train"] if args.mode == "both" else [args.mode]

    total_regions = 0
    total_time = 0
    results = []
    failures = []

    for model_name in models:
        if model_name not in all_models:
            print(f"Skipping {model_name} (not in CI list)")
            continue
        batch_size = all_models[model_name]
        for mode in modes:
            label = f"torchbench_{model_name}_{mode}"
            print(f"\n{'='*60}")
            print(f"  {label} (batch={batch_size})")
            print(f"{'='*60}")

            n, elapsed = capture_torchbench_model(model_name, batch_size, mode)
            total_regions += n
            total_time += elapsed
            results.append({"model": model_name, "mode": mode, "regions": n, "time": elapsed})
            if n == 0:
                failures.append(f"{model_name}/{mode}")
            print(f"  => {n} regions in {elapsed:.1f}s")

    print(f"\n{'='*60}")
    print(f"DONE: {total_regions} total regions from {len(results)} runs in {total_time:.0f}s")
    if failures:
        print(f"FAILURES ({len(failures)}): {', '.join(failures)}")
    print(f"{'='*60}")

    summary_path = OUTPUT_DIR / "capture_summary_torchbench.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
