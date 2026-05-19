"""
Recapture TorchBench CI models with the fixed capture hook.

Uses the PyTorch benchmark runner (TorchBenchmarkRunner) to load models
at correct batch sizes with proper setup.
"""
import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path("/tmp/pytorch-work/benchmarks/dynamo")))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch._dynamo
from torch._dynamo.testing import reduce_to_scalar_loss
from torch._inductor.utils import fresh_inductor_cache
from capture_hook import install_capture_hook, uninstall_capture_hook
from merge_captures import merge_one_capture

# Models that are known to be problematic (OOM, hang, or need special env)
SKIP_MODELS = {
    "tacotron2",  # OOM at CI batch sizes
    "phlippe_resnet",  # intermittent failures
}

OUTPUT_DIR = Path("/tmp/scratch_space/better_benchmark/repros")
GRAPH_DIR = Path("/tmp/scratch_space/better_benchmark/fx_graphs")
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
    import common
    from torchbench import TorchBenchmarkRunner, setup_torchbench_cwd

    label = f"torchbench_{model_name}_{mode}"
    cap_dir = Path(tempfile.mkdtemp())

    torch._dynamo.reset()
    torch.cuda.empty_cache()
    gc.collect()

    original_dir = None
    try:
        # TorchBench needs cwd set to the benchmark repo
        original_dir = setup_torchbench_cwd()

        sys.argv = ["capture", "--performance",
                    "--training" if mode == "train" else "--inference",
                    "--inductor", "--devices", "cuda",
                    "--batch-size", str(batch_size), "--only", model_name]
        runner = TorchBenchmarkRunner()
        args = common.parse_args(sys.argv[1:])
        runner.args = args
        runner.model_iter_fn = (
            runner.forward_and_backward_pass if mode == "train"
            else runner.forward_pass
        )

        _, _, model, example_inputs, batch_size = runner.load_model(
            "cuda", model_name, batch_size
        )

        if mode == "train":
            model.train()
        else:
            model.eval()

        install_capture_hook(str(cap_dir), label=label, graph_dir=str(GRAPH_DIR / label))

        t0 = time.time()
        with fresh_inductor_cache():
            compiled = torch.compile(model)
            if mode == "train":
                if isinstance(example_inputs, dict):
                    pred = compiled(**example_inputs)
                else:
                    pred = compiled(*example_inputs)
                loss = reduce_to_scalar_loss(pred)
                loss.backward()
            else:
                with torch.no_grad():
                    if isinstance(example_inputs, dict):
                        compiled(**example_inputs)
                    else:
                        compiled(*example_inputs)
            torch.cuda.synchronize()
        elapsed = time.time() - t0

        uninstall_capture_hook()

        n = merge_one_capture(cap_dir, OUTPUT_DIR, label, suite="torchbench", mode=mode)
        return n, elapsed

    except Exception as e:
        print(f"  FAILED: {e}")
        try:
            uninstall_capture_hook()
        except Exception:
            pass
        return 0, 0.0
    finally:
        if original_dir is not None:
            os.chdir(original_dir)
        torch.cuda.empty_cache()
        gc.collect()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Capture FX graphs from TorchBench models"
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

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    total_regions = 0
    total_time = 0
    results = []

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
            print(f"  => {n} regions in {elapsed:.1f}s")

    print(f"\n{'='*60}")
    print(f"DONE: {total_regions} total regions from {len(results)} model runs in {total_time:.0f}s")
    print(f"{'='*60}")

    summary_path = OUTPUT_DIR / "capture_summary_torchbench.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
