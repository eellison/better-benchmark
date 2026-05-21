"""
Recapture HuggingFace CI models with the fixed capture hook.

Uses the PyTorch benchmark runner to load models at correct batch sizes.
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path("/tmp/pytorch-work/benchmarks/dynamo")))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch._dynamo
from torch._inductor.utils import fresh_inductor_cache
from capture_hook import install_capture_hook, uninstall_capture_hook
from merge_captures import temporary_capture_for_merge

# Models that need HF auth or special downloads — skip
SKIP_MODELS = {
    "meta-llama/Llama-3.2-1B", "google/gemma-2-2b", "google/gemma-3-4b-it",
    "openai/whisper-tiny", "Qwen/Qwen3-0.6B", "mistralai/Mistral-7B-Instruct-v0.3",
    "openai/gpt-oss-20b",
}

OUTPUT_DIR = Path("/tmp/scratch_space/better_benchmark/repros")
HF_LIST = Path("/tmp/pytorch-work/benchmarks/dynamo/huggingface_models_list.txt")


def load_hf_models() -> dict[str, int]:
    sizes = {}
    for line in HF_LIST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) == 2:
            name, bs = parts[0].strip(), int(parts[1].strip())
            if name not in SKIP_MODELS:
                sizes[name] = bs
    return sizes


def capture_hf_model(model_name: str, batch_size: int, mode: str):
    """Capture one HF model. Returns (n_regions, time_s)."""
    import common
    from huggingface import HuggingfaceRunner

    label = f"hf_{model_name}_{mode}"

    torch._dynamo.reset()
    torch.cuda.empty_cache()
    gc.collect()

    with temporary_capture_for_merge(
        OUTPUT_DIR,
        label,
        suite="hf",
        mode=mode,
        prefix="recapture_hf_",
    ) as capture:
        cap_dir = capture.capture_dir
        try:
            sys.argv = ["capture", "--performance",
                        "--training" if mode == "train" else "--inference",
                        "--inductor", "--devices", "cuda",
                        "--batch-size", str(batch_size), "--only", model_name]
            runner = HuggingfaceRunner()
            args = common.parse_args(sys.argv[1:])
            runner.args = args
            runner.model_iter_fn = runner.forward_and_backward_pass if mode == "train" else runner.forward_pass

            _, _, model, inputs_dict, _ = runner.load_model("cuda", model_name, batch_size)

            if mode == "train":
                model.train()
            else:
                model.eval()

            model_dir = OUTPUT_DIR / "models" / "hf" / mode / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            install_capture_hook(str(cap_dir), label=label, graph_dir=str(model_dir))

            t0 = time.time()
            with fresh_inductor_cache():
                compiled = torch.compile(model)
                if mode == "train":
                    out = compiled(**inputs_dict)
                    if hasattr(out, "loss") and out.loss is not None:
                        out.loss.backward()
                    elif isinstance(out, torch.Tensor):
                        out.sum().backward()
                else:
                    with torch.no_grad():
                        compiled(**inputs_dict)
                torch.cuda.synchronize()
            elapsed = time.time() - t0

            uninstall_capture_hook()

            n = capture.merge()
            return n, elapsed

        except Exception as e:
            print(f"  FAILED: {e}")
            try:
                uninstall_capture_hook()
            except Exception:
                pass
            return 0, 0.0
        finally:
            torch.cuda.empty_cache()
            gc.collect()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--mode", choices=["infer", "train", "both"], default="both")
    args = parser.parse_args()

    all_models = load_hf_models()
    models = args.models or list(all_models.keys())
    modes = ["infer", "train"] if args.mode == "both" else [args.mode]


    total_regions = 0
    total_time = 0
    results = []

    for model_name in models:
        if model_name not in all_models:
            print(f"Skipping {model_name} (not in CI list)")
            continue
        batch_size = all_models[model_name]
        for mode in modes:
            label = f"hf_{model_name}_{mode}"
            print(f"\n{'='*60}")
            print(f"  {label} (batch={batch_size})")
            print(f"{'='*60}")

            n, elapsed = capture_hf_model(model_name, batch_size, mode)
            total_regions += n
            total_time += elapsed
            results.append({"model": model_name, "mode": mode, "regions": n, "time": elapsed})
            print(f"  => {n} regions in {elapsed:.1f}s")

    print(f"\n{'='*60}")
    print(f"DONE: {total_regions} total regions from {len(results)} model runs in {total_time:.0f}s")
    print(f"{'='*60}")

    summary_path = OUTPUT_DIR.parent / "capture_summary_hf.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
