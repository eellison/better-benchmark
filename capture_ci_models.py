"""
Capture all CI benchmark models at their correct batch sizes.

Reads batch sizes from:
  - /tmp/pytorch-work/benchmarks/dynamo/huggingface_models_list.txt
  - /tmp/pytorch-work/benchmarks/dynamo/timm_models_list.txt

Produces correct shapes.json entries with proper S() shape params.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch._dynamo
from torch._inductor.utils import fresh_inductor_cache
from capture_hook import install_capture_hook, uninstall_capture_hook
from merge_captures import merge_one_capture


PYTORCH_DIR = Path("/tmp/pytorch-work")
HF_MODELS_FILE = PYTORCH_DIR / "benchmarks/dynamo/huggingface_models_list.txt"
TIMM_MODELS_FILE = PYTORCH_DIR / "benchmarks/dynamo/timm_models_list.txt"
OUTPUT_DIR = Path("/tmp/scratch_space/better_benchmark/repros")


def load_hf_batch_sizes() -> dict[str, int]:
    sizes = {}
    for line in HF_MODELS_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) == 2:
            name, bs = parts[0].strip(), int(parts[1].strip())
            sizes[name] = bs
    return sizes


def load_timm_batch_sizes() -> dict[str, int]:
    sizes = {}
    for line in TIMM_MODELS_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 2:
            name, bs = parts[0], int(parts[1])
            sizes[name] = bs
    return sizes


def capture_one(model, inputs, label: str, suite: str, mode: str):
    """Capture one model run and merge into canonical set."""
    tmpdir = tempfile.mkdtemp()
    torch._dynamo.reset()
    install_capture_hook(tmpdir, label=label)
    try:
        with fresh_inductor_cache():
            compiled = torch.compile(model)
            if mode == "train":
                out = compiled(*inputs)
                if isinstance(out, torch.Tensor):
                    out.sum().backward()
                elif isinstance(out, (tuple, list)):
                    loss = sum(o.sum() for o in out if isinstance(o, torch.Tensor))
                    loss.backward()
            else:
                with torch.no_grad():
                    compiled(*inputs)
            torch.cuda.synchronize()
    except Exception as e:
        print(f"    COMPILE FAILED: {e}")
        uninstall_capture_hook()
        return 0
    uninstall_capture_hook()

    index_path = Path(tmpdir) / "index.json"
    if not index_path.exists():
        return 0

    n = merge_one_capture(Path(tmpdir), OUTPUT_DIR, label, suite=suite, mode=mode)
    return n


def capture_hf_models():
    """Capture HuggingFace models at CI batch sizes."""
    batch_sizes = load_hf_batch_sizes()
    print(f"\n=== HuggingFace Models ({len(batch_sizes)}) ===\n")

    # Models we can load without special setup
    # (skip LLMs that need HF downloads: llama, gemma, whisper, qwen, mistral, gpt-oss)
    skip = {"meta-llama/Llama-3.2-1B", "google/gemma-2-2b", "google/gemma-3-4b-it",
            "openai/whisper-tiny", "Qwen/Qwen3-0.6B", "mistralai/Mistral-7B-Instruct-v0.3",
            "openai/gpt-oss-20b"}

    sys.path.insert(0, str(PYTORCH_DIR / "benchmarks/dynamo"))

    for model_name, batch_size in batch_sizes.items():
        if model_name in skip:
            continue

        print(f"  {model_name} (batch={batch_size})...", end=" ", flush=True)

        try:
            from huggingface import HuggingfaceRunner
            runner = HuggingfaceRunner()
            # Use runner to get model + inputs at correct batch size
            # This handles seq_length, model loading, etc.
        except Exception as e:
            print(f"SKIP (runner failed: {e})")
            continue

        # For now just print what we'd capture
        print(f"TODO - need to wire up runner.load_model()")


def capture_timm_models():
    """Capture timm models at CI batch sizes (channels-last)."""
    batch_sizes = load_timm_batch_sizes()
    print(f"\n=== timm Models ({len(batch_sizes)}) ===\n")

    import timm

    for model_name, batch_size in batch_sizes.items():
        for mode in ["infer", "train"]:
            label = f"timm_{model_name}_{mode}"
            print(f"  {label} (batch={batch_size})...", end=" ", flush=True)

            try:
                model = timm.create_model(model_name, pretrained=False).cuda()
                model = model.to(memory_format=torch.channels_last)

                # Get input size from model config
                data_config = timm.data.resolve_model_data_config(model)
                input_size = data_config.get("input_size", (3, 224, 224))
                inp = torch.randn(batch_size, *input_size, device="cuda")
                inp = inp.to(memory_format=torch.channels_last)

                if mode == "train":
                    model.train()
                else:
                    model.eval()

                n = capture_one(model, (inp,), label, suite="timm", mode=mode)
                print(f"OK ({n} regions)")

            except Exception as e:
                print(f"FAIL ({e})")

            torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["hf", "timm", "all"], default="all")
    args = parser.parse_args()

    if args.suite in ("timm", "all"):
        capture_timm_models()
    if args.suite in ("hf", "all"):
        capture_hf_models()
