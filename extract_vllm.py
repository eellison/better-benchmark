"""Extract fusion regions from vLLM-relevant models.

Supports dense and MoE architectures. For very large models (DeepSeek, etc.),
reduces num_hidden_layers to fit on a single GPU while preserving kernel patterns.
"""
import os
import sys
import torch
import torch._inductor.config as inductor_config

sys.path.insert(0, "/tmp/scratch_space/better_benchmark")
PYTHONPATH = os.environ.get("PYTORCH_DIR", "/tmp/pytorch-work")
sys.path.insert(0, PYTHONPATH)

inductor_config.force_disable_caches = True
inductor_config.split_reductions = False

REPRO_DIR = "/tmp/scratch_space/better_benchmark/repros"


def extract_from_model(model_name, output_name=None, device_id=0,
                       inference_only=False, max_layers=None):
    from extract_reductions import run_aten_extraction
    from merge_captures import merge_one_capture
    from pathlib import Path
    from transformers import AutoConfig, AutoModelForCausalLM

    if output_name is None:
        safe_name = model_name.replace("/", "_")
        suffix = "_inference" if inference_only else ""
        output_name = f"vllm_{safe_name}{suffix}"

    output_dir = os.path.join("/tmp/scratch_space/better_benchmark/output", "aten_repros", output_name)
    device = f"cuda:{device_id}"

    config = AutoConfig.from_pretrained(model_name)

    # For nested configs (Llama-4 multimodal)
    text_config = getattr(config, "text_config", config)

    if hasattr(text_config, "use_cache"):
        text_config.use_cache = False

    # Shrink very deep models to fit on one GPU
    if max_layers and hasattr(text_config, "num_hidden_layers"):
        orig = text_config.num_hidden_layers
        if orig > max_layers:
            text_config.num_hidden_layers = max_layers
            print(f"Reduced {model_name} from {orig} to {max_layers} layers")

    batch_size = 4
    seq_len = min(getattr(text_config, "max_position_embeddings", 512), 512)
    vocab_size = getattr(text_config, "vocab_size", 32000)

    def make_model():
        m = AutoModelForCausalLM.from_config(config)
        if inference_only:
            return m.to(device).eval()
        return m.to(device).train()

    def make_args():
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        if inference_only:
            return [{"input_ids": input_ids}]
        labels = input_ids.clone()
        return [{"input_ids": input_ids, "labels": labels}]

    extractor = run_aten_extraction(make_model, make_args, output_dir,
                        model_name=output_name, inference_only=inference_only)

    # Merge into canonical repro set
    n = merge_one_capture(Path(output_dir), Path(REPRO_DIR), output_name)
    print(f"  Merged {n} regions into {REPRO_DIR}/canonical/")
    return n


VLLM_MODELS = [
    "Qwen/Qwen3-0.6B",
    "meta-llama/Llama-3.2-1B",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "openai/gpt-oss-20b",
    "facebook/opt-125m",
    "openai/whisper-tiny",
    # MoE models (use fewer layers to fit)
    "Qwen/Qwen3-30B-A3B",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "deepseek-ai/DeepSeek-V3",
]

# Large models need layer reduction to fit on 1 GPU
MAX_LAYERS = {
    "deepseek-ai/DeepSeek-V3": 4,
    "deepseek-ai/DeepSeek-R1": 4,
    "Qwen/Qwen3-30B-A3B": 4,
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": 4,
    "mistralai/Mistral-7B-Instruct-v0.3": 8,
    "openai/gpt-oss-20b": 4,
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="all",
                        help="HF model name, or 'all' for all vLLM benchmark models")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--list", action="store_true", help="List available models")
    args = parser.parse_args()

    if args.list:
        for m in VLLM_MODELS:
            print(m)
        sys.exit(0)

    if args.model == "all":
        for m in VLLM_MODELS:
            print(f"\n{'='*60}\n  {m}\n{'='*60}")
            ml = args.max_layers or MAX_LAYERS.get(m)
            try:
                extract_from_model(m, device_id=args.device,
                                   inference_only=args.inference_only,
                                   max_layers=ml)
            except Exception as e:
                import traceback
                print(f"SKIP {m}: {e}")
                traceback.print_exc()
            torch._dynamo.reset()
    else:
        ml = args.max_layers or MAX_LAYERS.get(args.model)
        extract_from_model(args.model, device_id=args.device,
                           inference_only=args.inference_only,
                           max_layers=ml)
