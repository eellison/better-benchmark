"""
Capture kernels from ALL CI perf benchmark models (HF + timm).
Skips torchbench (not installed) and HF LLM models (need large weight downloads).

For each model:
  1. Load model and inputs
  2. Install capture hook
  3. Run torch.compile forward pass (inference)
  4. Run torch.compile forward+backward pass (training)
  5. Merge captures into canonical set
"""
import gc
import os
import sys
import tempfile
import traceback
import time
import warnings
from pathlib import Path

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)
warnings.filterwarnings("ignore")

# Setup environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Add paths
sys.path.insert(0, "/tmp/scratch_space/better_benchmark")
sys.path.insert(0, "/tmp/pytorch-work")
sys.path.insert(0, "/tmp/pytorch-work/benchmarks/dynamo")

import logging
logging.disable(logging.WARNING)

import torch
import torch._dynamo
import torch._inductor.utils

from merge_captures import merge_one_capture

# ============================================================================
# HuggingFace models (non-LLM, from config)
# ============================================================================

HF_MODELS_FILE = "/tmp/pytorch-work/benchmarks/dynamo/huggingface_models_list.txt"
HF_EXTRA_MODELS = [
    "AllenaiLongformerBase",
    "Reformer",
    "T5Small",
    "DistillGPT2",
    "GoogleFnet",
    "YituTechConvBert",
]

# HF LLM models that require downloading large pretrained weights - skip these
HF_LLM_MODELS = {
    "meta-llama/Llama-3.2-1B",
    "google/gemma-2-2b",
    "google/gemma-3-4b-it",
    "openai/whisper-tiny",
    "Qwen/Qwen3-0.6B",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "openai/gpt-oss-20b",
}


def get_hf_model_list():
    """Get list of HF models from the models file."""
    models = {}
    with open(HF_MODELS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            model_name = parts[0].strip()
            batch_size = int(parts[1].strip())
            if model_name in HF_LLM_MODELS:
                continue
            models[model_name] = batch_size
    # Add extra models
    for m in HF_EXTRA_MODELS:
        if m not in models:
            models[m] = 8
    return models


def load_hf_model(model_name, batch_size, device, for_training=False):
    """Load a HuggingFace model and generate inputs."""
    import importlib
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoModelForSeq2SeqLM,
        ReformerConfig,
    )

    EXTRA_MODELS = {
        "AllenaiLongformerBase": (
            AutoConfig.from_pretrained("allenai/longformer-base-4096"),
            AutoModelForMaskedLM,
        ),
        "Reformer": (
            ReformerConfig(),
            AutoModelForMaskedLM,
        ),
        "T5Small": (
            AutoConfig.from_pretrained("t5-small"),
            AutoModelForSeq2SeqLM,
        ),
        "DistillGPT2": (
            AutoConfig.from_pretrained("distilgpt2"),
            AutoModelForCausalLM,
        ),
        "GoogleFnet": (
            AutoConfig.from_pretrained("google/fnet-base"),
            AutoModelForMaskedLM,
        ),
        "YituTechConvBert": (
            AutoConfig.from_pretrained("YituTech/conv-bert-base"),
            AutoModelForMaskedLM,
        ),
    }

    def get_module_cls_by_model_name(model_cls_name):
        _module_by_model_name = {
            "Speech2Text2Decoder": "transformers.models.speech_to_text_2.modeling_speech_to_text_2",
            "TrOCRDecoder": "transformers.models.trocr.modeling_trocr",
        }
        module_name = _module_by_model_name.get(model_cls_name, "transformers")
        module = importlib.import_module(module_name)
        return getattr(module, model_cls_name)

    if model_name in EXTRA_MODELS:
        config, model_cls = EXTRA_MODELS[model_name]
    else:
        model_cls = get_module_cls_by_model_name(model_name)
        config_cls = model_cls.config_class
        config = config_cls()
        # Set pad_token_id for certain models
        from transformers import (
            GPT2ForSequenceClassification,
            GPTNeoForSequenceClassification,
            GPTJForSequenceClassification,
        )
        if model_cls in [GPT2ForSequenceClassification, GPTNeoForSequenceClassification, GPTJForSequenceClassification] or \
           model_cls.__name__.startswith("Roberta") or model_cls.__name__.startswith("Marian"):
            config.pad_token_id = 0

    # Create model
    if "auto" in model_cls.__module__:
        model = model_cls.from_config(config)
    else:
        model = model_cls(config)

    model = model.to(device, dtype=torch.float32)

    # Disable kv cache
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if for_training:
        model.train()
    else:
        model.eval()

    # Generate inputs
    seq_length = _get_sequence_length(model_cls, model_name)
    vocab_size = model.config.vocab_size

    if model_name.startswith("Wav2Vec2"):
        inputs = {
            "input_values": torch.randn((batch_size, seq_length), device=device),
            "attention_mask": torch.randint(0, 2, (batch_size, seq_length), device=device, dtype=torch.int64),
        }
        if for_training:
            inputs["labels"] = torch.randint(0, vocab_size, (batch_size, 100), device=device, dtype=torch.int64)
    else:
        if model_name.endswith("MultipleChoice"):
            input_ids = torch.randint(0, vocab_size, (batch_size, 3, seq_length), device=device, dtype=torch.int64)
        elif model_name.startswith("Roberta"):
            input_ids = torch.randint(0, 1, (batch_size, seq_length), device=device, dtype=torch.int64)
        else:
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device, dtype=torch.int64)

        if "Bart" in model_name:
            input_ids[:, -1] = model.config.eos_token_id

        inputs = {"input_ids": input_ids}

        from transformers import (
            BlenderbotModel,
            BlenderbotSmallModel,
            BlenderbotForConditionalGeneration,
            PegasusModel,
            MarianModel,
            MarianMTModel,
        )

        if model_name.startswith(("T5", "M2M100", "MT5")) or model_cls in [
            BlenderbotModel, BlenderbotSmallModel, BlenderbotForConditionalGeneration,
            PegasusModel, MarianModel, MarianMTModel,
        ]:
            inputs["decoder_input_ids"] = input_ids

        if model_name.startswith("Lxmert"):
            visual_feat_dim = model.config.visual_feat_dim
            visual_pos_dim = model.config.visual_pos_dim
            inputs["visual_feats"] = torch.randn(batch_size, 42, visual_feat_dim, device=device)
            inputs["visual_pos"] = torch.randn(batch_size, 42, visual_pos_dim, device=device)

        # Add loss args for training
        if for_training:
            inputs = _add_loss_args(inputs, model_name, model_cls, model, batch_size, seq_length, vocab_size, device)

    return model, inputs


def _get_sequence_length(model_cls, model_name):
    if model_name.startswith(("Blenderbot",)):
        return 128
    elif model_name.startswith(("GPT2", "Bart", "T5", "PLBart", "MBart")):
        return 1024
    elif model_name in ("AllenaiLongformerBase", "BigBird"):
        return 1024
    elif model_name.startswith("OPT"):
        return 2048
    elif "Reformer" in model_name:
        return 4096
    elif model_name.startswith(("Albert", "Deberta", "Layout", "Electra", "XLNet", "MegatronBert", "Bert", "Roberta")) or \
         model_name in ("DistillGPT2", "GoogleFnet", "YituTechConvBert"):
        return 512
    elif model_name in ("TrOCRForCausalLM",):
        return 256
    elif model_name.startswith("MobileBert"):
        return 128
    elif model_name.startswith("Wav2Vec2"):
        return 10000
    else:
        return 128


def _add_loss_args(inputs, model_name, model_cls, model, batch_size, seq_length, vocab_size, device):
    from transformers import (
        ElectraForPreTraining,
        LxmertForPreTraining,
        AlbertForPreTraining,
    )

    if model_name.endswith("PreTraining"):
        if model_cls in [ElectraForPreTraining, LxmertForPreTraining]:
            inputs["labels"] = torch.randint(0, 1, (batch_size, seq_length), device=device, dtype=torch.int64)
        else:
            label_name = "sentence_order_label" if model_cls in [AlbertForPreTraining] else "next_sentence_label"
            inputs["labels"] = (torch.randint(0, vocab_size, (batch_size, seq_length), device=device, dtype=torch.int64),)
            inputs[label_name] = torch.randint(0, 1, (batch_size,), device=device, dtype=torch.int64)
    elif model_name.endswith("QuestionAnswering"):
        inputs["start_positions"] = torch.randint(0, seq_length, (batch_size,), device=device, dtype=torch.int64)
        inputs["end_positions"] = torch.randint(0, seq_length, (batch_size,), device=device, dtype=torch.int64)
    elif model_name.endswith(("MaskedLM", "HeadModel", "CausalLM", "DoubleHeadsModel")):
        inputs["labels"] = torch.randint(0, vocab_size, (batch_size, seq_length), device=device, dtype=torch.int64)
    elif model_name.endswith("TokenClassification"):
        inputs["labels"] = torch.randint(0, model.config.num_labels - 1, (batch_size, seq_length), device=device, dtype=torch.int64)
    elif model_name.endswith("MultipleChoice"):
        inputs["labels"] = torch.randint(0, 3, (batch_size,), device=device, dtype=torch.int64)
    elif model_name.endswith("SequenceClassification"):
        inputs["labels"] = torch.randint(0, model.config.num_labels - 1, (batch_size,), device=device, dtype=torch.int64)
    elif model_name.endswith("NextSentencePrediction"):
        inputs["labels"] = torch.randint(0, 1, (batch_size,), device=device, dtype=torch.int64)
    elif model_name.endswith("ForConditionalGeneration"):
        inputs["labels"] = torch.randint(0, vocab_size - 1, (batch_size, seq_length), device=device, dtype=torch.int64)
    elif model_name in HF_EXTRA_MODELS:
        inputs["labels"] = torch.randint(0, vocab_size, (batch_size, seq_length), device=device, dtype=torch.int64)
    else:
        # Can't determine loss args, skip training for this model
        return None
    return inputs


# ============================================================================
# TIMM models
# ============================================================================

TIMM_MODELS_FILE = "/tmp/pytorch-work/benchmarks/dynamo/timm_models_list.txt"


def get_timm_model_list():
    """Get list of timm models from the models file."""
    models = {}
    with open(TIMM_MODELS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            model_name = parts[0]
            batch_size = int(parts[1])
            models[model_name] = batch_size
    return models


def load_timm_model(model_name, batch_size, device, for_training=False):
    """Load a timm model and generate inputs."""
    from timm.models import create_model
    from timm.data import resolve_data_config
    import argparse

    model = create_model(
        model_name,
        in_chans=3,
        scriptable=False,
        num_classes=None,
        drop_rate=0.0,
        drop_path_rate=None,
        drop_block_rate=None,
        pretrained=False,  # Don't download weights
    )
    if model is None:
        raise RuntimeError(f"Failed to create model '{model_name}'")

    model = model.to(device)

    # Get input size from data config
    args = argparse.Namespace(img_size=None, input_size=None, mean=None, std=None, crop_pct=None, interpolation=None)
    data_config = resolve_data_config(vars(args), model=model, use_test_size=not for_training)
    input_size = data_config["input_size"]

    # Reduce batch size if needed (avoid OOM)
    if batch_size > 64:
        batch_size = 64

    torch.manual_seed(1337)
    input_tensor = torch.randint(256, size=(batch_size,) + input_size, device=device).to(dtype=torch.float32)
    mean = torch.mean(input_tensor)
    std_dev = torch.std(input_tensor)
    example_inputs = (input_tensor - mean) / std_dev

    if for_training:
        model.train()
    else:
        model.eval()

    return model, [example_inputs]


# ============================================================================
# Capture logic
# ============================================================================

def capture_model(model_name, suite, load_fn, batch_size, device, mode, repros_dir):
    """Capture kernels from a single model in a single mode."""
    from capture_hook import install_capture_hook, uninstall_capture_hook

    is_training = (mode == "training")
    label = f"{suite}_{model_name}_{mode}"

    # Create temp directory for captures
    tmpdir = tempfile.mkdtemp(prefix=f"capture_{label}_")

    try:
        # Reset state
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        gc.collect()

        # Load model
        with torch._inductor.utils.fresh_inductor_cache():
            model, inputs = load_fn(model_name, batch_size, device, for_training=is_training)

            if inputs is None:
                print(f"  [{label}] Skipped (no loss args for training)")
                return 0

            # Install capture hook
            install_capture_hook(output_dir=tmpdir, label=label)

            try:
                if is_training:
                    with torch.enable_grad():
                        compiled = torch.compile(model)
                        if isinstance(inputs, dict):
                            out = compiled(**inputs)
                        else:
                            out = compiled(*inputs)

                        # Compute loss and backward
                        if isinstance(out, dict):
                            loss = out.get("loss", None)
                            if loss is None:
                                # Try to get logits and compute loss
                                logits = out.get("logits", None)
                                if logits is not None:
                                    loss = logits.sum()
                                else:
                                    loss = sum(v.sum() for v in out.values() if isinstance(v, torch.Tensor) and v.requires_grad)
                        elif isinstance(out, (tuple, list)):
                            loss = out[0]
                            if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                                loss = loss.sum()
                        elif isinstance(out, torch.Tensor):
                            loss = out.sum()
                        else:
                            loss = None

                        if loss is not None and isinstance(loss, torch.Tensor) and loss.requires_grad:
                            loss.backward()
                else:
                    with torch.no_grad():
                        compiled = torch.compile(model)
                        if isinstance(inputs, dict):
                            out = compiled(**inputs)
                        else:
                            out = compiled(*inputs)

                torch.cuda.synchronize()

            finally:
                uninstall_capture_hook()

            # Merge captures
            merged = merge_one_capture(Path(tmpdir), repros_dir, label)
            return merged

    except torch.cuda.OutOfMemoryError:
        print(f"  [{label}] OOM - skipping")
        torch.cuda.empty_cache()
        gc.collect()
        return 0
    except Exception as e:
        print(f"  [{label}] Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 0
    finally:
        # Cleanup
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        gc.collect()


def main():
    device = "cuda"
    repros_dir = Path("/tmp/scratch_space/better_benchmark/repros")

    # Count existing patterns
    canonical_dir = repros_dir / "canonical"
    initial_patterns = len(list(canonical_dir.iterdir())) if canonical_dir.exists() else 0
    print(f"Initial canonical patterns: {initial_patterns}")

    total_merged = 0
    results = {"success": [], "failed": [], "skipped": []}

    # ========================================================================
    # HuggingFace models
    # ========================================================================
    print("\n" + "=" * 70)
    print("HUGGINGFACE MODELS")
    print("=" * 70)

    hf_models = get_hf_model_list()
    print(f"Found {len(hf_models)} HF models (excluding LLMs)")

    for model_name, batch_size in sorted(hf_models.items()):
        # Use smaller batch sizes to avoid OOM
        bs = min(batch_size, 8)

        for mode in ["inference", "training"]:
            label = f"hf_{model_name}_{mode}"
            print(f"\n>>> {label} (bs={bs})")
            t0 = time.time()

            n = capture_model(
                model_name, "hf", load_hf_model, bs, device, mode, repros_dir
            )

            elapsed = time.time() - t0
            if n > 0:
                total_merged += n
                results["success"].append(f"{label}: {n} regions ({elapsed:.1f}s)")
                print(f"  [{label}] Merged {n} regions in {elapsed:.1f}s")
            elif n == 0:
                results["skipped"].append(label)

    # ========================================================================
    # TIMM models
    # ========================================================================
    print("\n" + "=" * 70)
    print("TIMM MODELS")
    print("=" * 70)

    timm_models = get_timm_model_list()
    print(f"Found {len(timm_models)} timm models")

    for model_name, batch_size in sorted(timm_models.items()):
        bs = min(batch_size, 32)

        for mode in ["inference", "training"]:
            label = f"timm_{model_name}_{mode}"
            print(f"\n>>> {label} (bs={bs})")
            t0 = time.time()

            n = capture_model(
                model_name, "timm", load_timm_model, bs, device, mode, repros_dir
            )

            elapsed = time.time() - t0
            if n > 0:
                total_merged += n
                results["success"].append(f"{label}: {n} regions ({elapsed:.1f}s)")
                print(f"  [{label}] Merged {n} regions in {elapsed:.1f}s")
            elif n == 0:
                results["skipped"].append(label)

    # ========================================================================
    # Summary
    # ========================================================================
    final_patterns = len(list(canonical_dir.iterdir())) if canonical_dir.exists() else 0
    new_patterns = final_patterns - initial_patterns

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total regions merged: {total_merged}")
    print(f"New canonical patterns: {new_patterns}")
    print(f"Final canonical patterns: {final_patterns}")
    print(f"Successful captures: {len(results['success'])}")
    print(f"Failed/skipped: {len(results['skipped'])}")
    print()
    print("Successful:")
    for s in results["success"]:
        print(f"  {s}")
    if results["failed"]:
        print("\nFailed:")
        for f in results["failed"]:
            print(f"  {f}")


if __name__ == "__main__":
    main()
