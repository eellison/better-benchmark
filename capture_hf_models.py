"""
Capture kernels from HuggingFace models for the benchmark repro set.

Models captured:
1. EleutherAI/pythia-410m (GPT-NeoX architecture)
2. tiiuae/falcon-rw-1b (Falcon architecture)
3. microsoft/phi-2 (Phi architecture)
4. bert-large-uncased (BERT-large)
5. t5-base (T5 encoder-decoder)
6. distilbert-base-uncased (DistilBERT)
7. mistralai/Mistral-7B-v0.1 (Mistral architecture) - may OOM
8. Qwen/Qwen2-0.5B (as Gemma alternative - similar modern arch)
"""
import sys
import types
import importlib.machinery

# ============================================================
# MUST come first: fake torchvision to avoid version conflict
# between dev torch at /tmp/pytorch-work and conda torchvision
# ============================================================

class _FakeModule(types.ModuleType):
    """A module that returns None/dummy for any attribute access."""
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return lambda *a, **kw: None

class _AnyAttr:
    """A flexible class that has any attribute (for enums etc)."""
    def __getattr__(self, name):
        return name
    def __class_getitem__(cls, item):
        return cls

_modules_to_fake = [
    'torchvision', 'torchvision.io',
    'torchvision.transforms', 'torchvision.transforms.functional',
    'torchvision.transforms.v2', 'torchvision.transforms.v2.functional',
    'torchvision.datasets', 'torchvision.models',
    'torchvision.ops', 'torchvision.utils',
]
for _mod_name in _modules_to_fake:
    _m = _FakeModule(_mod_name)
    _m.__spec__ = importlib.machinery.ModuleSpec(_mod_name, None, origin='fake')
    _m.__path__ = []
    _m.__version__ = '0.20.0'
    sys.modules[_mod_name] = _m

sys.modules['torchvision.transforms'].InterpolationMode = _AnyAttr()
sys.modules['torchvision.transforms.functional'].InterpolationMode = _AnyAttr()
sys.modules['torchvision.io'].ImageReadMode = _AnyAttr()

# Now safe to import torch from dev path
sys.path.insert(0, '/tmp/pytorch-work')

import gc
import os
import json
import traceback
from pathlib import Path

import torch
import torch._inductor.utils

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, '/tmp/scratch_space/better_benchmark')
from capture_hook import install_capture_hook, uninstall_capture_hook

CAPTURES_BASE = Path("/tmp/scratch_space/better_benchmark/captures_hf")
CAPTURES_BASE.mkdir(exist_ok=True)

BATCH = 4
SEQ = 512

results = {}


def cleanup():
    """Free GPU memory between models."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def capture_model(model_name, label, load_fn, input_fn):
    """Generic capture routine for a model."""
    print(f"\n{'='*60}")
    print(f"CAPTURING: {label} ({model_name})")
    print(f"{'='*60}")

    output_dir = str(CAPTURES_BASE / label)
    cleanup()

    try:
        # Fresh inductor cache
        torch._inductor.utils.fresh_inductor_cache()

        # Load model
        print(f"  Loading model...")
        model = load_fn()
        model = model.cuda().eval()
        print(f"  Model loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        # Create inputs
        print(f"  Creating inputs...")
        inputs = input_fn()

        # Install capture hook
        install_capture_hook(output_dir, label=label)

        # Compile and run
        print(f"  Compiling...")
        compiled = torch.compile(model)

        print(f"  Running forward pass...")
        with torch.no_grad():
            out = compiled(**inputs)
            torch.cuda.synchronize()

        print(f"  Forward pass complete.")

        # Uninstall hook
        uninstall_capture_hook()

        # Count captures
        index_path = Path(output_dir) / "index.json"
        if index_path.exists():
            with open(index_path) as f:
                captures = json.load(f)
            n_captured = len(captures)
        else:
            n_captured = 0

        results[label] = {"status": "success", "captured": n_captured}
        print(f"  SUCCESS: Captured {n_captured} regions")

        # Cleanup model
        del model, compiled, out, inputs

    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        results[label] = {"status": "failed", "error": str(e)}
        try:
            uninstall_capture_hook()
        except:
            pass

    cleanup()


# ============================================================
# Model definitions
# ============================================================

def load_pythia():
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-410m", dtype=torch.float16
    )

def inputs_causal_lm():
    input_ids = torch.randint(0, 30000, (BATCH, SEQ), device="cuda")
    attention_mask = torch.ones(BATCH, SEQ, dtype=torch.long, device="cuda")
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def load_falcon():
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        "tiiuae/falcon-rw-1b", dtype=torch.float16, trust_remote_code=True
    )

def inputs_falcon():
    input_ids = torch.randint(0, 50000, (BATCH, SEQ), device="cuda")
    attention_mask = torch.ones(BATCH, SEQ, dtype=torch.long, device="cuda")
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def load_phi2():
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", dtype=torch.float16, trust_remote_code=True
    )

def inputs_phi2():
    input_ids = torch.randint(0, 50000, (BATCH, SEQ), device="cuda")
    attention_mask = torch.ones(BATCH, SEQ, dtype=torch.long, device="cuda")
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def load_bert_large():
    from transformers import AutoModelForMaskedLM
    return AutoModelForMaskedLM.from_pretrained(
        "bert-large-uncased", dtype=torch.float16
    )

def inputs_bert():
    input_ids = torch.randint(0, 30000, (BATCH, SEQ), device="cuda")
    attention_mask = torch.ones(BATCH, SEQ, dtype=torch.long, device="cuda")
    token_type_ids = torch.zeros(BATCH, SEQ, dtype=torch.long, device="cuda")
    return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}


def load_t5():
    from transformers import T5ForConditionalGeneration
    return T5ForConditionalGeneration.from_pretrained(
        "t5-base", dtype=torch.float16
    )

def inputs_t5():
    input_ids = torch.randint(0, 30000, (BATCH, SEQ), device="cuda")
    decoder_input_ids = torch.randint(0, 30000, (BATCH, SEQ // 4), device="cuda")
    attention_mask = torch.ones(BATCH, SEQ, dtype=torch.long, device="cuda")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
    }


def load_distilbert():
    from transformers import AutoModelForMaskedLM
    return AutoModelForMaskedLM.from_pretrained(
        "distilbert-base-uncased", dtype=torch.float16
    )

def inputs_distilbert():
    input_ids = torch.randint(0, 30000, (BATCH, SEQ), device="cuda")
    attention_mask = torch.ones(BATCH, SEQ, dtype=torch.long, device="cuda")
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def load_mistral():
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1", dtype=torch.float16,
        low_cpu_mem_usage=True
    )

def inputs_mistral():
    # Shorter seq for 7B model to avoid OOM
    input_ids = torch.randint(0, 30000, (BATCH, 256), device="cuda")
    attention_mask = torch.ones(BATCH, 256, dtype=torch.long, device="cuda")
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def load_qwen2():
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B", dtype=torch.float16, trust_remote_code=True
    )

def inputs_qwen():
    input_ids = torch.randint(0, 50000, (BATCH, SEQ), device="cuda")
    attention_mask = torch.ones(BATCH, SEQ, dtype=torch.long, device="cuda")
    return {"input_ids": input_ids, "attention_mask": attention_mask}


# ============================================================
# Run captures
# ============================================================

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Captures base: {CAPTURES_BASE}")

    models_to_capture = [
        ("EleutherAI/pythia-410m", "pythia_410m", load_pythia, inputs_causal_lm),
        ("tiiuae/falcon-rw-1b", "falcon_rw_1b", load_falcon, inputs_falcon),
        ("microsoft/phi-2", "phi_2", load_phi2, inputs_phi2),
        ("bert-large-uncased", "bert_large", load_bert_large, inputs_bert),
        ("t5-base", "t5_base", load_t5, inputs_t5),
        ("distilbert-base-uncased", "distilbert", load_distilbert, inputs_distilbert),
        ("mistralai/Mistral-7B-v0.1", "mistral_7b", load_mistral, inputs_mistral),
        ("Qwen/Qwen2-0.5B", "qwen2_0.5b", load_qwen2, inputs_qwen),
    ]

    for model_name, label, load_fn, input_fn in models_to_capture:
        capture_model(model_name, label, load_fn, input_fn)

    # Summary
    print(f"\n{'='*60}")
    print("CAPTURE SUMMARY")
    print(f"{'='*60}")
    total_captured = 0
    for label, info in results.items():
        status = info["status"]
        if status == "success":
            n = info["captured"]
            total_captured += n
            print(f"  {label}: {n} regions captured")
        else:
            print(f"  {label}: FAILED - {info['error'][:80]}")

    print(f"\n  TOTAL: {total_captured} regions captured across {sum(1 for v in results.values() if v['status']=='success')} models")

    # Save results
    results_path = CAPTURES_BASE / "capture_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")
