# Inductor Fusion Gap Analysis

Systematic analysis of multi-kernel fusion failures in PyTorch Inductor across HuggingFace and vLLM models.

## What this is

We extracted every inductor compilation region from 33 models (26 HF dynamo + 7 vLLM), isolated the ones that produce more than 1 kernel, and classified them by root cause. The goal is to identify the highest-impact fusion improvements for LLM inference on B200 GPUs.

## Key findings

59 of 423 successful inference compilation regions produce >1 kernel (14%).

| Root Cause | Repros | Impact |
|---|---|---|
| Q/K RoPE iteration domain mismatch | 8 | Every GQA model, every layer |
| Cross-entropy reduction mismatch | 28 | Training/eval with loss |
| ConcatKernel blocking pointwise_cat | 2 | DeepSeek-V3, Qwen3 |
| Qwen3 RMSNorm+RoPE intermediate deps | 5 | Qwen3 family |
| MoE assert_async + two-stage any | 1 | MoE models |
| Large multi-op regions (expected) | 12 | Various |
| Other (Longformer, Reformer, GPT-OSS) | 3 | Model-specific |

See [CLASSIFICATION.md](docs/CLASSIFICATION.md) for detailed analysis.

## Repository structure

```
repo/
  README.md
  docs/
    CLASSIFICATION.md          # Full fusion failure classification
    METHODOLOGY.md             # How we extracted and probed
  repros/
    multi_kernel_inference/    # 59 standalone repro files (torch.compile, no deps)
  scripts/
    extract_reductions.py      # Extract compilation regions from models
    probe_batch.py             # Probe kernel counts across GPUs
    classify_fusion.py         # Run TORCH_LOGS to classify fusion failures
    run_with_logs.py           # Run single repro with fusion logging
  analysis/
    fusion_classification.json # Machine-readable classification results
    SUMMARY.json               # Kernel counts for all multi-kernel repros
```

## Running a repro

Each repro is self-contained. No pytorch modifications needed.

```bash
# Count kernels
python -c "
import torch, torch._inductor.metrics as m
m.reset()
from repros.multi_kernel_inference.k03_vllm_mistralai_Mistral_7B_Instruct_v0_3_inference__region_011_pointwise_108892bc6e62_8f1de50a import Repro, make_inputs
compiled = torch.compile(Repro().cuda())
with torch.no_grad(): compiled(*make_inputs())
print(f'Kernels: {m.generated_kernel_count}')
"

# See fusion decisions
TORCH_LOGS="ir_pre_fusion,fusion,ir_post_fusion" python repros/multi_kernel_inference/k03_vllm_mistralai_Mistral_7B_Instruct_v0_3_inference__region_011_pointwise_108892bc6e62_8f1de50a.py
```

## Models covered

**HF Dynamo (26):** Albert, AllenaiLongformerBase, BartForCausalLM, BertForMaskedLM, BlenderbotForCausalLM, BlenderbotForConditionalGeneration, DebertaV2ForMaskedLM, DistilBertForMaskedLM, DistillGPT2, ElectraForCausalLM, GoogleFnet, GPT2ForSequenceClassification, GPTNeoForCausalLM, GPTNeoForSequenceClassification, LayoutLMForMaskedLM, M2M100ForConditionalGeneration, MBartForCausalLM, MegatronBertForCausalLM, MobileBertForMaskedLM, MT5ForConditionalGeneration, OPTForCausalLM, PegasusForCausalLM, PLBartForCausalLM, Reformer, RobertaForCausalLM, Qwen3-0.6B

**vLLM (7):** DeepSeek-V3, Mistral-7B-Instruct-v0.3, Qwen3-0.6B, Qwen3-30B-A3B, GPT-OSS-20B, Llama-3.1-8B, Llama-4-Scout-17B-16E
