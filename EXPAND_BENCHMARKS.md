# Benchmark Repro Set Expansion Tracker

## Current State

**675 canonical repros** from:
- 69 HF dynamo models (B200 local extract)
- 55 H100 CI tlparse (huggingface perf nightly shard 1-5, timm shard 1-7, torchbench shard 1-9)

Note: HF CI coverage is thin — only ~5 models per shard from tlparse, not the full ~50 model suite.

## Expansion Sources (Prioritized)

1. Run HF benchmark suite locally with `capture_hook` (all ~50 models, training + inference)
2. Run timm suite locally with `capture_hook_postfusion` (channels-last, real post-layout-opt strides)
3. Run torchbench suite locally
4. vLLM models (already have some from B200 local)
5. GenAI benchmark from pytorch/pytorch (`benchmarks/dynamo/gen_ai_bench.py`)
6. tritonbench (`github.com/meta-pytorch/tritonbench`) — standalone kernel benchmarks with multiple operator implementations (flash attention, GEMM, layernorm, softmax, etc.)
7. Helion kernels (once available)

## Capture Methods

| Method | Script | When to Use |
|--------|--------|-------------|
| 1 | `capture_hook.py` | Pre-scheduling, fast, good for transformers |
| 2 | `capture_hook_postfusion.py` | Post-scheduling, captures real strides after layout opt. Use for conv/channels-last models (timm) |
| 3 | `ingest_tlparse.py` | From CI artifacts, no local GPU needed, but only gets subset |

## What to Track Per Expansion

- Date
- Source
- # new patterns added
- # shapes added to existing patterns
- Hardware captured on (B200, H100, etc.)
- Pass rate after merge
- Notable new findings

## Planned Expansions

- [ ] Full HF suite (local, training mode) — expected ~200 new patterns
- [ ] Full timm suite (local, channels-last, postfusion hook) — expected ~100 new patterns
- [ ] GenAI bench (gen_ai_bench.py) — targeted LLM patterns
- [ ] tritonbench (meta-pytorch/tritonbench) — reference impls + additional patterns
- [ ] Helion kernels — alternative implementations for impls/ dirs
- [ ] H100 CI full run (all 21 shards, not just the extracted subset)
- [ ] A100 CI run — cross-hardware comparison
- [ ] Backward/training graphs (capture with grad enabled)

## Notes

- Capture (compile + extract) doesn't need precise GPU timing, so it's safe to run without the benchmark lock
- Don't run `bench_parallel.py` (timing-sensitive) concurrently with capture on the same GPU
- If both need to run in parallel, assign different GPUs (e.g., capture on GPU 1, bench on GPU 0)

## Commands

### Full HF suite (local)
```bash
python benchmarks/dynamo/huggingface.py --backend inductor --training \
  --hooks capture_hook.py --output-dir repros/hf_full/
```

### Full timm suite (local, channels-last, postfusion)
```bash
python benchmarks/dynamo/timm_models.py --backend inductor --channels-last \
  --hooks capture_hook_postfusion.py --output-dir repros/timm_full/
```

### Torchbench suite (local)
```bash
python benchmarks/dynamo/torchbench.py --backend inductor --training \
  --hooks capture_hook.py --output-dir repros/torchbench/
```

### GenAI bench
```bash
python benchmarks/dynamo/gen_ai_bench.py --backend inductor \
  --hooks capture_hook.py --output-dir repros/genai/
```

### Ingest from CI tlparse artifacts
```bash
python ingest_tlparse.py --artifact-dir /path/to/ci/artifacts/ \
  --output-dir repros/ci_ingest/
```

### tritonbench (meta-pytorch/tritonbench)
```bash
# Clone tritonbench and use its operator benchmarks as reference impls
git clone https://github.com/meta-pytorch/tritonbench
# Operators available: flash_attention, softmax, layernorm, gemm, jagged, etc.
# These can serve as:
# 1. Alternative impls for existing repros (put in impls/ dirs)
# 2. Additional repro patterns we don't yet cover
# 3. Reference "best known" kernels to compare inductor against
python tritonbench/run.py --op softmax --mode fwd --device cuda
```

## Expansion Log

| Date | Source | New Patterns | New Shapes | Hardware | Pass Rate | Notes |
|------|--------|-------------|------------|----------|-----------|-------|
| (initial) | B200 local + H100 CI | 675 | — | B200/H100 | baseline | — |
| 2026-05-16 | timm channels-last (5 models) | 25 | 148 shapes | B200 | 100% | ResNet50, EfficientNet, MobileNet, ConvNeXt, RegNet |
| 2026-05-16 | HF inference (8 models) | 38 | 87 regions | B200 | 100% | Pythia, Falcon, Phi-2, BERT-large, T5, DistilBERT, Mistral, Qwen2 |
| 2026-05-16 | GenAI patterns (7 ops) | 8 | 9 regions | B200 | 100% | SwiGLU, GeGLU, GQA, RMSNorm, RoPE, causal attn, flash epilogue |
| 2026-05-16 | GenAI varied shapes | 3 | 31 configs | B200 | 100% | decode/prefill/long-context/large-model shapes |
| 2026-05-16 | Full CI suite (51 models x 2) | 638 | 2879 regions | B200 | 97.9% | 33 HF + 18 timm, inference+training. 1499 total repros. |
| 2026-05-16 | HF training (5 models) | 56 | 135 regions | B200 | 100% | BERT, DistilBERT, T5, Pythia, Qwen2 backward passes |
