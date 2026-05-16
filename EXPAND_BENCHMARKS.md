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
