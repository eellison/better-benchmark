# Benchmark Infrastructure Roadmap

## Current State (2026-05-19)

Infrastructure validated end-to-end:
- Capture hook: DAG-structure hash, transparent ops (getitem/views fuse through), shape lifting
- Dedup: same pattern_hash → one repro with multiple shape configs
- Compact shapes.txt: `T([128, 64, 56, 56], f32, stride=(...)), S([8192])` format
- Benchmark harness: multi-shape, memcopy SOL, coord-descent tuning, kernel counting
- Parallel runner: persistent workers, GPU locks, compare workflow

---

## Recapture Progress (this session)

### [DONE] timm (18 models, channels_last, batch from CI list)
```
adv_inception_v3 128       ghostnet_100 512         nfnet_l0 128
beit_base_patch16_224 128  inception_v3 128         repvgg_a2 128
convnextv2_nano 128        mobilenetv2_100 128      swin_base 128
deit_base 128              mobilenetv3_large 512    tf_efficientnet_b0 128
deit_tiny 128              mobilevit_s 128          visformer_small 128
dm_nfnet_f0 128            vit_base_dinov2 128      vit_base_siglip 128
```
- Both infer + train, channels-last
- 1399 regions → ~351 unique patterns
- 351/355 pass (4 marked failing: int8 pool offset bounds, Bool dtype)
- ~7 min wall time (2x B200 parallel)

### [IN PROGRESS] HF (32 models, infer + train)
- Batch sizes from `huggingface_models_list.txt` (4-256 per model)
- Running on both GPUs in parallel (16 models each)
- Skipping 7 models needing HF auth/download

### [PARTIAL] torchbench — via CI tlparse artifacts
- tlparse artifacts from CI have `inductor_post_grad_graph_*.txt` and `fx_graph_runnable_*.txt`
- Downloaded 15 graphs from run 26008397786 (shards with cache misses)
- These are full post-grad graphs (segment_anything, etc.) — can feed to capture_hook.process_graph()
- Most shards had cache hits (no graph logged). Need a run with `fx_graph_cache=False` for full coverage
- Path: `gh run download <run_id> --repo pytorch/pytorch --pattern "tlparse-*torchbench*"`
- Script needed: `scripts/ingest_tlparse.py` to parse post_grad_graph txt → process_graph → merge

### [TODO] GenAI / tritonbench
- Decode patterns, operator-level kernels
- vLLM configs

### [TODO] Final validation + commit
- Validate ALL repros runnable (mark failures, don't delete)
- Commit per-suite (timm, HF, torchbench, genai)
- Optimize execution: subprocess pool, GPU lock around timing only

---

## Short-term (next few days)

1. Full recapture of all suites at correct CI batch sizes
2. Validate 100% pass rate (default + all shape configs)
3. Run baseline benchmark sweep on clean set
4. Freeze as `benchmarks/v2.json`
5. Land easy inductor fixes with benchmark evidence:
   - combo_kernels enable (3x on 60% of severe gaps)
   - num_warps=2 for persistent INNER (1.28x)
   - Raise persistent RBLOCK to 4096 on B200 (1.4-1.6x)

---

## Medium-term (next 2 weeks)

- ND grid codegen for transpose/NHWC (3.0-3.4x)
- MixOrderReduction threshold (2.0x, every norm backward)
- Rotary embed pattern match (2.04x, every LLM)
- Cross-entropy: fuse gather into softmax (1.69x)
- Nightly regression tracking
- `bench_report.py` for PR descriptions

---

## Longer-term

### Coverage expansion
- torchbench: 58 CI models (conv-heavy: YOLO, maskrcnn, stable_diffusion_unet)
- Quantized models (int8, fp8)
- Dynamic shape captures (same model at multiple batch sizes)
- Backward for all timm models
- More decode shapes: batch=64/128, seq=1

### Harder inductor fixes
- Batched weight-grad reduction kernel (6.5x)
- ConcatKernel virtual views (1.5-2x)
- Push reduction through cat (2x)
- Horizontal fusion for same-shape reductions (1.5x)

### CI integration
- GitHub Action: on PR touching inductor, post benchmark diff as comment
- Web dashboard: gap distribution, historical trends
- Per-model regression detection
- Nightly sweeps on main with alerting

---

## Validation Workflow (for any inductor change)

```bash
# 1. Baseline
python scripts/bench_parallel.py repros/canonical/ --tag before

# 2. Apply fix

# 3. After
python scripts/bench_parallel.py repros/canonical/ --tag after

# 4. Compare
python scripts/bench_parallel.py --compare before after
```
