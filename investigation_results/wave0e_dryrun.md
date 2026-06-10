# Wave 0e Mini Dry-Run Results (2026-06-10)

## Configuration
- GPU: CUDA_VISIBLE_DEVICES=1, INDUCTOR_GPU_BENCH_LOCK=1
- Corpus: /tmp/wave0e_corpus/repros
- Script: scripts/run_recapture.py

## Models Tested

| Suite | Mode | Model | Status | Regions | Wall Time | Roundtrip |
|-------|------|-------|--------|---------|-----------|-----------|
| timm | infer | mobilenetv2_100 | ok | 24 | 11.4s | 1 ok / 0 failed |
| timm | train | tf_efficientnet_b0 | ok | 213 | 60.8s | 2 ok / 0 failed |
| torchbench | infer | resnet18 | ok | 12 | 12.5s | 1 ok / 0 failed |
| torchbench | infer | moco | skipped | 0 | 0s | - |
| hf | infer | AlbertForMaskedLM | ok | 15 | 16.0s | 1 ok / 0 failed |

## Verification Results

### 1. run_log complete
- 5 entries total: 4 ok, 1 skipped, 0 failed
- Every status recorded with wall_time, region_count, roundtrip_summary

### 2. Resumability
- PASS: Re-running all captures with same --corpus-root correctly skips
  all 4 ok models (dry-run shows "0 models to capture, N already done")
- Skipped model (moco) correctly re-attempts (status=skipped != status=ok)

### 3. Every captured sidecar roundtrip:ok
- PASS: All 5 full_graph_*.meta.json files have `"roundtrip": "ok"`
- 0 failures, 0 missing stamps

### 4. shapes.json present per repro
- PASS: 48/48 canonical repro dirs have shapes.json
- Dtype confirmed bf16 in infer mode signatures
- Train mode (autocast) shows bf16 tensors from autocast context

### 5. skipped.json has the skip with reason
- PASS: skipped.json contains moco entry with reason "Known distributed model: moco"

## Dtype Policy Verification
- **Infer mode**: signatures show bf16 dtypes (e.g., `T([128, 320, 7, 7], bf16, ...)`)
- **Train mode**: autocast produces bf16 compute tensors alongside f32 parameters
  (correct AMP behavior — parameters stay f32, activations cast to bf16)

## Notes
- The train mode (tf_efficientnet_b0) produces 213 regions / 2 full graphs
  because backward generates a separate post-grad graph
- Total capture time for all 5 models: ~101s (compile-dominated)
- Atomic log writes confirmed: run_log.json fully consistent after each model
