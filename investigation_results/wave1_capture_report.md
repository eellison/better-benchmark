# Wave-1 Capture Report

**Date:** 2026-06-11 / 2026-06-12  
**Corpus root:** /tmp/scratch_space/recapture_corpus/repros  
**Branch:** investigations-june-2026  
**Run span:** 2026-06-11T19:09:59 to 2026-06-12T13:59:03 (18.8 hours wall, 3.8h compute)

---

## Per-Suite Per-Mode Results

| Suite | Mode | Attempted | OK | Failed | Skipped | Regions |
|-------|------|-----------|-----|--------|---------|---------|
| genai | static | 8 | 8 | 0 | 0 | 12 |
| hf | infer | 39 | 38 | 1 | 0 | 509 |
| hf | train | 39 | 32 | 0 | 7 | 1773 |
| timm | infer | 18 | 18 | 0 | 0 | 465 |
| timm | train | 18 | 17 | 1 | 0 | 1821 |
| torchbench | infer | 54 | 22 | 30 | 2 | 343 |
| torchbench | train | 54 | 18 | 34 | 2 | 1420 |
| **Total** | | **230** | **153** | **66** | **11** | **6343** |

---

## Validation Verdict

**PASSED: all hard invariants satisfied.**

```
[SKIP] baseline repro-count check (non-default corpus root)
[PASS] all 1681 repro.py files parse successfully
[PASS] all 1681 repros have loadable input configs
[PASS] all manifest patterns resolve to canonical/ (1681 hashes)
[PASS] all 1681 canonical directories contain repro.py
[PASS] sampled shapes pass eager validation
[OK]   multi-model patterns have shapes data
[OK]   _shapes_config values parseable (or check skipped)
[SKIP] suite-coverage check (baseline-relative)
```

Per run_log.json: 153/153 ok entries report region-level roundtrip pass (no failed roundtrips).

---

## Failure & Skip Taxonomy

### 1. Llama: Gated Repo Pending Meta Review (1 model, 1 entry)

| Model Key | Error |
|-----------|-------|
| hf/infer/meta-llama/Llama-3.2-1B | 403: request to access model awaiting review from repo authors |

HF token works; Meta agreement awaiting approval. Inference capture blocked; train entry skipped separately (LLM-train class below).

### 2. LLM-Train Skips: Upstream Declares Inference-Only (7 entries)

| Model Key | Reason |
|-----------|--------|
| hf/train/Qwen/Qwen3-0.6B | generation benchmark: inference-only by design |
| hf/train/google/gemma-2-2b | generation benchmark: inference-only by design |
| hf/train/google/gemma-3-4b-it | generation benchmark: inference-only by design |
| hf/train/meta-llama/Llama-3.2-1B | generation benchmark: inference-only by design |
| hf/train/mistralai/Mistral-7B-Instruct-v0.3 | generation benchmark: inference-only by design |
| hf/train/openai/gpt-oss-20b | generation benchmark: inference-only by design |
| hf/train/openai/whisper-tiny | generation benchmark: inference-only by design |

Upstream CI declares eager_fail_to_run; runner loss=pred[0]=logits, backward unsupported. TODO filed in plan (c942cfd30) for our-construction training capture of modern LLMs.

### 3. Swin Train: Compile Timeout (1 entry)

| Model Key | Error |
|-----------|-------|
| timm/train/swin_base_patch4_window7_224 | Subprocess timeout after 5460s (91 min) |

### 4. Torchbench: 64 Failures (30 infer + 34 train)

#### 4a. Detectron2 / COCO install.py (13 models, 26 entries)

detectron2_fasterrcnn_r_101_c4, detectron2_fasterrcnn_r_101_dc5, detectron2_fasterrcnn_r_101_fpn, detectron2_fasterrcnn_r_50_c4, detectron2_fasterrcnn_r_50_dc5, detectron2_fasterrcnn_r_50_fpn, detectron2_fcos_r_50_fpn, detectron2_maskrcnn_r_101_c4, detectron2_maskrcnn_r_101_fpn, detectron2_maskrcnn_r_50_c4, detectron2_maskrcnn_r_50_fpn, vision_maskrcnn, yolov3.

Error: `AssertionError: Couldn't find coco2017/coco128 minimal data dir, please run install.py again.`

#### 4b. FB-Internal Manifold (2 models, 3 entries)

Super_SloMo (infer+train), pytorch_CycleGAN_and_pix2pix (train only).

Error: `No module named 'torchbenchmark.util.framework.fb'`

#### 4c. Models Absent From Checkout / Missing Dependencies (11 models, 22 entries)

| Model | Missing Module |
|-------|---------------|
| DALLE2_pytorch | dalle2_pytorch |
| attention_is_all_you_need_pytorch | ImportError: could not import model |
| drq | kornia |
| fambench_dlrm | optim |
| fambench_xlmr | fairseq |
| fastNLP_Bert | fastNLP |
| pplbench_beanmachine | ImportError: could not import model |
| pytorch_struct | ImportError: could not import model |
| speech_transformer | kaldi_io |
| tacotron2 | inflect |
| detectron2_maskrcnn | detectron2 |

#### 4d. Model Quirks

| Model Key | Category | Error |
|-----------|----------|-------|
| torchbench/infer/maml | maml grad | element 0 does not require grad and does not have a grad_fn |
| torchbench/train/maml | maml grad | Eager run failed |
| torchbench/infer/maml_omniglot | missing data | FileNotFoundError: batch.pt |
| torchbench/train/maml_omniglot | missing data | FileNotFoundError: batch.pt |
| torchbench/train/pyhpc_equation_of_state | pyhpc no train bsize | DEFAULT_TRAIN_BSIZE not implemented |
| torchbench/train/pyhpc_isoneutral_mixing | pyhpc no train bsize | DEFAULT_TRAIN_BSIZE not implemented |
| torchbench/train/pyhpc_turbulent_kinetic_energy | pyhpc no train bsize | DEFAULT_TRAIN_BSIZE not implemented |
| torchbench/infer/opacus_cifar10 | opacus wave-2 | capture hook index bounds validation failure |
| torchbench/train/opacus_cifar10 | opacus wave-2 | Eager run failed |
| torchbench/infer/pytorch_stargan | stargan | FileNotFoundError: celeba/list_attr_celeba.txt |
| torchbench/train/pytorch_stargan | stargan | FileNotFoundError: celeba/list_attr_celeba.txt |
| torchbench/infer/soft_actor_critic | LAPACK | torch.geqrf requires LAPACK |
| torchbench/train/soft_actor_critic | LAPACK | torch.geqrf requires LAPACK |

### 5. Torchbench Skips: Distributed (2 models, 4 entries)

| Model | Reason |
|-------|--------|
| dlrm | Known distributed model |
| moco | Known distributed model |

---

## Value-Inference Fixes Found By The Run

Nine classes of value-inference correctness bugs discovered and fixed during capture:

| # | Class | Commit | Description |
|---|-------|--------|-------------|
| 1 | Maxpool offset constants (window-center) | 02c2bc397 | Constant window-center generation for maxpool offset tensors (TB train assert class) |
| 2 | Maxpool lifted kernel-size resolution | def9bb7f0 | Resolve LIFTED kernel-size params via meta['val'] (vgg16 class) |
| 3 | Min-across-consumers bounds | 8ac1c4a65 | MIN across all consumers + follow gather data edge (hf assert class) |
| 4 | Arithmetic-chain inversion (Longformer) | 1a1fe97a5 | Invert arithmetic chains between placeholder and index consumer |
| 5 | Float-to-index chains (OPT) | fed14c065 | Float placeholders feeding index chains get bounded non-negative gen |
| 6 | index_put position-to-dim mapping (Yitu) | f34959391 | index_put/index.Tensor indices position maps to DIM incl. None slots |
| 7 | Permutation detection through lifted alloc/iota (gpt-oss MoE) | 292151139 | Resolve lifted alloc shapes + iota lengths |
| 8 | Alias groups (packed qkv) | c01b3b237 | Shared-storage inputs preserved end to end; device ordinal normalization |
| 9 | Zero-input configs | 0339098f9 | [] != missing; zero-input points are valid configs |

Additional infrastructure fixes landed in the same window:

| Class | Commit | Description |
|-------|--------|-------------|
| fp32-only / autocast policy (GoogleFnet) | cdfef2e15 | Honor upstream fp32-only list; LLM generation benchmarks inference-only |
| Iterative dedupe structural hash (pytorch-side, MobileBert/XLNet) | dde209019 | Fix bound test: assert correct dataflow roles (data arg -> 2, index arg -> 512) |
| storage_offset / u64 annotation | 62b41ae94 | annotation parser: storage_offset unknown (None) not 0; u16/u32/u64 dtype tokens |
| Codec u16/u32/u64 + fp8 | 50603c5bb | codec: u16/u32/u64 + fp8 short dtypes; tests for offset/dtype preservation |
| Alias nbytes through shapes.json | d28e80b4c | alias review fixes: nbytes travels through shapes.json load; merge backfills richer inputs |
| genai static recapture | 5498cfbd0 | Static recapture of genai microbenchmark graphs through the new pipeline |

---

## Caveats

### Models captured before mid-run fixes lack alias tags and tighter bounds metadata

152 of 153 ok models were captured before the last inference fix (permutation detection, 292151139, 2026-06-12T13:59). Per user decision: no recapture needed; roundtrip validation passes for all.

Aggregate breakdown by fix timestamp:
- Before alias_groups (c01b3b237, 2026-06-11T20:04): earliest captures (timm/infer batch).
- Before min-across-consumers (8ac1c4a65, 2026-06-11T21:49): majority of timm + early hf/torchbench.
- Before maxpool lifted-param (def9bb7f0, 2026-06-11T22:21): additional hf models.
- After all 2026-06-11 fixes: remaining hf, genai batch (2026-06-12).
- After all fixes including OPT/Yitu/perm: 1 model (gpt-oss-20b infer recapture).

### swin_base_patch4_window7_224 train: 91-minute compile timeout

Overnight run optional.

---

## Corpus Statistics

| Metric | Value |
|--------|-------|
| Canonical pattern dirs | 1681 |
| Total shapes.json files | 1681 |
| Total shape points | 4712 |
| Total regions captured | 6343 |
| Unique model/mode combos (OK) | 153 |
| Unique models (all statuses) | 119 (genai: 8, hf: 39, timm: 18, torchbench: 54) |
| Compute wall time (ok entries) | 3.8h |

**Previous corpus:** 1482 canonical dirs, fp32, pre-fidelity.  
**Current corpus:** 1681 canonical dirs, bf16 (inference) / AMP (training), with structured inputs, alias groups, roundtrip validation, and value-inference annotations.

---

## Next Steps

1. Post-capture bench sweep: run performance measurements on captured corpus.
2. Regenerate oracle work queue per INVEST_INSTRUCTIONS.MD claim flow.
3. Section 5: oracle migration.
4. Section 6: atomic flip.
5. Llama capture: pending Meta approval of gated-repo access request.
6. swin overnight run: optional (91-min compile; low priority).
7. Wave 2: dynamic shapes (vllm, genai-dynamic, opacus full suite) + LLM-train TODO (our-construction training graphs for modern models per c942cfd30 plan).
