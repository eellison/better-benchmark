# Model Performance Attribution Report

## Overview

Models analyzed: 10
Total compile time across all models: 37363.2 us
Total achievable time: 21897.7 us
Total gap (opportunity): 15465.4 us
Overall potential speedup: 1.706x

### Key Limitations

1. **No per-shape oracles**: Each repro measured at ONE specific shape.
   Model may run at different batch/seq shapes; oracle doesn't directly apply.
2. **Non-fusible ops not captured**: matmul, conv, embedding are not in corpus.
3. **No cross-partition overhead**: dispatch, memory pressure, boundaries not modeled.
4. **Approach B can be totally off** without per-shape measurements,
   since the full graph runs at the model's actual shape while partition
   timings come from potentially different shapes.

## Summary Table (sorted by gap)

| Model | Graphs | Partitions | Compile (us) | Achievable (us) | Gap (us) | Speedup |
|-------|--------|-----------|-------------|-----------------|----------|---------|
| torchbench_hf_T5_train | 2 | 25 | 8948.4 | 2983.1 | 5965.3 | 3.000x |
| genai_SoftmaxBackward | 2 | 2 | 6242.3 | 4623.3 | 1619.0 | 1.350x |
| genai_SoftmaxForward | 1 | 1 | 3494.1 | 1942.9 | 1551.2 | 1.798x |
| torchbench_pytorch_unet_infer | 1 | 3 | 1629.2 | 119.5 | 1509.7 | 13.631x |
| timm_adv_inception_v3_infer | 1 | 12 | 3187.5 | 1999.3 | 1188.2 | 1.594x |
| timm_inception_v3_infer | 1 | 12 | 3187.5 | 1999.3 | 1188.2 | 1.594x |
| torchbench_resnet18_train | 2 | 12 | 4054.4 | 3060.7 | 993.6 | 1.325x |
| torchbench_squeezenet1_1_train | 2 | 13 | 4784.0 | 4059.0 | 725.0 | 1.179x |
| torchbench_squeezenet1_1_infer | 1 | 5 | 1235.7 | 539.3 | 696.4 | 2.291x |
| torchbench_hf_T5_base_infer | 1 | 11 | 600.1 | 571.3 | 28.8 | 1.050x |

## Detailed Per-Model Reports

# Model Attribution: torchbench_hf_T5_train

## Summary

- Graphs: 2
- Total partitions: 25
- Captured in corpus: 25
- With timing data: 25
- With oracle floor: 1

## Approach A: Static Accounting (Bottom-Up)

- Sum compile time (best config): 8948.4 us
- Achievable floor (min of oracle/compile): 2983.1 us
- Gap (opportunity): 5965.3 us
- Potential speedup: 3.000x

### Limitations

- Oracles measured at ONE specific shape only; model may run at different shapes
- Non-fusible ops (matmul, conv, etc.) not captured in repro corpus
- No cross-partition overhead accounted for (dispatch, memory pressure)

## Per-Partition Breakdown

| # | Graph | Repro ID | Kind | Compile (us) | Oracle (us) | Gap (us) | Classification |
|---|-------|----------|------|-------------|-------------|----------|----------------|
| 1 | torchbench_hf_T5_train_000 | amax_sum_sum_5352dbff8c2c | reduction | 330.8 | N/A | N/A | amax_sum_sum |
| 2 | torchbench_hf_T5_train_000 | amax_sum_557afc9e24ea | reduction | 326.4 | N/A | N/A | amax_sum |
| 3 | torchbench_hf_T5_train_000 | amax_sum_amax_68fa105ccaf0 | reduction | 804.9 | N/A | N/A | online_softmax_cross_entropy |
| 4 | torchbench_hf_T5_train_000 | mean_mean_6cb471f41ecf | reduction | 39.0 | N/A | N/A | mean_mean |
| 5 | torchbench_hf_T5_train_000 | pointwise_73d2b3dfee0b | pointwise | 16.0 | N/A | N/A | pointwise |
| 6 | torchbench_hf_T5_train_000 | pointwise_9e64ab4c6018 | pointwise | 56.7 | N/A | N/A | pointwise |
| 7 | torchbench_hf_T5_train_000 | amax_sum_ab3238e72dc2 | reduction | 249.6 | N/A | N/A | amax_sum |
| 8 | torchbench_hf_T5_train_000 | pointwise_af2e722aed5d | pointwise | 24.8 | N/A | N/A | pointwise |
| 9 | torchbench_hf_T5_train_000 | mean_b440f707e405 | reduction | 20.2 | N/A | N/A | mean |
| 10 | torchbench_hf_T5_train_000 | mean_d87780a73e70 | reduction | 30.4 | N/A | N/A | mean |
| 11 | torchbench_hf_T5_train_000 | pointwise_e31a9c1b437e | pointwise | 26.1 | N/A | N/A | pointwise |
| 12 | torchbench_hf_T5_train_000 | mean_feaf8e3a5f0b | reduction | 30.7 | N/A | N/A | mean |
| 13 | torchbench_hf_T5_train_001 | pointwise_035529d03c14 | pointwise | 13.3 | N/A | N/A | pointwise |
| 14 | torchbench_hf_T5_train_001 | pointwise_187bbcc9d7ba | pointwise | 16.2 | N/A | N/A | pointwise |
| 15 | torchbench_hf_T5_train_001 | sum_sum_sum_5a4f87bbd879 | reduction | 126.8 | N/A | N/A | sum_sum_sum |
| 16 | torchbench_hf_T5_train_001 | sum_5f7b790ecd66 | reduction | 194.5 | N/A | N/A | sum |
| 17 | torchbench_hf_T5_train_001 | sum_sum_6dbca49c2393 | reduction | 24.5 | N/A | N/A | sum_sum |
| 18 | torchbench_hf_T5_train_001 | sum_a8c7f14e9218 | reduction | 68.0 | N/A | N/A | sum |
| 19 | torchbench_hf_T5_train_001 | sum_sum_sum_afd118ca907d | reduction | 6364.2 | 398.9 | 5965.3 | sum_sum_sum |
| 20 | torchbench_hf_T5_train_001 | pointwise_b46f81501f11 | pointwise | 20.7 | N/A | N/A | pointwise |
| 21 | torchbench_hf_T5_train_001 | sum_sum_bc46e8b5be67 | reduction | 19.9 | N/A | N/A | sum_sum |
| 22 | torchbench_hf_T5_train_001 | sum_bfc058b2d299 | reduction | 12.1 | N/A | N/A | sum |
| 23 | torchbench_hf_T5_train_001 | pointwise_d604053b7cd8 | pointwise | 31.4 | N/A | N/A | pointwise |
| 24 | torchbench_hf_T5_train_001 | sum_sum_e6b32bb1b384 | reduction | 25.4 | N/A | N/A | sum_sum |
| 25 | torchbench_hf_T5_train_001 | sum_sum_fdefa5604cda | reduction | 75.9 | N/A | N/A | sum_sum |

## Time by Classification

| Classification | Count | Compile (us) | Achievable (us) | Gap (us) |
|---------------|-------|-------------|-----------------|----------|
| sum_sum_sum | 2 | 6491.0 | 525.7 | 5965.3 |
| amax_sum_sum | 1 | 330.8 | 330.8 | 0.0 |
| amax_sum | 2 | 576.0 | 576.0 | 0.0 |
| online_softmax_cross_entropy | 1 | 804.9 | 804.9 | 0.0 |
| mean_mean | 1 | 39.0 | 39.0 | 0.0 |
| pointwise | 8 | 205.2 | 205.2 | 0.0 |
| mean | 3 | 81.3 | 81.3 | 0.0 |
| sum | 3 | 274.7 | 274.7 | 0.0 |
| sum_sum | 4 | 145.7 | 145.7 | 0.0 |

## Approach B: End-to-End (Top-Down)

No full_graph_*.py files found for this model.
End-to-end timing not available.

---

# Model Attribution: genai_SoftmaxBackward

## Summary

- Graphs: 2
- Total partitions: 2
- Captured in corpus: 2
- With timing data: 2
- With oracle floor: 1

## Approach A: Static Accounting (Bottom-Up)

- Sum compile time (best config): 6242.3 us
- Achievable floor (min of oracle/compile): 4623.3 us
- Gap (opportunity): 1619.0 us
- Potential speedup: 1.350x

### Limitations

- Oracles measured at ONE specific shape only; model may run at different shapes
- Non-fusible ops (matmul, conv, etc.) not captured in repro corpus
- No cross-partition overhead accounted for (dispatch, memory pressure)

## Per-Partition Breakdown

| # | Graph | Repro ID | Kind | Compile (us) | Oracle (us) | Gap (us) | Classification |
|---|-------|----------|------|-------------|-------------|----------|----------------|
| 1 | genai_SoftmaxBackward_000 | amax_sum_sum_dc96a87ba8db | reduction | 3257.3 | 1638.4 | 1619.0 | online_softmax_cross_entropy |
| 2 | genai_SoftmaxBackward_001 | sum_e00c7291b6ee | reduction | 2985.0 | N/A | N/A | multi_output_reduction_templates |

## Time by Classification

| Classification | Count | Compile (us) | Achievable (us) | Gap (us) |
|---------------|-------|-------------|-----------------|----------|
| online_softmax_cross_entropy | 1 | 3257.3 | 1638.4 | 1619.0 |
| multi_output_reduction_templates | 1 | 2985.0 | 2985.0 | 0.0 |

## Approach B: End-to-End (Top-Down)

No full_graph_*.py files found for this model.
End-to-end timing not available.

---

# Model Attribution: genai_SoftmaxForward

## Summary

- Graphs: 1
- Total partitions: 1
- Captured in corpus: 1
- With timing data: 1
- With oracle floor: 1

## Approach A: Static Accounting (Bottom-Up)

- Sum compile time (best config): 3494.1 us
- Achievable floor (min of oracle/compile): 1942.9 us
- Gap (opportunity): 1551.2 us
- Potential speedup: 1.798x

### Limitations

- Oracles measured at ONE specific shape only; model may run at different shapes
- Non-fusible ops (matmul, conv, etc.) not captured in repro corpus
- No cross-partition overhead accounted for (dispatch, memory pressure)

## Per-Partition Breakdown

| # | Graph | Repro ID | Kind | Compile (us) | Oracle (us) | Gap (us) | Classification |
|---|-------|----------|------|-------------|-------------|----------|----------------|
| 1 | genai_SoftmaxForward_000 | amax_sum_3ed297ef02cd | reduction | 3494.1 | 1942.9 | 1551.2 | online_softmax_cross_entropy |

## Time by Classification

| Classification | Count | Compile (us) | Achievable (us) | Gap (us) |
|---------------|-------|-------------|-----------------|----------|
| online_softmax_cross_entropy | 1 | 3494.1 | 1942.9 | 1551.2 |

## Approach B: End-to-End (Top-Down)

No full_graph_*.py files found for this model.
End-to-end timing not available.

---

# Model Attribution: torchbench_pytorch_unet_infer

## Summary

- Graphs: 1
- Total partitions: 3
- Captured in corpus: 3
- With timing data: 3
- With oracle floor: 1

## Approach A: Static Accounting (Bottom-Up)

- Sum compile time (best config): 1629.2 us
- Achievable floor (min of oracle/compile): 119.5 us
- Gap (opportunity): 1509.7 us
- Potential speedup: 13.631x

### Limitations

- Oracles measured at ONE specific shape only; model may run at different shapes
- Non-fusible ops (matmul, conv, etc.) not captured in repro corpus
- No cross-partition overhead accounted for (dispatch, memory pressure)

## Per-Partition Breakdown

| # | Graph | Repro ID | Kind | Compile (us) | Oracle (us) | Gap (us) | Classification |
|---|-------|----------|------|-------------|-------------|----------|----------------|
| 1 | torchbench_pytorch_unet_infer_000 | pointwise_3698b0e606e1 | pointwise | 11.7 | N/A | N/A | pointwise |
| 2 | torchbench_pytorch_unet_infer_000 | pointwise_8793407401ab | pointwise | 1609.7 | 100.0 | 1509.7 | layout_indexing_stencil_fusion |
| 3 | torchbench_pytorch_unet_infer_000 | pointwise_d66b13ff6ca0 | pointwise | 7.8 | N/A | N/A | pointwise |

## Time by Classification

| Classification | Count | Compile (us) | Achievable (us) | Gap (us) |
|---------------|-------|-------------|-----------------|----------|
| layout_indexing_stencil_fusion | 1 | 1609.7 | 100.0 | 1509.7 |
| pointwise | 2 | 19.5 | 19.5 | 0.0 |

## Approach B: End-to-End (Top-Down)

No full_graph_*.py files found for this model.
End-to-end timing not available.

---

# Model Attribution: timm_adv_inception_v3_infer

## Summary

- Graphs: 1
- Total partitions: 12
- Captured in corpus: 12
- With timing data: 12
- With oracle floor: 2

## Approach A: Static Accounting (Bottom-Up)

- Sum compile time (best config): 3187.5 us
- Achievable floor (min of oracle/compile): 1999.3 us
- Gap (opportunity): 1188.2 us
- Potential speedup: 1.594x

### Limitations

- Oracles measured at ONE specific shape only; model may run at different shapes
- Non-fusible ops (matmul, conv, etc.) not captured in repro corpus
- No cross-partition overhead accounted for (dispatch, memory pressure)

## Per-Partition Breakdown

| # | Graph | Repro ID | Kind | Compile (us) | Oracle (us) | Gap (us) | Classification |
|---|-------|----------|------|-------------|-------------|----------|----------------|
| 1 | timm_adv_inception_v3_infer | pointwise_13f26d420251 | pointwise | 640.8 | 100.0 | 540.8 | layout_indexing_stencil_fusion |
| 2 | timm_adv_inception_v3_infer | pointwise_3698b0e606e1 | pointwise | 11.7 | N/A | N/A | pointwise |
| 3 | timm_adv_inception_v3_infer | pointwise_4f6d85385d17 | pointwise | 87.8 | N/A | N/A | pointwise |
| 4 | timm_adv_inception_v3_infer | mean_5c93e9826aa8 | reduction | 72.6 | N/A | N/A | mean |
| 5 | timm_adv_inception_v3_infer | pointwise_6d842b54b40d | pointwise | 257.0 | N/A | N/A | pointwise |
| 6 | timm_adv_inception_v3_infer | pointwise_71e3a6c09140 | pointwise | 747.5 | 100.0 | 647.5 | layout_indexing_stencil_fusion |
| 7 | timm_adv_inception_v3_infer | pointwise_a14dcfc06344 | pointwise | 353.1 | N/A | N/A | pointwise |
| 8 | timm_adv_inception_v3_infer | pointwise_a4e42370cdb9 | pointwise | 109.5 | N/A | N/A | pointwise |
| 9 | timm_adv_inception_v3_infer | mean_d0fc206717a8 | reduction | 67.3 | N/A | N/A | mean |
| 10 | timm_adv_inception_v3_infer | pointwise_e26de0a669ae | pointwise | 354.0 | N/A | N/A | pointwise |
| 11 | timm_adv_inception_v3_infer | pointwise_e5a8078b814d | pointwise | 401.2 | N/A | N/A | pointwise |
| 12 | timm_adv_inception_v3_infer | pointwise_ff13db5dc626 | pointwise | 85.0 | N/A | N/A | pointwise |

## Time by Classification

| Classification | Count | Compile (us) | Achievable (us) | Gap (us) |
|---------------|-------|-------------|-----------------|----------|
| layout_indexing_stencil_fusion | 2 | 1388.2 | 200.0 | 1188.2 |
| pointwise | 8 | 1659.4 | 1659.4 | 0.0 |
| mean | 2 | 139.9 | 139.9 | 0.0 |

## Approach B: End-to-End (Top-Down)

Full graph files available for end-to-end benchmarking:
  - /tmp/scratch_space/better_benchmark/repros/models/timm/infer/timm_adv_inception_v3_infer/full_graph_000.py

Run with: `python scripts/model_attribution.py --model <name> --bench`

NOTE: Approach B requires GPU. Without per-shape oracles, the comparison
between A and B reveals cross-graph overhead but the 'achievable' number
from A may not match the actual achievable with correct-shape oracles.

---

# Model Attribution: timm_inception_v3_infer

## Summary

- Graphs: 1
- Total partitions: 12
- Captured in corpus: 12
- With timing data: 12
- With oracle floor: 2

## Approach A: Static Accounting (Bottom-Up)

- Sum compile time (best config): 3187.5 us
- Achievable floor (min of oracle/compile): 1999.3 us
- Gap (opportunity): 1188.2 us
- Potential speedup: 1.594x

### Limitations

- Oracles measured at ONE specific shape only; model may run at different shapes
- Non-fusible ops (matmul, conv, etc.) not captured in repro corpus
- No cross-partition overhead accounted for (dispatch, memory pressure)

## Per-Partition Breakdown

| # | Graph | Repro ID | Kind | Compile (us) | Oracle (us) | Gap (us) | Classification |
|---|-------|----------|------|-------------|-------------|----------|----------------|
| 1 | timm_inception_v3_infer | pointwise_13f26d420251 | pointwise | 640.8 | 100.0 | 540.8 | layout_indexing_stencil_fusion |
| 2 | timm_inception_v3_infer | pointwise_3698b0e606e1 | pointwise | 11.7 | N/A | N/A | pointwise |
| 3 | timm_inception_v3_infer | pointwise_4f6d85385d17 | pointwise | 87.8 | N/A | N/A | pointwise |
| 4 | timm_inception_v3_infer | mean_5c93e9826aa8 | reduction | 72.6 | N/A | N/A | mean |
| 5 | timm_inception_v3_infer | pointwise_6d842b54b40d | pointwise | 257.0 | N/A | N/A | pointwise |
| 6 | timm_inception_v3_infer | pointwise_71e3a6c09140 | pointwise | 747.5 | 100.0 | 647.5 | layout_indexing_stencil_fusion |
| 7 | timm_inception_v3_infer | pointwise_a14dcfc06344 | pointwise | 353.1 | N/A | N/A | pointwise |
| 8 | timm_inception_v3_infer | pointwise_a4e42370cdb9 | pointwise | 109.5 | N/A | N/A | pointwise |
| 9 | timm_inception_v3_infer | mean_d0fc206717a8 | reduction | 67.3 | N/A | N/A | mean |
| 10 | timm_inception_v3_infer | pointwise_e26de0a669ae | pointwise | 354.0 | N/A | N/A | pointwise |
| 11 | timm_inception_v3_infer | pointwise_e5a8078b814d | pointwise | 401.2 | N/A | N/A | pointwise |
| 12 | timm_inception_v3_infer | pointwise_ff13db5dc626 | pointwise | 85.0 | N/A | N/A | pointwise |

## Time by Classification

| Classification | Count | Compile (us) | Achievable (us) | Gap (us) |
|---------------|-------|-------------|-----------------|----------|
| layout_indexing_stencil_fusion | 2 | 1388.2 | 200.0 | 1188.2 |
| pointwise | 8 | 1659.4 | 1659.4 | 0.0 |
| mean | 2 | 139.9 | 139.9 | 0.0 |

## Approach B: End-to-End (Top-Down)

Full graph files available for end-to-end benchmarking:
  - /tmp/scratch_space/better_benchmark/repros/models/timm/infer/timm_inception_v3_infer/full_graph_000.py

Run with: `python scripts/model_attribution.py --model <name> --bench`

NOTE: Approach B requires GPU. Without per-shape oracles, the comparison
between A and B reveals cross-graph overhead but the 'achievable' number
from A may not match the actual achievable with correct-shape oracles.

---

# Model Attribution: torchbench_resnet18_train

## Summary

- Graphs: 2
- Total partitions: 12
- Captured in corpus: 12
- With timing data: 12
- With oracle floor: 1

## Approach A: Static Accounting (Bottom-Up)

- Sum compile time (best config): 4054.4 us
- Achievable floor (min of oracle/compile): 3060.7 us
- Gap (opportunity): 993.6 us
- Potential speedup: 1.325x

### Limitations

- Oracles measured at ONE specific shape only; model may run at different shapes
- Non-fusible ops (matmul, conv, etc.) not captured in repro corpus
- No cross-partition overhead accounted for (dispatch, memory pressure)

## Per-Partition Breakdown

| # | Graph | Repro ID | Kind | Compile (us) | Oracle (us) | Gap (us) | Classification |
|---|-------|----------|------|-------------|-------------|----------|----------------|
| 1 | torchbench_resnet18_train_000 | pointwise_100a39b686e3 | pointwise | 5.1 | N/A | N/A | pointwise |
| 2 | torchbench_resnet18_train_000 | var_mean_mean_188f42114f68 | reduction | 11.1 | N/A | N/A | var_mean_mean |
| 3 | torchbench_resnet18_train_000 | var_mean_323ce31f395b | reduction | 46.9 | N/A | N/A | var_mean |
| 4 | torchbench_resnet18_train_000 | var_mean_598830735cf6 | reduction | 163.7 | N/A | N/A | var_mean |
| 5 | torchbench_resnet18_train_000 | var_mean_65e90900fd65 | reduction | 12.0 | N/A | N/A | var_mean |
| 6 | torchbench_resnet18_train_000 | var_mean_var_mean_de5c360021d3 | reduction | 48.9 | N/A | N/A | var_mean_var_mean |
| 7 | torchbench_resnet18_train_001 | sum_22b5cd24890b | reduction | 6.1 | N/A | N/A | sum |
| 8 | torchbench_resnet18_train_001 | sum_sum_25d0782111b0 | reduction | 20.3 | N/A | N/A | sum_sum |
| 9 | torchbench_resnet18_train_001 | sum_sum_sum_3579253dcf89 | reduction | 778.1 | N/A | N/A | sum_sum_sum |
| 10 | torchbench_resnet18_train_001 | sum_sum_3c5efeeb4188 | reduction | 9.0 | N/A | N/A | sum_sum |
| 11 | torchbench_resnet18_train_001 | sum_sum_a55455ad36bb | reduction | 466.8 | N/A | N/A | sum_sum |
| 12 | torchbench_resnet18_train_001 | sum_sum_cdaed89f373c | reduction | 2486.2 | 1492.6 | 993.6 | multi_output_reduction_templates |

## Time by Classification

| Classification | Count | Compile (us) | Achievable (us) | Gap (us) |
|---------------|-------|-------------|-----------------|----------|
| multi_output_reduction_templates | 1 | 2486.2 | 1492.6 | 993.6 |
| pointwise | 1 | 5.1 | 5.1 | 0.0 |
| var_mean_mean | 1 | 11.1 | 11.1 | 0.0 |
| var_mean | 3 | 222.7 | 222.7 | 0.0 |
| var_mean_var_mean | 1 | 48.9 | 48.9 | 0.0 |
| sum | 1 | 6.1 | 6.1 | 0.0 |
| sum_sum | 3 | 496.1 | 496.1 | 0.0 |
| sum_sum_sum | 1 | 778.1 | 778.1 | 0.0 |

## Approach B: End-to-End (Top-Down)

No full_graph_*.py files found for this model.
End-to-end timing not available.

---

# Model Attribution: torchbench_squeezenet1_1_train

## Summary

- Graphs: 2
- Total partitions: 13
- Captured in corpus: 13
- With timing data: 13
- With oracle floor: 2

## Approach A: Static Accounting (Bottom-Up)

- Sum compile time (best config): 4784.0 us
- Achievable floor (min of oracle/compile): 4059.0 us
- Gap (opportunity): 725.0 us
- Potential speedup: 1.179x

### Limitations

- Oracles measured at ONE specific shape only; model may run at different shapes
- Non-fusible ops (matmul, conv, etc.) not captured in repro corpus
- No cross-partition overhead accounted for (dispatch, memory pressure)

## Per-Partition Breakdown

| # | Graph | Repro ID | Kind | Compile (us) | Oracle (us) | Gap (us) | Classification |
|---|-------|----------|------|-------------|-------------|----------|----------------|
| 1 | torchbench_squeezenet1_1_train_000 | pointwise_3abc926270f6 | pointwise | 5.9 | N/A | N/A | pointwise |
| 2 | torchbench_squeezenet1_1_train_000 | mean_3b7ac433d31c | reduction | 75.6 | N/A | N/A | mean |
| 3 | torchbench_squeezenet1_1_train_000 | pointwise_5ae8759ec547 | pointwise | 300.9 | N/A | N/A | pointwise |
| 4 | torchbench_squeezenet1_1_train_000 | pointwise_70e5a4aca4b5 | pointwise | 64.9 | N/A | N/A | pointwise |
| 5 | torchbench_squeezenet1_1_train_000 | pointwise_7f55bac8afd0 | pointwise | 1081.3 | N/A | N/A | pointwise |
| 6 | torchbench_squeezenet1_1_train_000 | pointwise_c3360ba4828a | pointwise | 201.7 | N/A | N/A | pointwise |
| 7 | torchbench_squeezenet1_1_train_001 | sum_0becf9609ad7 | reduction | 85.8 | N/A | N/A | sum |
| 8 | torchbench_squeezenet1_1_train_001 | sum_18262b26934c | reduction | 1304.5 | 579.5 | 725.0 | structured_pool_upsample_backward_reduce |
| 9 | torchbench_squeezenet1_1_train_001 | sum_sum_3219a09ab96a | reduction | 326.4 | N/A | N/A | sum_sum |
| 10 | torchbench_squeezenet1_1_train_001 | sum_399b79c70c7b | reduction | 83.8 | N/A | N/A | sum |
| 11 | torchbench_squeezenet1_1_train_001 | sum_sum_8bcd6e12dcd4 | reduction | 782.4 | 5004.9 | 0.0 | structured_pool_upsample_backward_reduce |
| 12 | torchbench_squeezenet1_1_train_001 | sum_sum_baa6d8522274 | reduction | 79.7 | N/A | N/A | sum_sum |
| 13 | torchbench_squeezenet1_1_train_001 | sum_sum_ced249279c9d | reduction | 391.1 | N/A | N/A | sum_sum |

## Time by Classification

| Classification | Count | Compile (us) | Achievable (us) | Gap (us) |
|---------------|-------|-------------|-----------------|----------|
| structured_pool_upsample_backward_reduce | 2 | 2086.9 | 1361.9 | 725.0 |
| pointwise | 5 | 1654.8 | 1654.8 | 0.0 |
| mean | 1 | 75.6 | 75.6 | 0.0 |
| sum | 2 | 169.6 | 169.6 | 0.0 |
| sum_sum | 3 | 797.2 | 797.2 | 0.0 |

## Approach B: End-to-End (Top-Down)

No full_graph_*.py files found for this model.
End-to-end timing not available.

---

# Model Attribution: torchbench_squeezenet1_1_infer

## Summary

- Graphs: 1
- Total partitions: 5
- Captured in corpus: 5
- With timing data: 5
- With oracle floor: 1

## Approach A: Static Accounting (Bottom-Up)

- Sum compile time (best config): 1235.7 us
- Achievable floor (min of oracle/compile): 539.3 us
- Gap (opportunity): 696.4 us
- Potential speedup: 2.291x

### Limitations

- Oracles measured at ONE specific shape only; model may run at different shapes
- Non-fusible ops (matmul, conv, etc.) not captured in repro corpus
- No cross-partition overhead accounted for (dispatch, memory pressure)

## Per-Partition Breakdown

| # | Graph | Repro ID | Kind | Compile (us) | Oracle (us) | Gap (us) | Classification |
|---|-------|----------|------|-------------|-------------|----------|----------------|
| 1 | torchbench_squeezenet1_1_infer_000 | pointwise_0a306c604828 | pointwise | 74.7 | N/A | N/A | pointwise |
| 2 | torchbench_squeezenet1_1_infer_000 | pointwise_3abc926270f6 | pointwise | 5.9 | N/A | N/A | pointwise |
| 3 | torchbench_squeezenet1_1_infer_000 | mean_432300e645a8 | reduction | 46.6 | N/A | N/A | mean |
| 4 | torchbench_squeezenet1_1_infer_000 | pointwise_485ad74a8b0e | pointwise | 796.4 | 100.0 | 696.4 | layout_indexing_stencil_fusion |
| 5 | torchbench_squeezenet1_1_infer_000 | pointwise_f9c1d1b08ddb | pointwise | 312.1 | N/A | N/A | pointwise |

## Time by Classification

| Classification | Count | Compile (us) | Achievable (us) | Gap (us) |
|---------------|-------|-------------|-----------------|----------|
| layout_indexing_stencil_fusion | 1 | 796.4 | 100.0 | 696.4 |
| pointwise | 3 | 392.7 | 392.7 | 0.0 |
| mean | 1 | 46.6 | 46.6 | 0.0 |

## Approach B: End-to-End (Top-Down)

No full_graph_*.py files found for this model.
End-to-end timing not available.

---

# Model Attribution: torchbench_hf_T5_base_infer

## Summary

- Graphs: 1
- Total partitions: 11
- Captured in corpus: 11
- With timing data: 11
- With oracle floor: 1

## Approach A: Static Accounting (Bottom-Up)

- Sum compile time (best config): 600.1 us
- Achievable floor (min of oracle/compile): 571.3 us
- Gap (opportunity): 28.8 us
- Potential speedup: 1.050x

### Limitations

- Oracles measured at ONE specific shape only; model may run at different shapes
- Non-fusible ops (matmul, conv, etc.) not captured in repro corpus
- No cross-partition overhead accounted for (dispatch, memory pressure)

## Per-Partition Breakdown

| # | Graph | Repro ID | Kind | Compile (us) | Oracle (us) | Gap (us) | Classification |
|---|-------|----------|------|-------------|-------------|----------|----------------|
| 1 | torchbench_hf_T5_base_infer_000 | any_mean_33b10ff8a837 | reduction | 11.2 | N/A | N/A | any_mean |
| 2 | torchbench_hf_T5_base_infer_000 | any_mean_417d9efb98a0 | reduction | 11.9 | N/A | N/A | any_mean |
| 3 | torchbench_hf_T5_base_infer_000 | pointwise_4498011749b4 | pointwise | 45.0 | N/A | N/A | pointwise |
| 4 | torchbench_hf_T5_base_infer_000 | amax_sum_4bd27b112605 | reduction | 135.1 | N/A | N/A | amax_sum |
| 5 | torchbench_hf_T5_base_infer_000 | amax_sum_4fb4f9a0bf43 | reduction | 67.4 | 38.6 | 28.8 | amax_sum |
| 6 | torchbench_hf_T5_base_infer_000 | amax_sum_5f0c26b7e967 | reduction | 183.2 | N/A | N/A | amax_sum |
| 7 | torchbench_hf_T5_base_infer_000 | any_mean_6b05ef27be02 | reduction | 11.1 | N/A | N/A | any_mean |
| 8 | torchbench_hf_T5_base_infer_000 | amax_sum_a7ede60031ef | reduction | 91.0 | N/A | N/A | amax_sum |
| 9 | torchbench_hf_T5_base_infer_000 | pointwise_af2e722aed5d | pointwise | 24.8 | N/A | N/A | pointwise |
| 10 | torchbench_hf_T5_base_infer_000 | mean_fb3bbfd51007 | reduction | 7.6 | N/A | N/A | mean |
| 11 | torchbench_hf_T5_base_infer_000 | any_mean_fde44ec669a9 | reduction | 11.8 | N/A | N/A | any_mean |

## Time by Classification

| Classification | Count | Compile (us) | Achievable (us) | Gap (us) |
|---------------|-------|-------------|-----------------|----------|
| amax_sum | 4 | 476.7 | 447.9 | 28.8 |
| any_mean | 4 | 46.0 | 46.0 | 0.0 |
| pointwise | 2 | 69.8 | 69.8 | 0.0 |
| mean | 1 | 7.6 | 7.6 | 0.0 |

## Approach B: End-to-End (Top-Down)

No full_graph_*.py files found for this model.
End-to-end timing not available.

---

