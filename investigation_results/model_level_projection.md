# Model-Level Performance Projection

## Overall Summary

- **Total models analyzed**: 184
- **Total repros with timing data**: 1134
- **Total compile time (sum across all models)**: 217615.7 us
- **Total achievable time (best of compile/oracle)**: 161108.0 us
- **Total gap (opportunity)**: 56507.7 us
- **Overall potential speedup**: 1.351x
- **Overall time savings**: 26.0%

## Top 20 Models by Potential Speedup

| Rank | Model | Repros | Compile (us) | Achievable (us) | Speedup | Gap (us) |
|------|-------|--------|--------------|-----------------|---------|----------|
| 1 | torchbench_pyhpc_equation_of_state_infer | 1 | 83.9 | 14.9 | 5.622x | 69.0 |
| 2 | torchbench_hf_T5_train | 13 | 8479.6 | 1600.5 | 5.298x | 6879.1 |
| 3 | torchbench_pyhpc_isoneutral_mixing_infer | 1 | 939.8 | 211.9 | 4.435x | 727.9 |
| 4 | torchbench_pyhpc_turbulent_kinetic_energy_infer | 1 | 548.8 | 153.7 | 3.570x | 395.1 |
| 5 | torchbench_shufflenet_v2_x1_0_infer | 3 | 185.0 | 63.1 | 2.933x | 121.9 |
| 6 | hf_GPTJForQuestionAnswering_train | 2 | 36.3 | 13.1 | 2.760x | 23.1 |
| 7 | genai_CrossEntropyForward | 1 | 2224.0 | 956.2 | 2.326x | 1267.8 |
| 8 | torchbench_resnet152_train | 2 | 343.6 | 161.0 | 2.135x | 182.6 |
| 9 | torchbench_resnet152_infer | 4 | 129.8 | 63.8 | 2.034x | 66.0 |
| 10 | hf_AllenaiLongformerBase_infer | 9 | 1344.4 | 678.3 | 1.982x | 666.1 |
| 11 | genai_SoftmaxBackward | 2 | 6242.3 | 3326.6 | 1.876x | 2915.7 |
| 12 | torchbench_vgg16_train | 10 | 2324.6 | 1266.6 | 1.835x | 1058.0 |
| 13 | torchbench_densenet121_infer | 11 | 423.1 | 230.6 | 1.835x | 192.5 |
| 14 | hf_AllenaiLongformerBase_train | 27 | 7497.2 | 4397.0 | 1.705x | 3100.2 |
| 15 | torchbench_timm_vovnet_infer | 2 | 47.0 | 27.7 | 1.700x | 19.4 |
| 16 | timm_mobilenetv2_100_train | 8 | 652.1 | 385.8 | 1.690x | 266.3 |
| 17 | torchbench_resnet18_train | 1 | 2517.0 | 1492.6 | 1.686x | 1024.5 |
| 18 | torchbench_resnet50_train | 1 | 664.6 | 398.4 | 1.668x | 266.2 |
| 19 | timm_mobilenetv3_large_100_train | 11 | 2118.3 | 1288.1 | 1.644x | 830.2 |
| 20 | torchbench_hf_Longformer_train | 16 | 1008.1 | 613.4 | 1.643x | 394.6 |

## Detailed Breakdown (Top 20 Models)

### 1. torchbench_pyhpc_equation_of_state_infer

- **Total compile time**: 83.9 us
- **Achievable time**: 14.9 us
- **Potential speedup**: 5.622x
- **Repros with data**: 1 / 1
- **Total gap**: 69.0 us

**Top 3 Bottleneck Repros:**

| Repro | Gap (us) | Compile (us) | Oracle (us) | Classification |
|-------|----------|--------------|-------------|----------------|
| pointwise_a2382a85ee99 | 69.0 | 83.9 | 14.9 | pointwise_expression_heuristic |

### 2. torchbench_hf_T5_train

- **Total compile time**: 8479.6 us
- **Achievable time**: 1600.5 us
- **Potential speedup**: 5.298x
- **Repros with data**: 13 / 14
- **Total gap**: 6879.1 us

**Top 3 Bottleneck Repros:**

| Repro | Gap (us) | Compile (us) | Oracle (us) | Classification |
|-------|----------|--------------|-------------|----------------|
| sum_sum_sum_afd118ca907d | 5913.3 | 6364.2 | 450.9 | fused_softmax_backward_attention_backward |
| amax_sum_amax_68fa105ccaf0 | 563.9 | 804.9 | 241.0 | online_softmax_ce_heuristic |
| amax_sum_557afc9e24ea | 194.2 | 326.4 | 132.2 | online_softmax_ce_heuristic |

### 3. torchbench_pyhpc_isoneutral_mixing_infer

- **Total compile time**: 939.8 us
- **Achievable time**: 211.9 us
- **Potential speedup**: 4.435x
- **Repros with data**: 1 / 2
- **Total gap**: 727.9 us

**Top 3 Bottleneck Repros:**

| Repro | Gap (us) | Compile (us) | Oracle (us) | Classification |
|-------|----------|--------------|-------------|----------------|
| pointwise_70c0751eb408 | 727.9 | 939.8 | 211.9 | layout_stencil_functional_update_heuristic |

### 4. torchbench_pyhpc_turbulent_kinetic_energy_infer

- **Total compile time**: 548.8 us
- **Achievable time**: 153.7 us
- **Potential speedup**: 3.570x
- **Repros with data**: 1 / 2
- **Total gap**: 395.1 us

**Top 3 Bottleneck Repros:**

| Repro | Gap (us) | Compile (us) | Oracle (us) | Classification |
|-------|----------|--------------|-------------|----------------|
| pointwise_531d72f1b34a | 395.1 | 548.8 | 153.7 | layout_stencil_functional_update_heuristic |

### 5. torchbench_shufflenet_v2_x1_0_infer

- **Total compile time**: 185.0 us
- **Achievable time**: 63.1 us
- **Potential speedup**: 2.933x
- **Repros with data**: 3 / 3
- **Total gap**: 121.9 us

**Top 3 Bottleneck Repros:**

| Repro | Gap (us) | Compile (us) | Oracle (us) | Classification |
|-------|----------|--------------|-------------|----------------|
| pointwise_cf3acd87ba9e | 55.7 | 76.7 | 21.0 | pointwise_expression_heuristic |
| pointwise_2eae25328292 | 33.2 | 54.2 | 21.1 | pointwise_expression_heuristic |
| pointwise_e4cfa8694326 | 33.1 | 54.1 | 21.0 | pointwise_expression_heuristic |

### 6. hf_GPTJForQuestionAnswering_train

- **Total compile time**: 36.3 us
- **Achievable time**: 13.1 us
- **Potential speedup**: 2.760x
- **Repros with data**: 2 / 2
- **Total gap**: 23.1 us

**Top 3 Bottleneck Repros:**

| Repro | Gap (us) | Compile (us) | Oracle (us) | Classification |
|-------|----------|--------------|-------------|----------------|
| sum_sum_sum_e3016fe2bbb7 | 19.2 | 26.1 | 6.9 | multi_output_reduction_templates |
| amax_sum_sum_2a28fce37306 | 3.9 | 10.1 | 6.2 | online_softmax_cross_entropy |

### 7. genai_CrossEntropyForward

- **Total compile time**: 2224.0 us
- **Achievable time**: 956.2 us
- **Potential speedup**: 2.326x
- **Repros with data**: 1 / 1
- **Total gap**: 1267.8 us

**Top 3 Bottleneck Repros:**

| Repro | Gap (us) | Compile (us) | Oracle (us) | Classification |
|-------|----------|--------------|-------------|----------------|
| amax_sum_f0661488d68c | 1267.8 | 2224.0 | 956.2 | online_softmax_cross_entropy |

### 8. torchbench_resnet152_train

- **Total compile time**: 343.6 us
- **Achievable time**: 161.0 us
- **Potential speedup**: 2.135x
- **Repros with data**: 2 / 2
- **Total gap**: 182.6 us

**Top 3 Bottleneck Repros:**

| Repro | Gap (us) | Compile (us) | Oracle (us) | Classification |
|-------|----------|--------------|-------------|----------------|
| var_mean_598830735cf6 | 107.9 | 163.7 | 55.9 | norm_template_heuristic |
| sum_sum_2847f46e0c5a | 74.8 | 179.9 | 105.1 | structured_scatter_reduce_heuristic |

### 9. torchbench_resnet152_infer

- **Total compile time**: 129.8 us
- **Achievable time**: 63.8 us
- **Potential speedup**: 2.034x
- **Repros with data**: 4 / 4
- **Total gap**: 66.0 us

**Top 3 Bottleneck Repros:**

| Repro | Gap (us) | Compile (us) | Oracle (us) | Classification |
|-------|----------|--------------|-------------|----------------|
| pointwise_f2df04089ff4 | 58.6 | 83.7 | 25.1 | pointwise_expression_heuristic |
| pointwise_57287c224f10 | 6.7 | 20.0 | 13.3 | pointwise_expression_heuristic |
| pointwise_f2b2f0c4db0c | 0.6 | 13.8 | 13.2 | pointwise_expression_heuristic |

### 10. hf_AllenaiLongformerBase_infer

- **Total compile time**: 1344.4 us
- **Achievable time**: 678.3 us
- **Potential speedup**: 1.982x
- **Repros with data**: 9 / 21
- **Total gap**: 666.1 us

**Top 3 Bottleneck Repros:**

| Repro | Gap (us) | Compile (us) | Oracle (us) | Classification |
|-------|----------|--------------|-------------|----------------|
| amax_sum_68fe981b18dd | 470.5 | 665.5 | 195.1 | online_softmax_ce_heuristic |
| amax_sum_sum_d85e67b00643 | 195.4 | 556.8 | 361.4 | online_softmax_ce_heuristic |
| any_17b234fd7f4f | 0.2 | 7.1 | 7.0 | bandwidth_bound_reduction |

### 11. genai_SoftmaxBackward

- **Total compile time**: 6242.3 us
- **Achievable time**: 3326.6 us
- **Potential speedup**: 1.876x
- **Repros with data**: 2 / 2
- **Total gap**: 2915.7 us

**Top 3 Bottleneck Repros:**

| Repro | Gap (us) | Compile (us) | Oracle (us) | Classification |
|-------|----------|--------------|-------------|----------------|
| amax_sum_sum_dc96a87ba8db | 1619.0 | 3257.3 | 1638.4 | online_softmax_cross_entropy |
| sum_e00c7291b6ee | 1296.7 | 2985.0 | 1688.3 | multi_output_reduction_heuristic |

### 12. torchbench_vgg16_train

- **Total compile time**: 2324.6 us
- **Achievable time**: 1266.6 us
- **Potential speedup**: 1.835x
- **Repros with data**: 10 / 10
- **Total gap**: 1058.0 us

**Top 3 Bottleneck Repros:**

| Repro | Gap (us) | Compile (us) | Oracle (us) | Classification |
|-------|----------|--------------|-------------|----------------|
| sum_34d3dca078b3 | 756.7 | 1062.8 | 306.1 | structured_pool_upsample_backward_reduce |
| sum_14fe7b321763 | 182.0 | 276.6 | 94.6 | structured_scatter_reduce_heuristic |
| sum_8a66186d1ffe | 85.9 | 147.7 | 61.8 | structured_scatter_reduce_heuristic |

### 13. torchbench_densenet121_infer

- **Total compile time**: 423.1 us
- **Achievable time**: 230.6 us
- **Potential speedup**: 1.835x
- **Repros with data**: 11 / 11
- **Total gap**: 192.5 us

**Top 3 Bottleneck Repros:**

| Repro | Gap (us) | Compile (us) | Oracle (us) | Classification |
|-------|----------|--------------|-------------|----------------|
| pointwise_437415a3398d | 170.2 | 291.8 | 121.6 | SCHEDULER_FUSION |
| pointwise_1383c536f620 | 6.0 | 18.4 | 12.4 | pointwise_expression_heuristic |
| pointwise_a5ce2f1040ee | 4.9 | 16.2 | 11.3 | pointwise_expression_heuristic |

### 14. hf_AllenaiLongformerBase_train

- **Total compile time**: 7497.2 us
- **Achievable time**: 4397.0 us
- **Potential speedup**: 1.705x
- **Repros with data**: 27 / 31
- **Total gap**: 3100.2 us

**Top 3 Bottleneck Repros:**

| Repro | Gap (us) | Compile (us) | Oracle (us) | Classification |
|-------|----------|--------------|-------------|----------------|
| amax_sum_4c524f75213e | 668.3 | 867.6 | 199.3 | online_softmax_ce_heuristic |
| amax_sum_87e1fb077f24 | 630.9 | 898.9 | 268.0 | online_softmax_ce_heuristic |
| sum_sum_74337e6f6544 | 625.7 | 2121.6 | 1495.9 | multi_output_reduction_heuristic |

### 15. torchbench_timm_vovnet_infer

- **Total compile time**: 47.0 us
- **Achievable time**: 27.7 us
- **Potential speedup**: 1.700x
- **Repros with data**: 2 / 2
- **Total gap**: 19.4 us

**Top 3 Bottleneck Repros:**

| Repro | Gap (us) | Compile (us) | Oracle (us) | Classification |
|-------|----------|--------------|-------------|----------------|
| pointwise_976768b988b8 | 11.2 | 23.6 | 12.3 | pointwise_expression_heuristic |
| pointwise_c0e132c66e3b | 8.1 | 23.4 | 15.3 | pointwise_expression_heuristic |

### 16. timm_mobilenetv2_100_train

- **Total compile time**: 652.1 us
- **Achievable time**: 385.8 us
- **Potential speedup**: 1.690x
- **Repros with data**: 8 / 8
- **Total gap**: 266.3 us

**Top 3 Bottleneck Repros:**

| Repro | Gap (us) | Compile (us) | Oracle (us) | Classification |
|-------|----------|--------------|-------------|----------------|
| sum_sum_d550f2d61c6c | 159.8 | 331.5 | 171.7 | SCHEDULER_FUSION |
| sum_sum_67d7103962e7 | 91.7 | 140.3 | 48.6 | multi_output_reduction_heuristic |
| sum_sum_6f518556b3ea | 12.3 | 42.0 | 29.6 | multi_output_reduction_heuristic |

### 17. torchbench_resnet18_train

- **Total compile time**: 2517.0 us
- **Achievable time**: 1492.6 us
- **Potential speedup**: 1.686x
- **Repros with data**: 1 / 1
- **Total gap**: 1024.5 us

**Top 3 Bottleneck Repros:**

| Repro | Gap (us) | Compile (us) | Oracle (us) | Classification |
|-------|----------|--------------|-------------|----------------|
| sum_sum_cdaed89f373c | 1024.5 | 2517.0 | 1492.6 | multi_output_reduction_templates |

### 18. torchbench_resnet50_train

- **Total compile time**: 664.6 us
- **Achievable time**: 398.4 us
- **Potential speedup**: 1.668x
- **Repros with data**: 1 / 1
- **Total gap**: 266.2 us

**Top 3 Bottleneck Repros:**

| Repro | Gap (us) | Compile (us) | Oracle (us) | Classification |
|-------|----------|--------------|-------------|----------------|
| sum_sum_145a8857ebf2 | 266.2 | 664.6 | 398.4 | SCATTER_REDUCE |

### 19. timm_mobilenetv3_large_100_train

- **Total compile time**: 2118.3 us
- **Achievable time**: 1288.1 us
- **Potential speedup**: 1.644x
- **Repros with data**: 11 / 13
- **Total gap**: 830.2 us

**Top 3 Bottleneck Repros:**

| Repro | Gap (us) | Compile (us) | Oracle (us) | Classification |
|-------|----------|--------------|-------------|----------------|
| sum_sum_a3da3e05e5db | 300.7 | 692.0 | 391.3 | multi_output_reduction_heuristic |
| sum_sum_7388c7a6f044 | 170.3 | 440.1 | 269.7 | multi_output_reduction_heuristic |
| sum_sum_ee85624361a0 | 107.1 | 237.7 | 130.6 | SCHEDULER_FUSION |

### 20. torchbench_hf_Longformer_train

- **Total compile time**: 1008.1 us
- **Achievable time**: 613.4 us
- **Potential speedup**: 1.643x
- **Repros with data**: 16 / 17
- **Total gap**: 394.6 us

**Top 3 Bottleneck Repros:**

| Repro | Gap (us) | Compile (us) | Oracle (us) | Classification |
|-------|----------|--------------|-------------|----------------|
| sum_5a4992885bd6 | 130.8 | 199.7 | 68.9 | multi_output_reduction_heuristic |
| amax_sum_67d7c2666a5c | 128.4 | 246.8 | 118.4 | online_softmax_ce_heuristic |
| amax_sum_93cb2fd0355b | 107.9 | 229.3 | 121.4 | online_softmax_ce_heuristic |

## Classification Summary (across all models)

Root causes of performance gaps, ranked by total opportunity.

| Classification | Count | Total Gap (us) |
|---------------|-------|----------------|
| multi_output_reduction_heuristic | 116 | 11258.5 |
| online_softmax_ce_heuristic | 50 | 8087.3 |
| fused_softmax_backward_attention_backward | 1 | 5913.3 |
| online_softmax_cross_entropy | 8 | 4335.6 |
| structured_scatter_reduce_heuristic | 25 | 3249.2 |
| multi_output_reduction_templates | 5 | 3196.0 |
| structured_pool_upsample_backward_reduce | 4 | 2794.1 |
| SCATTER_REDUCE | 20 | 2668.2 |
| pointwise_expression_heuristic | 75 | 2521.4 |
| norm_template_canonicalization | 2 | 2238.5 |
| norm_template_heuristic | 82 | 1903.0 |
| BANDWIDTH_BOUND | 6 | 1665.6 |
| SCHEDULER_FUSION | 15 | 1137.0 |
| layout_stencil_functional_update_heuristic | 2 | 1123.0 |
| COOPERATIVE_SPLIT_K | 14 | 878.5 |
| ALGEBRAIC_ELIMINATION | 2 | 759.6 |
| NEW_PATTERN | 8 | 337.0 |
| scatter_reduce (structured relative-position scatter) | 1 | 279.3 |
| RECOMPUTE_FUSION | 1 | 278.2 |
| UNKNOWN | 16 | 210.9 |
| SCATTER_REDUCE / SCHEDULER_FUSION | 1 | 71.7 |
| ONLINE_CROSS_ENTROPY (codegen template overhead in looped online softmax) | 1 | 65.3 |
| BANDWIDTH_BOUND / COOPERATIVE_SPLIT_K | 1 | 64.5 |
| scatter_reduce (embedding-backward scatter-reduce) | 1 | 50.8 |
| scatter_reduce (BERT-large embedding/layernorm-backward) | 1 | 44.6 |

## Models Already at Floor (51 models)

These models have potential speedup <= 1.01x (compile matches oracle).

- timm_visformer_small_train: 1.007x (5 repros)
- hf_AlbertForMaskedLM_train: 1.006x (18 repros)
- torchbench_hf_Bart_infer: 1.005x (6 repros)
- torchbench_llama_infer: 1.004x (5 repros)
- torchbench_vgg16_infer: 1.004x (2 repros)
- torchbench_hf_Roberta_base_train: 1.003x (2 repros)
- torchbench_moco_infer: 1.003x (7 repros)
- hf_GPTNeoForCausalLM_infer: 1.000x (3 repros)
- timm_visformer_small_infer: 1.000x (3 repros)
- torchbench_BERT_pytorch_infer: 1.000x (2 repros)
