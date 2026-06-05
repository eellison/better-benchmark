# amax_sum_sum_e2f518f0a274

## Compile: 157.28us, Oracle: 91.58us, Gap: 1.717x

## Diagnosis: NEW_PATTERN (fused online cross-entropy with masked gather)

## Root cause: Inductor emits 2 kernels for the GPT-2 shifted-label cross-entropy-mean: (1) a fused reduction computing the full [2048, 50257] log-softmax via online max/sum, gather of the target log-probability, and masked-loss accumulation, and (2) a scalar reduction dividing total loss by valid count. The oracle uses a dedicated per-row online cross-entropy kernel that never materializes the full log-softmax matrix -- it streams through the 50257-element rows in blocks of 4096, maintaining running (max, sum_exp) accumulators and directly computing loss = logsumexp - target_logit for each row, then sums losses and valid counts in a small final reduction. The key performance difference is that Inductor's fused kernel still reads the full [2048, 50257] logits matrix twice (once for amax, once for softmax) because its online softmax template does a two-pass approach, while the oracle achieves true single-pass online logsumexp.

## Fix path: Add an Inductor pattern/lowering for shifted-label ignore-index cross entropy that recognizes the decomposed constant_pad/slice/amax/sub/exp/sum/log/sub/ne/where/gather/neg/where/sum/sum/div graph as cross-entropy-mean, and emits a single-pass online logsumexp kernel that: (1) reads each row in blocks accumulating (max, sum_exp), (2) gathers the target logit inline, (3) computes loss = log(sum_exp) + max - target_logit, (4) masks ignore-index positions, and (5) reduces loss/count across rows. This is a well-known optimization (FlashAttention-style online reduction applied to cross-entropy).

## Status: design_todo

## Details

- Model: torchbench_hf_GPT2_train_001, torchbench_hf_GPT2_large_train_001
- Pattern: amax+sum+sum+sum reduction (shifted causal LM cross-entropy mean)
- Shape: [4, 512] labels + [4, 512, 50257] logits -> scalar loss
- Row width: 50257 (vocab size) -- too large for persistent, requires chunked approach
- Inductor kernels: 2 (fused online softmax + scalar reduction)
- Oracle kernels: 2 (per-row online xent with block_n=4096 + mean reduction)
- Config exploration: coordinate_descent_tuning (157.7us), combo+multi3 (157.6us) -- no config improvement.
- The oracle achieves true single-pass online computation (one read of logits), while Inductor's template appears to read logits twice.
- File references: /tmp/pytorch-work/torch/_inductor/fx_passes/post_grad.py (pattern registration), /tmp/pytorch-work/torch/_inductor/codegen/triton.py (reduction template)
