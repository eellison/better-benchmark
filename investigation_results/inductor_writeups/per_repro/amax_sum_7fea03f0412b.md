# amax_sum_7fea03f0412b

## Classification: `ALREADY_FIXED`

## Pattern

Masked attention softmax + dropout + permute (BERT)

- Model: torchbench_BERT_pytorch_train
- Shape: mask [128, 128] i64, bmm [1536, 128, 128] f32 (viewed as [128, 12, 128, 128])
- Reduction dim (rnumel): 128
- Output: f32[1536, 128, 128] (permuted masked softmax+dropout result)

## Measurements (NVIDIA B200, CD tuning ON)

| Metric | Value |
|--------|-------|
| Compile (best) | 113 us |
| Oracle (persistent masked softmax+dropout) | 153 us |
| Gap (compile/oracle) | 0.74x |
| Kernel count | 1 |
| scalar_reduction_accumulators effect | None (rnumel=128, persistent) |

## Diagnosis

Inductor generates a **single kernel** and the compile (113 us) is **faster** than the oracle (153 us). The gap is < 1x, meaning inductor already produces optimal code for this pattern.

With rnumel=128, this is a small-rnumel case. The reduction is fully persistent (fits in registers). The `scalar_reduction_accumulators` optimization does not apply because there is no reduction loop to optimize.

The oracle achieves slightly worse performance because:
- Its block_k=128 persistent kernel may have suboptimal occupancy on B200
- Inductor's auto-tuned kernel selects better launch configuration for the hardware
- The mask expansion (unsqueeze + repeat) is handled more efficiently by inductor's fused approach

## Inductor Closure

No action needed. Compile already beats the oracle by 1.35x. Inductor's single-kernel persistent reduction with fused masked softmax + dropout + permute is optimal for this small-rnumel (128) case on B200.
