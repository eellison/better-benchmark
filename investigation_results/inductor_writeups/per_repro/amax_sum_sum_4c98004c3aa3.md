# amax_sum_sum_4c98004c3aa3

## Classification: `ALREADY_FIXED`

## Pattern

Cross-entropy loss with log-softmax: amax + exp + sum + log + gather + masked reduction + mean

- Model: hf_MegatronBertForCausalLM_infer
- Shape: logits [8192, 29056] f32, tokens [16, 512] i64
- Reduction dim (rnumel): 29056 (large vocab)
- Output: scalar f32 (mean cross-entropy loss)

## Measurements (NVIDIA B200, CD tuning ON)

| Metric | Value |
|--------|-------|
| Compile (best) | 224 us |
| Oracle (online softmax+xent, tiled) | 169 us |
| Gap (compile/oracle) | 1.33x |
| Kernel count | 2 |
| scalar_reduction_accumulators effect | **Significant**: 306 us (OFF) vs 225 us (ON) = 1.36x speedup |

## Diagnosis

This is a **large-rnumel** case (29056) where `scalar_reduction_accumulators` provides a real benefit. The config is already ON by default (enabled since it was merged), reducing compile time from 306 us to 225 us.

The remaining 1.33x gap (224 us vs 169 us) comes from:
- The oracle uses a single fused online softmax + cross-entropy kernel that computes log_softmax, gathers the target logprob, and reduces in one pass over the vocab dimension
- Inductor generates 2 kernels: (1) fused amax+sub+exp+sum+log+sub for the log-softmax rows, (2) gather+neg+where+sum+div for the loss reduction
- The 2-kernel approach re-reads the log-softmax intermediate from HBM for the gather, while the oracle keeps it in registers

### Scalar accumulators impact

With `scalar_reduction_accumulators=True` (default), the inner reduction loop over rnumel=29056 accumulates into scalars instead of keeping a full R0_BLOCK tile alive, reducing register pressure and enabling larger R0_BLOCK configs. This is the expected benefit for large-rnumel reductions.

### Remaining gap

The 1.33x residual gap after scalar_acc is from the 2-kernel boundary. Closing it requires fusing the gather/loss computation into the softmax reduction kernel (online cross-entropy template).

## Inductor Closure

- `scalar_reduction_accumulators`: already ON and effective (1.36x improvement over OFF).
- Remaining closure path: online cross-entropy template that fuses softmax + gather + loss in one pass.
- Priority: low-moderate (1.33x gap, 55 us absolute, but the key optimization is already deployed).
