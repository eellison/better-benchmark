# amax_sum_sum_6939e2db29e3

## Classification: `ALREADY_FIXED`

## Pattern

Cross-entropy loss with log-softmax (training variant): amax + exp + sum + log + gather + masked reduction + mean

- Model: hf_MegatronBertForCausalLM_train
- Shape: logits [8192, 29056] f32, tokens [16, 512] i64
- Reduction dim (rnumel): 29056 (large vocab)
- Output: scalar f32 (mean cross-entropy loss)

## Measurements (NVIDIA B200, CD tuning ON)

| Metric | Value |
|--------|-------|
| Compile (best) | 229 us |
| Oracle (online softmax+xent, tiled) | 168 us |
| Gap (compile/oracle) | 1.37x |
| Kernel count | 2 |
| scalar_reduction_accumulators effect | **Significant** (same pattern as 4c98004c3aa3) |

## Diagnosis

This is the training-time variant of the same MegatronBert cross-entropy pattern as `amax_sum_sum_4c98004c3aa3`. The shapes are identical ([8192, 29056] logits, [16, 512] tokens) with only minor op-graph differences (argument order, slightly fewer ops in the mask handling).

The analysis is identical to `4c98004c3aa3`:
- **Large rnumel (29056)**: `scalar_reduction_accumulators` is beneficial and already ON by default
- **2 kernels**: inductor splits log-softmax from the gather+loss reduction
- **1.37x gap**: the oracle fuses everything into one pass, avoiding the intermediate re-read
- The difference from 4c98004c3aa3 (1.33x vs 1.37x) is noise-level variation

### Scalar accumulators impact

Same as sibling: ~1.36x improvement when enabled (from ~306 us down to ~229 us). Already deployed.

### Remaining gap

Same as sibling: requires online cross-entropy template to fuse softmax + gather + loss in one kernel pass.

## Inductor Closure

- `scalar_reduction_accumulators`: already ON and effective.
- Remaining closure path: online cross-entropy template (shared with 4c98004c3aa3).
- Priority: low-moderate (1.37x gap, 61 us absolute, key optimization already deployed).
