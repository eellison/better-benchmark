# sum_43bbf21a2e59

## Compile: 13.63us, Oracle: 12.8us, Gap: 1.065x

## Classification: PERMUTE_SIDE_SIBLING

## Root Cause

The oracle computes the complete GPT-J GELU-backward pointwise expression once, stores the transposed-view output layout [16384, 128], and accumulates the sibling column-sum [16384] in the same Triton pass. However, the gap is only 6.5% -- essentially at the performance floor for this workload.

The oracle description notes this is "BANDWIDTH_BOUND" -- the remaining cost is dominated by streaming two f32 inputs [128, 16384] each, storing the full f32 tensor output, tanh/SFU work, and the column reduction. The oracle's advantage is marginal.

Inductor generates a pointwise kernel for the GELU backward computation that materializes the output, then separately handles the permute and sum.

## Kernel Count
- Oracle: 1 kernel (GELU backward + transpose store + column sum)
- Inductor: likely 3 kernels (pointwise GELU, partial reduction, final reduction)

## Config Exploration
| Config | Result |
|--------|--------|
| combo_kernels + CDT | 13.63 us (1.065x) |

The 6.5% gap is near the measurement floor for this shape.

## Fix Assessment: Design doc (same as sum_13195092a57b)

Same PERMUTE_SIDE_SIBLING pattern as sum_13195092a57b and sum_66b92a2b30bb. The scheduler cannot fuse the transposed output materialization with the sibling column reduction.

However, the gap here is minimal (6.5%) because the GELU backward is compute-heavy (tanh, pow, mul chains), so the extra memory traffic from separate kernels is proportionally smaller relative to the total runtime.

### Relevant files:
- `/tmp/pytorch-work/torch/_inductor/scheduler.py`: fusion decisions
- `/tmp/pytorch-work/torch/_inductor/ir.py`: realize_hint

### Affected repro count:
Part of the broader PERMUTE_SIDE_SIBLING family (sum_13195092a57b, sum_66b92a2b30bb, sum_7953a8b0bbba).

## Details
- Model: hf_GPTJForCausalLM (train)
- Shape: [128, 16384] f32 (two inputs)
- Pattern: GELU backward (tanh-based) -> view -> permute [16384, 128] (output 0) + sum(dim=0) [16384] (output 1)
- Near bandwidth floor due to compute-heavy GELU backward
