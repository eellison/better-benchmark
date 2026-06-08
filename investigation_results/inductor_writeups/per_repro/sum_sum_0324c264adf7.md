# sum_sum_0324c264adf7

## Compile: 372.35us, Oracle: 397.57us, Gap: 0.937x

## Classification: BAD_ORACLE

## Root Cause

The oracle (bandwidth_bound) is slower than torch.compile. The compiled code at 372.35us outperforms the oracle at 397.57us, giving a ratio of 0.937x. The oracle implements the full RMSNorm backward with a fused row epilogue plus partial column reduction matching Inductor's mixed-order schedule, but Inductor's autotuned configuration achieves marginally better throughput for this large bandwidth-bound workload.

## Kernel Count
- Oracle: N/A (slower than compile)
- Inductor: baseline is already faster

## Config Exploration
No config exploration needed -- oracle is slower than compile.

## Fix Assessment: No action needed

The oracle is a BAD_ORACLE. Inductor already wins on this repro.

## Details
- Model: RMSNorm backward (fused row epilogue + partial column reduction)
- Shape: output [512] f32 + [1152000, 512] bf16
- The oracle's schedule matches Inductor's but cannot beat its autotuned tile sizes for this bandwidth-dominated workload. The full bf16 matrix read/write and f32 partial traffic dominate at ~6.7% gap.
