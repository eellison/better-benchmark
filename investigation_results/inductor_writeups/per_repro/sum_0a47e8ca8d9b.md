# sum_0a47e8ca8d9b

## Compile: 191.30us, Oracle: 262.98us, Gap: 0.727x

## Classification: BAD_ORACLE

## Root Cause

The oracle (bert_softmax_backward_512) is slower than torch.compile. The compiled code at 191.30us outperforms the oracle at 262.98us, giving a ratio of 0.727x. This means Inductor is already generating superior code for this workload.

## Kernel Count
- Oracle: N/A (slower than compile)
- Inductor: baseline is already faster

## Config Exploration
No config exploration needed -- oracle is slower than compile.

## Fix Assessment: No action needed

The oracle is a BAD_ORACLE. Inductor already wins on this repro.

## Details
- Model: BERT softmax backward (512 sequence length)
- Shape: output [384, 512, 512] f32
