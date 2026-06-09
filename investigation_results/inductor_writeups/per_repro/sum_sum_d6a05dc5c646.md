# sum_sum_d6a05dc5c646

## Compile: 214.08us, Oracle: 124.0us, Gap: 1.73x

## Diagnosis: SCHEDULER_FUSION

## Root cause: Inductor schedules the MobileViT SiLU-backward plus batch-norm-backward fragment as 5 separate Triton kernels: the SiLU derivative computation, two sibling channel reduction kernels (sum([0,2,3])), and the dependent full-tensor epilogue that writes the returned [128,16,128,128] gradient and [16] side vector. The oracle shares the fused SiLU derivative producer across both channel reductions, then sinks the finalized per-channel summaries directly into the returned tensor in one coordinated plan.

## Fix path: Add scheduler/codegen support for shared channel reductions with finalized-scalar epilogues that write both tensor and vector outputs. The key insight is that the SiLU derivative (sigmoid * (1 + x*(1-sigmoid))) is cheap to recompute and can be shared between the two sum([0,2,3]) reductions without materializing the [128,16,128,128] intermediate.

## Status: design_todo

## Details

- Model: timm_mobilevit_s (train)
- Pattern: sum+sum reduction (SiLU backward + batch-norm backward)
- Inductor kernels: 5 unique Triton kernels
- Ops: sub, mul, unsqueeze, neg, exp, add, reciprocal, sum([0,2,3]), squeeze, pow, div
- Shapes: [128,16,128,128] f32 inputs (128 batches, 16 channels, 128x128 spatial), [16] channel reductions
- Data volume: 128*16*128*128*4 = 128MB per tensor read -- bandwidth-dominated
- The gap (1.73x) comes from: (1) materializing the SiLU-derivative intermediate ([128,16,128,128], 128MB), (2) reading the same large tensors 3+ times across separate kernels, (3) the two channel reductions reading the same producer independently
- combo_kernels=True makes it worse (241us), aggressive_fusion=True also worse (243us)
- No existing Inductor config flag addresses this gap
