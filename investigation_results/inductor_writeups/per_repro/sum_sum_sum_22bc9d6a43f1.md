# sum_sum_sum_22bc9d6a43f1

## Classification: MULTI_RESOLUTION_OUTPUT_FUSION

## Benchmark Results
- Oracle: 67.3 us
- Inductor compiled: 136.86 us
- Ratio: 2.03x
- linear_reduction_elimination fires (2 dependent reductions eliminated algebraically)

## Kernel Count
- Inductor: 2 kernels (down from 3 thanks to linear_reduction_elimination)
- Oracle: 2 kernels (spatial summary + final dependent reduction)

## Root Cause

This is fundamentally a **scheduler/fusion** issue, not purely algebraic.

After `linear_reduction_elimination` rewrites the dependent reductions algebraically,
Inductor still produces 2 kernels with incompatible iteration domains:

1. **Kernel 0** (xnumel=192, r0_numel=100352): Per-channel reductions `sum([0,2,3])`
   - 192 programs, each reducing 128*28*28=100352 elements
   - Outputs: 4 per-channel [192] vectors

2. **Kernel 1** (xnumel=150528, r0_numel=128): Batch-only reduction `sum([0], keepdim=True) -> [1,192,28,28]`
   - 150528 programs, each reducing 128 elements
   - Reads the SAME large [128,192,28,28] data as Kernel 0

The 2x gap comes from Kernel 1 re-reading ~96MB that Kernel 0 already processed.

## Why The Scheduler Cannot Fuse These

The two reductions have incompatible grid structures:
- Kernel 0: 192 programs, reducing ALL spatial positions per channel
- Kernel 1: 150528 programs, reducing batch per spatial position

Fusing requires multi-resolution output support: one kernel producing both
per-channel scalars AND per-spatial-position vectors from a single data pass.

## What Inductor Needs (Design Doc)

**Enhancement needed**: "Multi-resolution output reduction" in the scheduler.

When two reductions share the same large input tensor but have different reduction axes
(`sum([0,2,3])->>[C]` and `sum([0])->[1,C,H,W]`), the per-channel kernel should
emit atomic_add stores for the finer-grained output as a side effect.

**Files to modify**:
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` - fusion for multi-axis reductions
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` - atomic accumulator codegen
- `/tmp/pytorch-work/torch/_inductor/ir.py` - multi-resolution output IR support

**Affected repro count**: ~41 sum_sum_sum repros share this pattern.
