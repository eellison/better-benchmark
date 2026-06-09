# sum_sum_79eb2b561a83

## Compile: 311.1us, Oracle: 292.7us, Gap: 1.063x

## Diagnosis: COOPERATIVE_SPLIT_K

## Root cause: Inductor schedules the GhostNet batch-norm-backward as 3 kernels: (1) a reduction kernel that fuses the subtraction and computes both sibling channel sums over [N,H,W]=[512,56,56], (2) a persistent reduction for channel statistic finalization, and (3) a pointwise epilogue for the full [512,48,56,56] output. The oracle uses 2 kernels by employing atomic-add cooperative split-K for the channel reductions (splitting the N*HW dimension across thread blocks, accumulating via atomic adds) and then a single epilogue kernel that writes both the tensor output and the [48] vector output using the finalized sums. The Inductor version materializes the reduction intermediates between passes.

## Fix path: Add a cooperative split-K channel reduction template that uses atomic accumulation across the N*HW dimension for small-channel BN-backward patterns. The scheduler needs to recognize that with C=48 channels and N*HW=512*3136=1,605,632 elements per channel, a split-K approach with atomic accumulation is profitable. Changes needed in `torch/_inductor/choices.py` (cooperative reduction heuristic for channel reductions) and `torch/_inductor/scheduler.py` (fuse epilogue with finalized channel sums).

## Status: design_todo

## Details

- Model: timm_ghostnet_100 (training backward, 1 shape)
- Pattern: BN backward with dual channel reductions and dependent tensor epilogue
- Shape: [512, 48, 56, 56] f32 -> [512, 48, 56, 56] gradient output + [48] weight gradient
- Inductor: 3 kernels (reduction, persistent reduction, pointwise epilogue)
- Oracle: 2 kernels (atomic split-K channel sums, combined epilogue writing both outputs)
- The gap is small (1.063x) because Inductor's existing fusion already handles most of the work well; the remaining gap is from one extra kernel launch and the intermediate buffer materialization between the reduction and epilogue
- File references: `/tmp/pytorch-work/torch/_inductor/choices.py`, `/tmp/pytorch-work/torch/_inductor/scheduler.py`
