# pointwise_c3360ba4828a - Virtual Cat + Dropout + Sibling Masks Fusion Gap

## Benchmark Result
- Oracle: 149.44 us
- Compile: 203.62 us
- Ratio: 1.363x
- Status: GOOD (oracle wins)

## Classification
SCHEDULER_FUSION

## Root Cause

The oracle computes the full SqueezeNet fire module epilogue in **one** kernel:
1. Two ReLU activations on separate conv outputs (f32[512,256,13,13] each)
2. Virtual channel-concatenation (no materialized cat buffer)
3. Seeded Inductor dropout (rand > 0.5, scale by 2.0) over the concatenated [512,512,13,13]
4. Two boolean backward masks (le_scalar for each ReLU output)

All done in a single pass over the input elements without materializing:
- The concatenated f32[512,512,13,13] intermediate (saves 512*512*13*13*4 = 173MB write+read)
- The individual ReLU outputs (computed inline)

Inductor emits **two** kernels:
1. `triton_poi_fused_0`: Computes both ReLUs, writes the le_masks, and writes ReLU outputs to intermediate buffers.
2. `triton_poi_fused_gt_inductor_lookup_seed_inductor_random_mul_1`: Reads the concatenated ReLU intermediates, applies stochastic dropout, writes the dropout output.

The fundamental issue: Inductor materializes the ReLU outputs as intermediate buffers because the cat consumer and the le_mask consumers have different shapes. The cat creates a [512,512,13,13] tensor from two [512,256,13,13] inputs, and the dropout operates on the larger shape. Inductor's scheduler cannot keep the cat as a "virtual" multi-source producer while threading the dropout RNG and sibling mask stores through the same kernel.

## Kernel Count
- Inductor: 2 kernels  
- Oracle: 1 kernel

## Memory Traffic Analysis

**Oracle** (1 kernel):
- Reads: 2 * 512*256*13*13*4 = 173.9 MB (two conv inputs)
- Writes: 512*512*13*13*4 = 173.9 MB (dropout) + 2*512*256*13*13*1 = 43.5 MB (bool masks)
- Total: ~391 MB

**Inductor** (2 kernels):
- Kernel 1 reads: 2 * 86.9 MB = 173.9 MB, writes: 2 * 86.9 MB (relu outputs) + 43.5 MB (masks) = ~261 MB
- Kernel 2 reads: 173.9 MB (relu intermediates via cat), writes: 173.9 MB (dropout) = ~348 MB
- Total reads+writes: ~783 MB (2x more than oracle)

The extra 173.9 MB write + 173.9 MB read of the ReLU intermediates explains the 1.36x slowdown.

## Config Exploration
- `combo_kernels=True`: no change (2 kernels)
- `combo_kernel_per_subkernel_blocks=True`: no change
- `triton.multi_kernel=3`: no change
- `coordinate_descent_tuning=True`: already enabled

No config resolves this.

## Fix Direction (Design Doc)

**Enhancement needed**: Virtual-cat producer fusion with seeded dropout and sibling stores.

The scheduler needs to:
1. Recognize that `cat([relu(x0), relu(x1)], dim=1)` can be kept virtual -- each element of the concatenated output maps back to exactly one of the two inputs
2. When the cat feeds into dropout (which uses positional RNG), the dropout can be applied in input-space by computing the correct output offset for each input element
3. The sibling le_mask outputs (which read the ReLU intermediates) can be written in the same pass

This requires the scheduler to model "virtual concat" as a zero-cost layout transformation and fuse through it.

**File references**:
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` (fusion decisions)
- `/tmp/pytorch-work/torch/_inductor/ir.py` (ConcatKernel, or virtual cat representation)
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (RNG indexing for positional dropout)

**Affected repros**: SqueezeNet and other architectures with concat + dropout in training (fire modules, DenseNet dense blocks).

## Source
- Label: torchbench_squeezenet1_1_train_000
- Pattern: two ReLUs -> virtual cat -> positional dropout + two backward masks
