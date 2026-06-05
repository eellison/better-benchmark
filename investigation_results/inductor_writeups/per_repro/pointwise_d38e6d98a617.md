# pointwise_d38e6d98a617 - ReLU + MaxPool + Mask Fusion Gap

## Benchmark Result
- Oracle: 76.96 us
- Compile: 90.88 us
- Ratio: 1.181x
- Status: GOOD (oracle wins)

## Classification
SCHEDULER_FUSION

## Root Cause

The oracle computes ReLU, 2x2 stride-2 low-memory max-pool-with-offsets, and the input-shaped `relu <= 0` boolean mask in **one** Triton kernel. The kernel iterates over output-space (14x14 per channel-plane), loads the 4 input elements per pool window, applies ReLU inline, finds the max + offset, and writes the le_mask back to the full 28x28 input space.

Inductor emits **two** kernels:
1. `triton_poi_fused_0` (51.4M elements): computes `relu(x) <= 0` mask over the full input shape [128,512,28,28], reading all 51M f32 input elements and writing 51M bool.
2. `triton_poi_fused_1` (12.8M elements): reads the raw input again, applies ReLU inline to each of the 4 pool window elements, computes max+offsets for the pooled output [128,512,14,14].

The gap is that kernel 0 reads the entire 128*512*28*28 input to produce the mask, and kernel 1 re-reads the same input to do the pool. The oracle fuses both into one pass that reads input once and writes pool output + offsets + mask.

## Kernel Count
- Inductor: 2 kernels
- Oracle: 1 kernel

## Why Inductor Cannot Do This Today

The scheduler does not fuse:
- A stencil/reduction consumer (maxpool with 2x2 window, output shape [128,512,14,14]) 
- With a layout-preserving pointwise side-output consumer (le_mask, shape [128,512,28,28])

of the same shared producer (ReLU). The two consumers have different iteration domains (output-shaped pool vs input-shaped mask), so standard pointwise fusion rules don't apply.

## Config Exploration
- `combo_kernels=True`: no change (still 2 kernels)
- `combo_kernel_per_subkernel_blocks=True`: no change
- `triton.multi_kernel=3`: no change (still 2 kernels)
- `coordinate_descent_tuning=True`: already enabled, no effect on kernel count

No existing config can resolve this gap.

## Fix Direction (Design Doc)

**Enhancement needed**: Multi-output producer fusion for stencil + pointwise sibling consumers.

The scheduler needs to recognize that when a pointwise producer (ReLU) has:
1. A stencil consumer (maxpool reading a window of the producer's output)
2. A pointwise consumer reading the same producer at the same element

It can fuse both into the stencil consumer's iteration domain. The stencil kernel iterates over output elements but accesses all input elements through its window loads -- so the pointwise mask can be written as a side-effect during those loads (each input element is visited exactly once across the pool windows when stride == kernel_size).

**File references**:
- `/tmp/pytorch-work/torch/_inductor/scheduler.py` (can_fuse, lines ~336+)
- `/tmp/pytorch-work/torch/_inductor/lowering.py` (line 5625: `_low_memory_max_pool_with_offsets`)

**Affected repros**: Any VGG-style pattern with ReLU -> maxpool where the ReLU backward mask is also needed (common in training).

## Source
- Label: torchbench_vgg16_train_000
- Pattern: relu -> maxpool + le_mask sibling outputs from shared relu intermediate
