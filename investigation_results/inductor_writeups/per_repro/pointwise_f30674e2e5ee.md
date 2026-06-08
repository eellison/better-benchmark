# pointwise_f30674e2e5ee

## Classification: BANDWIDTH_BOUND

## Current Result

- Family: `pad_copy`
- Oracle path: `repros/canonical/pointwise_f30674e2e5ee/oracle_right_bottom_pad_copy.py`
- Correctness: PASS (assumed from oracle docstring)
- Status: `at_floor`

## Diagnosis

The oracle computes the complete right-and-bottom zero `constant_pad_nd` scope by directly materializing PyTorch's contiguous-or-channels-last padded output in one storage-order Triton kernel, whereas Inductor already lowers this isolated pad to an equivalent one-kernel output materialization with essentially the same CUDAGraph timing. Inductor cannot materially improve this local repro today because the exact contract still requires a fresh padded allocation, reading every interior input element, writing every output element, and zeroing the narrow pad fringe with no producer fusion or algebraic work left to remove.

## Root cause

This is a pure bandwidth-bound pad-copy operation. The oracle and Inductor both emit a single kernel that reads the input and writes the padded output. No fusion, algebraic elimination, or tiling improvement can save memory traffic here -- the workload is dominated by mandatory read+write of the full tensor.

## Kernel count

- Oracle: 1 kernel (_pad_nchw_kernel or _pad_nhwc_kernel depending on layout)
- Inductor: 1 kernel (equivalent pointwise pad materialization)

## Config exploration

No config can improve a bandwidth-bound single-kernel pad copy. The gap (if any) is within measurement noise.

## Recommendation

Record as at-floor pad-copy case. No action unless broader pointwise memory-codegen, allocation, or launch-overhead improvements move both implementations together.

## Relevant files

- `/tmp/pytorch-work/torch/_inductor/lowering.py` (constant_pad_nd lowering)
- `/tmp/pytorch-work/torch/_inductor/codegen/triton.py` (pointwise codegen)
