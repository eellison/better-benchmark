# var_mean_e7e57aea37e1

## Compile: 9.25us, Oracle: 8.54us, Gap: 1.082x

## Diagnosis: PERSISTENT_REDUCTION_STRATEGY

## Root cause: The oracle computes the CycleGAN instance-normalization ReLU scope for N=1 NCHW tensors by reducing each channel's population variance/mean over the spatial plane (128x128=16384 elements) and writing the final contiguous normalized ReLU output directly from the same Triton program. Inductor generates a single kernel (triton_red_fused_add_mul_relu_rsqrt_sub_var_mean) but uses a looped reduction for the spatial dimension.

The 8% gap comes from the oracle using a persistent approach that avoids re-reading the spatial plane for the normalization epilogue. With 128 channels and spatial=16384, the oracle keeps partial results in registers/shared memory rather than looping twice.

## Kernel count
- Inductor: 1 kernel (red_fused_add_mul_relu_rsqrt_sub_var_mean)
- Oracle: 1 kernel (persistent instance-norm + ReLU)

## Config exploration results
- multi_kernel=1 (default): 9.25us (ratio 1.082x)
- multi_kernel=2: 9.18us (ratio 1.104x) - no improvement
- multi_kernel=3: 9.18us (ratio 1.095x) - no improvement
- coordinate_descent_tuning=True, combo_kernels=True: already enabled

## Classification: PERSISTENT_REDUCTION_STRATEGY

Same class as var_mean_fa8932a4b21f: the reduction dimension (spatial=16384) could benefit from a persistent approach that avoids re-reading input for the normalization epilogue. However, 16384 is larger and may not fully fit in registers; the oracle likely uses a chunked persistent approach.

## Fix path
Enhancement in `/tmp/pytorch-work/torch/_inductor/choices.py`: For instance-norm patterns (reduction over spatial dimensions with small channel count, N=1), consider persistent or semi-persistent reduction strategies that keep data in shared memory/registers between the statistics pass and the normalization pass.

## Status: design_doc
