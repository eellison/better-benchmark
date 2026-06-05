# var_mean_fa8932a4b21f

## Compile: 20.26us, Oracle: 17.86us, Gap: 1.134x

## Diagnosis: PERSISTENT_REDUCTION_STRATEGY

## Root cause: The oracle computes the complete Whisper residual LayerNorm scope in one Triton row kernel, including the strided fp16 residual add, fp16-range clamp and clone round-trip, fp32 population var_mean over hidden=384, affine scale/bias, fp16 cast, and three aliasing [12000,384] views. Inductor generates a single kernel as well (triton_red_fused_add_clamp_max_clamp_min_clone_convert_element_type_mul_rsqrt_sub_var_mean_view), but uses a non-persistent (looped Welford) reduction strategy for hidden=384.

The oracle uses a persistent approach with BLOCK_H=512 (next power of 2 above 384), fitting the entire row in one tile and avoiding re-reading the strided input. Inductor's looped Welford reads the input twice (once for mean/variance pass, once for normalization pass).

## Kernel count
- Inductor: 1 kernel (red_fused_add_clamp_max_clamp_min_clone_convert_element_type_mul_rsqrt_sub_var_mean_view)
- Oracle: 1 kernel (persistent single-pass row kernel)

## Config exploration results
- multi_kernel=1 (default): 20.26us (ratio 1.134x)
- multi_kernel=2: 20.26us (ratio 1.130x) - no improvement
- multi_kernel=3: 20.22us (ratio 1.137x) - no improvement
- coordinate_descent_tuning=True, combo_kernels=True: already enabled

## Classification: PERSISTENT_REDUCTION_STRATEGY

Same root cause as var_mean_034578d167c5: for hidden=384 (next_pow2=512), the entire row fits in registers. The norm template should detect this and emit a persistent (single-pass) reduction instead of a two-pass Welford loop. The strided fp16 input with non-standard stride causes the two-pass approach to re-read from DRAM.

## Fix path
Enhancement in `/tmp/pytorch-work/torch/_inductor/choices.py`: Lower the persistent reduction threshold for row-normalization patterns where hidden dimension fits in a single RBLOCK tile. For hidden=384, BLOCK_H=512 easily fits in registers, enabling single-pass computation.

Related: var_mean_034578d167c5 (identical pattern, Whisper with hidden=384)

## Status: design_doc
