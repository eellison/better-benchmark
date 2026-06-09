# amax_sum_66dc4fac757d

## Compile: 13.66us, Oracle: 11.9us, Gap: 1.148x

## Diagnosis: NEW_PATTERN (multi-row persistent softmax template)

## Root cause: Inductor already emits a single fused online softmax kernel (`triton_per_fused_add_convert_element_type_exp_log_prepare_softmax_online_sub_view_0`) processing one row at a time. The oracle processes 8 rows per Triton program using a multi-row persistent template with a [BLOCK_ROWS, BLOCK_COLS] tile, reducing kernel launch overhead and improving warp utilization for the 49152 rows x 128 columns workload. The 14.8% gap comes from launch/scheduling overhead amortized across multiple rows and better register/L2 utilization from the wider tile.

## Fix path: Extend Inductor's persistent reduction template for small-row softmax-like patterns to process multiple rows per program. When the row width is small (128 here) and fits entirely in registers, tile multiple rows (e.g., 4-8) per block to reduce grid size from 49152 to ~6144 blocks, amortizing launch cost and improving SM utilization.

## Status: design_todo

## Details

- Model: torchbench_hf_Reformer_infer_005
- Pattern: amax+sum reduction (logsumexp softmax with fp16 rounding)
- Shape: [768, 64, 128] viewed as [1, 12, 64, 64, 128], reduction over last dim (128 elements)
- Total rows: 1*12*64*64 = 49152
- Inductor kernels: 1 (fused online softmax -- good!)
- Oracle kernels: 1 (multi-row persistent softmax, 8 rows/block)
- The gap is relatively small (1.15x) since Inductor already fuses correctly; the improvement is purely from multi-row tiling.
- Config exploration: coordinate_descent_tuning (13.28us), combo+multi3 (13.28us) -- no config closes the gap.
- The fp16 intermediate rounding (logsumexp cast to fp16, then subtract/exp in fp16) is faithfully reproduced by both.
- File references: /tmp/pytorch-work/torch/_inductor/codegen/triton.py (persistent reduction template), /tmp/pytorch-work/torch/_inductor/choices.py (reduction strategy selection)
