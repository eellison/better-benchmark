"""
Reference Triton softmax kernel from tritonbench.
Source: tritonbench/operators/softmax/operator.py

This is the hand-written Triton implementation that tritonbench uses
as the "triton_softmax" baseline for benchmarking against torch and inductor.

Maps to our canonical pattern: amax_sum_5a19f8539c85 (softmax)
"""
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-row softmax kernel. One program per row."""
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load row into SRAM with mask
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
    # Numerical stability: subtract max
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


@triton.jit
def softmax_kernel_multirow(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    n_rows,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    """Multi-row softmax kernel for small N with many rows.
    Uses 2D [ROWS_PER_PROGRAM, BLOCK_SIZE] layout.
    Mirrors torch.compile's persistent_reduction approach.
    """
    prog_idx = tl.program_id(0)
    row_offsets = prog_idx * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # [ROWS_PER_PROGRAM, BLOCK_SIZE] index grid
    indices = row_offsets[:, None] * input_row_stride + col_offsets[None, :]
    mask = (row_offsets[:, None] < n_rows) & (col_offsets[None, :] < n_cols)

    # Load all rows at once
    data = tl.load(input_ptr + indices, mask=mask, other=-float("inf"))

    # Row-wise max, exp, sum, normalize -- reductions along axis=1
    row_max = tl.max(data, axis=1)[:, None]
    numerator = tl.exp(data - row_max)
    denominator = tl.sum(numerator, axis=1)[:, None]
    softmax_output = numerator / denominator

    # Store
    out_indices = row_offsets[:, None] * output_row_stride + col_offsets[None, :]
    tl.store(output_ptr + out_indices, softmax_output, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Run the hand-written Triton softmax kernel."""
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)

    # Heuristic for num_warps
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    elif BLOCK_SIZE >= 1024:
        num_warps = 4
    else:
        num_warps = max(1, BLOCK_SIZE // 256)

    # For small N with many rows, use multi-row kernel
    if BLOCK_SIZE <= 512 and n_rows >= 8192:
        ROWS_PER_PROGRAM = min(4096 // BLOCK_SIZE, 8)
        grid = ((n_rows + ROWS_PER_PROGRAM - 1) // ROWS_PER_PROGRAM,)
        multirow_warps = max(2, min(ROWS_PER_PROGRAM * BLOCK_SIZE // 256, 8))
        softmax_kernel_multirow[grid](
            y, x,
            x.stride(0), y.stride(0),
            n_cols, n_rows,
            num_warps=multirow_warps,
            BLOCK_SIZE=BLOCK_SIZE,
            ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
        )
    else:
        softmax_kernel[(n_rows,)](
            y, x,
            x.stride(0), y.stride(0),
            n_cols,
            num_warps=num_warps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return y


if __name__ == "__main__":
    # Quick test
    x = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
    y = triton_softmax(x)
    y_ref = torch.softmax(x.float(), dim=-1).half()
    print(f"Max diff: {(y - y_ref).abs().max().item():.6f}")
    print("OK")
