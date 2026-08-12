"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete bf16 row-prefix slice plus bf16-to-fp32 cast scope as one row-tiled Triton materialization, reading columns `0:197951` from each contiguous 197952-wide input row and writing the fresh contiguous fp32 `[512,197951]` output, whereas Inductor's isolated slice/cast lowering has the same mandatory input read, widening conversion, output write, allocation, and launch envelope for this repro; Inductor cannot remove the skipped last-column indexing locally because the returned tensor is the trimmed fp32 materialization with no producer or consumer fusion left; the fix is BANDWIDTH_BOUND: record this as a row-slice cast floor unless broader pointwise layout-copy codegen or launch/allocation work moves both implementations together."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 512
IN_COLS = 197952
OUT_COLS = 197951


@triton.jit
def _row_prefix_bf16_to_f32_kernel(
    input_ptr,
    output_ptr,
    IN_STRIDE: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < OUT_STRIDE
    values = tl.load(input_ptr + row * IN_STRIDE + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + row * OUT_STRIDE + cols, values, mask=mask)


# (T([512,197952], bf16))
@oracle_impl(hardware="B200", point="b4e770d9", BLOCK_N=4096, num_warps=8)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int):
    (x,) = inputs
    out = torch.empty_strided(
        (ROWS, OUT_COLS),
        (OUT_COLS, 1),
        device=x.device,
        dtype=torch.float32,
    )
    _row_prefix_bf16_to_f32_kernel[(ROWS, triton.cdiv(OUT_COLS, BLOCK_N))](
        x,
        out,
        IN_STRIDE=IN_COLS,
        OUT_STRIDE=OUT_COLS,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=4,
    )
    return out
