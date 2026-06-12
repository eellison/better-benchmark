"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete bf16 bottom-row `constant_pad_nd` scope by materializing the fresh contiguous padded output directly, copying the input prefix and zero-filling only the appended rows, whereas Inductor lowers the isolated pad through a generic pad/copy schedule for each embedding-table shape; Inductor cannot materially improve this local repro beyond a specialized copy template because the exact contract still requires reading the full input, writing the full output, preserving the non-contiguous MobileBERT input stride, allocating a fresh contiguous result, and zeroing the narrow tail; the fix is BANDWIDTH_BOUND: record this as an at-floor pad-copy case unless measured B200 timing shows a point-specific memory-codegen or strided-copy gap."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _bottom_pad_contiguous_kernel(
    input_ptr,
    output_ptr,
    IN_NUMEL: tl.constexpr,
    OUT_NUMEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    out_mask = offsets < OUT_NUMEL
    copy_mask = offsets < IN_NUMEL
    values = tl.load(input_ptr + offsets, mask=copy_mask, other=0.0)
    tl.store(output_ptr + offsets, values, mask=out_mask)


@triton.jit
def _bottom_pad_strided_2d_kernel(
    input_ptr,
    output_ptr,
    IN_ROWS: tl.constexpr,
    OUT_ROWS: tl.constexpr,
    COLS: tl.constexpr,
    IN_S0: tl.constexpr,
    IN_S1: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < OUT_ROWS) & (cols[None, :] < COLS)
    in_bounds = mask & (rows[:, None] < IN_ROWS)
    in_offsets = rows[:, None] * IN_S0 + cols[None, :] * IN_S1
    out_offsets = rows[:, None] * COLS + cols[None, :]
    values = tl.load(input_ptr + in_offsets, mask=in_bounds, other=0.0)
    tl.store(output_ptr + out_offsets, values, mask=mask)


def _launch_bottom_pad(
    inputs,
    *,
    BLOCK: int,
    BLOCK_M: int,
    BLOCK_N: int,
    num_warps: int,
):
    x, pad = inputs
    rows = int(x.shape[0])
    cols = int(x.shape[1])
    out_rows = rows + int(pad[-1])
    out = torch.empty_strided(
        (out_rows, cols),
        (cols, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    out_numel = out_rows * cols

    if x.is_contiguous():
        _bottom_pad_contiguous_kernel[(triton.cdiv(out_numel, BLOCK),)](
            x,
            out,
            IN_NUMEL=x.numel(),
            OUT_NUMEL=out_numel,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=4,
        )
    else:
        _bottom_pad_strided_2d_kernel[
            (triton.cdiv(out_rows, BLOCK_M), triton.cdiv(cols, BLOCK_N))
        ](
            x,
            out,
            IN_ROWS=rows,
            OUT_ROWS=out_rows,
            COLS=cols,
            IN_S0=x.stride(0),
            IN_S1=x.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=4,
        )
    return out


# (T([20005,768], bf16), S([0,0,0,3]))
@oracle_impl(hardware="B200", point="ba1b8f0f", BLOCK=2048, BLOCK_M=32, BLOCK_N=32, num_warps=4)
# (T([50265,1024], bf16), S([0,0,0,7]))
@oracle_impl(hardware="B200", point="6883fad3", BLOCK=1024, BLOCK_M=32, BLOCK_N=32, num_warps=4)
# (T([128100,1536], bf16), S([0,0,0,4]))
@oracle_impl(hardware="B200", point="1779a8cb", BLOCK=1024, BLOCK_M=32, BLOCK_N=32, num_warps=4)
# (T([30522,768], bf16), S([0,0,0,6]))
@oracle_impl(hardware="B200", point="cd997de8", BLOCK=1024, BLOCK_M=32, BLOCK_N=32, num_warps=4)
# (T([50257,768], bf16), S([0,0,0,7]))
@oracle_impl(hardware="B200", point="cb779bb6", BLOCK=1024, BLOCK_M=32, BLOCK_N=32, num_warps=4)
# (T([2,4096], bf16), S([0,0,0,6]))
@oracle_impl(hardware="B200", point="10c5c015", BLOCK=1024, BLOCK_M=32, BLOCK_N=32, num_warps=4)
# (T([30522,128], bf16), S([0,0,0,6]))
@oracle_impl(hardware="B200", point="d67c38a5", BLOCK=1024, BLOCK_M=32, BLOCK_N=32, num_warps=4)
# (T([50257,2048], bf16), S([0,0,0,7]))
@oracle_impl(hardware="B200", point="360d77c3", BLOCK=1024, BLOCK_M=32, BLOCK_N=32, num_warps=4)
# (T([50265,768], bf16), S([0,0,0,7]))
@oracle_impl(hardware="B200", point="d59edba9", BLOCK=1024, BLOCK_M=32, BLOCK_N=32, num_warps=4)
# (T([50005,768], bf16), S([0,0,0,3]))
@oracle_impl(hardware="B200", point="ea31889c", BLOCK=1024, BLOCK_M=32, BLOCK_N=32, num_warps=4)
# (T([30522,512], bf16, stride=(1,30528)), S([0,0,0,6]))
@oracle_impl(hardware="B200", point="4c38b93b", BLOCK=1024, BLOCK_M=32, BLOCK_N=32, num_warps=4)
def oracle_forward(
    inputs,
    *,
    BLOCK: int,
    BLOCK_M: int,
    BLOCK_N: int,
    num_warps: int,
):
    return _launch_bottom_pad(
        inputs,
        BLOCK=BLOCK,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
