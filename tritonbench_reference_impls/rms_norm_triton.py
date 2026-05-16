"""
Reference Triton RMS norm kernel from tritonbench.
Source: tritonbench/operators/rms_norm/fused_triton.py

Hand-written Triton implementation for RMS norm backward.
Forward uses a reference PyTorch implementation.
Maps to our canonical pattern: mean_bf5fef91b4d7 (rms_norm)
"""
import math

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"M_INCREMENT": M_INCREMENT},
            num_warps=w,
        )
        for M_INCREMENT in [1, 2, 4, 8, 16]
        for w in [2, 4, 8]
    ],
    key=["M", "N"],
)
@triton.jit
def _rms_norm_bwd_fused(
    DX,
    DY,
    DW,
    X,
    W,
    RMS,
    stride,
    N,
    M,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    M_INCREMENT: tl.constexpr,
    N_POW_2: tl.constexpr,
):
    """Fused RMS norm backward kernel.

    Computes dx and partial dw in a single pass over the data.
    Each program handles BLOCK_SIZE_M rows and all N columns.
    """
    pid = tl.program_id(0)
    start_row = pid * BLOCK_SIZE_M
    grad_w = tl.full([BLOCK_SIZE_N], 0, tl.float32)
    cols = tl.arange(0, BLOCK_SIZE_N)
    if N_POW_2:
        col_mask = None
    else:
        col_mask = cols < N

    w = tl.load(W + cols, mask=col_mask).to(tl.float32)[None, :]

    for cur_row in tl.range(0, BLOCK_SIZE_M, M_INCREMENT):
        rows = start_row + cur_row + tl.arange(0, M_INCREMENT)
        row_indices = rows * stride
        row_mask = rows < M

        rms = tl.load(RMS + rows, mask=row_mask).to(tl.float32)[:, None]

        if N_POW_2:
            index_mask = row_mask[:, None]
        else:
            index_mask = row_mask[:, None] & col_mask[None, :]

        indices = row_indices[:, None] + cols[None, :]

        x = tl.load(X + indices, mask=index_mask, other=0).to(tl.float32)
        dy = tl.load(DY + indices, mask=index_mask, other=0).to(tl.float32)

        # Compute dx
        m = dy * w
        row_dot = tl.sum(m * x, axis=1)[:, None]
        scale = -(1.0 / N) * rms * rms * rms
        dx = rms * m
        dx += scale * row_dot * x

        tl.store(DX + indices, dx, mask=index_mask)
        grad_w += tl.sum((dy * x) * rms, axis=0)

    tl.store(DW + pid * N + cols, grad_w, mask=col_mask)


def rms_norm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """RMS norm forward (reference implementation)."""
    rms = 1.0 / torch.sqrt(torch.mean(x.float().square(), dim=-1, keepdim=True) + eps)
    return (x * rms * weight).to(x.dtype), rms


if __name__ == "__main__":
    M, H = 2048, 4096
    x = torch.randn(M, H, dtype=torch.bfloat16, device='cuda')
    w = torch.ones(H, dtype=torch.bfloat16, device='cuda')
    y, rms = rms_norm_forward(x, w)
    # Compare with manual
    variance = x.float().pow(2).mean(-1, keepdim=True)
    y_ref = (w * x * torch.rsqrt(variance + 1e-6)).to(x.dtype)
    print(f"Max diff: {(y - y_ref).abs().max().item():.6f}")
    print("OK")
