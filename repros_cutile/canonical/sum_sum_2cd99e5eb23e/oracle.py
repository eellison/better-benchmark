"""cuTile port of sum_sum_2cd99e5eb23e: static RMSNorm backward.

Matches Triton's 2-kernel structure:
  1. `_rmsnorm_bwd_row_kernel` — per row-block compute bf16 output and
     accumulate col-summed x * rstd * scale into partial buffer.
  2. `_finalize_cols_kernel` — reduce column partials over row-blocks.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


M = 1_152_000
N = 512
INV_N = 1.0 / 512.0


def _ceil_pow2(value: int) -> int:
    return 1 << (value - 1).bit_length()


@ct.kernel
def _rmsnorm_bwd_row_kernel(
    scale_ptr,     # bf16 () -> viewed as (1,)
    weight_ptr,    # f32 (N,)
    rstd_ptr,      # f32 (M, 1) -> viewed as (M,)
    x_ptr,         # bf16 (M, N)
    out_ptr,       # bf16 (M, N)
    partial_ptr,   # f32 (num_blocks, N)
    ROW_BLOCK: ct.Constant[int],
    N_: ct.Constant[int],
):
    pid = ct.bid(0)

    scale_tile = ct.load(scale_ptr, index=(0,), shape=(1,))
    scale_f = ct.astype(scale_tile, ct.float32)  # (1,)

    weight = ct.load(weight_ptr, index=(0,), shape=(N_,))
    scaled_weight_1d = weight * scale_f  # broadcasting (N,) * (1,) => (N,)
    scaled_weight = ct.reshape(scaled_weight_1d, (1, N_))

    x = ct.load(x_ptr, index=(pid, 0), shape=(ROW_BLOCK, N_))
    x_f = ct.astype(x, ct.float32)
    rstd = ct.load(rstd_ptr, index=(pid,), shape=(ROW_BLOCK,))
    rstd_col = ct.reshape(rstd, (ROW_BLOCK, 1))

    weighted_x = scaled_weight * x_f
    row_sum = ct.sum(weighted_x, axis=1, keepdims=True)
    rstd3 = rstd_col * rstd_col * rstd_col
    row_scale = row_sum * (-0.5) * rstd3 * INV_N
    first_term = scaled_weight * rstd_col
    second_term = row_scale * (x_f * 2.0)
    out = first_term + second_term
    ct.store(out_ptr, index=(pid, 0), tile=ct.astype(out, ct.bfloat16))

    # Column reductions: col_terms = x * rstd * scale, sum axis=0
    scale_2d = ct.reshape(scale_f, (1, 1))
    col_terms = x_f * rstd_col * scale_2d
    col_sum = ct.sum(col_terms, axis=0, keepdims=True)
    ct.store(partial_ptr, index=(pid, 0), tile=col_sum)


@ct.kernel
def _finalize_cols_kernel(
    partial_ptr,   # f32 (NUM_BLOCKS, N)
    out_ptr,       # f32 (N,)
    NUM_BLOCKS: ct.Constant[int],
    BLOCKS_P2: ct.Constant[int],
    N_: ct.Constant[int],
):
    col_block = ct.bid(0)
    # Load a (BLOCKS_P2, 1) tile — one column per program.
    tile = ct.load(partial_ptr, index=(0, col_block), shape=(BLOCKS_P2, 1),
                   padding_mode=ct.PaddingMode.ZERO)
    b_idx = ct.arange(BLOCKS_P2, dtype=ct.int32)
    b_valid = ct.reshape(b_idx < NUM_BLOCKS, (BLOCKS_P2, 1))
    zero_f = ct.zeros((BLOCKS_P2, 1), dtype=ct.float32)
    tile = ct.where(b_valid, tile, zero_f)
    total = ct.sum(tile, axis=0)  # (1,)
    ct.store(out_ptr, index=(col_block,), tile=total)


@oracle_impl(hardware="B200", point="cbc6f48e", ROW_BLOCK=64)
def oracle_forward(inputs, *, ROW_BLOCK: int):
    scale, weight, rstd, x, *_shape = inputs

    num_row_blocks = M // ROW_BLOCK  # 1152000 / 64 = 18000
    out = torch.empty_strided((M, N), (N, 1), device=x.device, dtype=torch.bfloat16)
    partials = torch.empty((num_row_blocks, N), device=x.device, dtype=torch.float32)
    grad_weight = torch.empty((N,), device=x.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    scale_view = scale.view(1)
    rstd_view = rstd.view(M)
    ct.launch(
        stream,
        (num_row_blocks, 1, 1),
        _rmsnorm_bwd_row_kernel,
        (scale_view, weight, rstd_view, x, out, partials, ROW_BLOCK, N),
    )
    blocks_p2 = _ceil_pow2(num_row_blocks)
    ct.launch(
        stream,
        (N, 1, 1),
        _finalize_cols_kernel,
        (partials, grad_weight, num_row_blocks, blocks_p2, N),
    )
    return out, grad_weight
