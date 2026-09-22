"""cuTile port of sum_1e8a518eb72e: Demucs masked slice-scatter + column sum.

Two cuTile kernels matching Triton's structure:
  1. _where_partial_kernel: reads src/mask/scalar, computes `where(mask, fill, src)`
     with the mask-aware scalar-fill, writes both the padded slice_scatter output
     (at PAD_LEFT-shifted position) and the `where` output, and produces per-
     channel per-tile partial sums.
  2. _finalize_sum_kernel: reduces the partials to the final [C] f32 sum.

The slice_scatter zero-padding outside the source region is initialized via
torch.zeros before the kernel (pure memory setup, not a compute/reduction op).
The reductions themselves are entirely in cuTile — no host `.sum(...)`.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4
C = 128
W = 23212
PAD_LEFT = 355
PADDED_W = 23923
BLOCK_C = 4
BLOCK_N = 1024
FINAL_BLOCK = 128  # power of 2 >= N * n_col_tiles


@ct.kernel
def _where_partial_kernel(
    src_ptr,        # bf16 [N, C, W]
    mask_ptr,       # b8   [N, C, W]  (may have padded strides)
    scalar_ptr,     # bf16 [1]
    padded_ptr,     # bf16 [N, C, PADDED_W]  (pre-zeroed outside source region)
    where_ptr,      # bf16 [N, C, W]
    partial_ptr,    # f32  [N, C, n_col_tiles]
    W_C: ct.Constant[int],
    PAD_LEFT_C: ct.Constant[int],
    BLOCK_C_C: ct.Constant[int],
    BLOCK_N_C: ct.Constant[int],
):
    ch_block = ct.bid(0)
    batch = ct.bid(1)
    col_tile = ct.bid(2)

    src = ct.load(
        src_ptr, index=(batch, ch_block, col_tile),
        shape=(1, BLOCK_C_C, BLOCK_N_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    m = ct.load(
        mask_ptr, index=(batch, ch_block, col_tile),
        shape=(1, BLOCK_C_C, BLOCK_N_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    scalar = ct.load(scalar_ptr, index=(0,), shape=(1,))

    src_2d = ct.reshape(src, (BLOCK_C_C, BLOCK_N_C))
    m_2d = ct.reshape(m, (BLOCK_C_C, BLOCK_N_C))
    scalar_2d = ct.reshape(scalar, (1, 1))
    scalar_bcast = ct.full((BLOCK_C_C, BLOCK_N_C), 0.0, dtype=ct.bfloat16) + scalar_2d

    where_val = ct.where(m_2d, scalar_bcast, src_2d)

    col_idx = ct.arange(BLOCK_N_C, dtype=ct.int32) + col_tile * BLOCK_N_C
    col_valid_1d = col_idx < W_C
    col_valid_2d = ct.reshape(col_valid_1d, (1, BLOCK_N_C))
    true_2d = ct.full((BLOCK_C_C, BLOCK_N_C), True, dtype=ct.bool_)
    false_2d = ct.full((BLOCK_C_C, BLOCK_N_C), False, dtype=ct.bool_)
    valid_2d = ct.where(col_valid_2d, true_2d, false_2d)

    # Partial sum: cast to f32, mask OOB cols, reduce over col axis (axis=1).
    where_f = ct.astype(where_val, ct.float32)
    zero_f = ct.zeros((BLOCK_C_C, BLOCK_N_C), dtype=ct.float32)
    where_f_masked = ct.where(valid_2d, where_f, zero_f)
    partial = ct.sum(where_f_masked, axis=1)  # (BLOCK_C,)
    partial_3d = ct.reshape(partial, (1, BLOCK_C_C, 1))
    ct.store(partial_ptr, index=(batch, ch_block, col_tile), tile=partial_3d)

    # Scatter where_out and padded_out.
    ch_offsets_1d = ct.arange(BLOCK_C_C, dtype=ct.int32) + ch_block * BLOCK_C_C
    ch_offsets_2d = ct.reshape(ch_offsets_1d, (BLOCK_C_C, 1))
    col_offsets_2d = ct.reshape(col_idx, (1, BLOCK_N_C))
    ch_full = ct.full((BLOCK_C_C, BLOCK_N_C), 0, dtype=ct.int32) + ch_offsets_2d
    col_full = ct.full((BLOCK_C_C, BLOCK_N_C), 0, dtype=ct.int32) + col_offsets_2d
    batch_full = ct.full((BLOCK_C_C, BLOCK_N_C), batch, dtype=ct.int32)

    ct.scatter(where_ptr, (batch_full, ch_full, col_full), where_val, mask=valid_2d)
    padded_col = col_full + PAD_LEFT_C
    ct.scatter(padded_ptr, (batch_full, ch_full, padded_col), src_2d, mask=valid_2d)


@ct.kernel
def _finalize_sum_kernel(
    partial_ptr,   # f32 [N, C, n_col_tiles]
    sum_ptr,       # f32 [C]
    N_C: ct.Constant[int],
    N_TILES_C: ct.Constant[int],
    FINAL_BLOCK_C: ct.Constant[int],
    BLOCK_C_C: ct.Constant[int],
):
    ch_block = ct.bid(0)
    # Load partial[:, ch_block*BLOCK_C:(ch_block+1)*BLOCK_C, :] padded to
    # (N, BLOCK_C, FINAL_BLOCK). We use padding_mode ZERO for tile-cols > n_col_tiles.
    tile = ct.load(
        partial_ptr, index=(0, ch_block, 0),
        shape=(N_C, BLOCK_C_C, FINAL_BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    # Mask out cols beyond n_col_tiles (padding is zero, so the sum is
    # already correct — but be explicit for safety).
    col_idx = ct.arange(FINAL_BLOCK_C, dtype=ct.int32)
    col_valid = ct.reshape(col_idx < N_TILES_C, (1, 1, FINAL_BLOCK_C))
    zero = ct.zeros((N_C, BLOCK_C_C, FINAL_BLOCK_C), dtype=ct.float32)
    masked = ct.where(col_valid, tile, zero)
    # Sum over axes 0 and 2 → (BLOCK_C,)
    total_2d = ct.sum(masked, axis=2)  # (N, BLOCK_C)
    total = ct.sum(total_2d, axis=0)   # (BLOCK_C,)
    ct.store(sum_ptr, index=(ch_block,), tile=total)


@oracle_impl(hardware="B200", point="aa121e37")
def oracle_forward(inputs):
    src, mask, scalar, shape0 = inputs
    device = src.device
    del shape0

    n_col_tiles = (W + BLOCK_N - 1) // BLOCK_N  # 23
    # Pre-zero the padded output; the kernel only writes the source region.
    padded = torch.zeros(
        (N, C, PADDED_W), device=device, dtype=torch.bfloat16,
    )
    where_out = torch.empty(
        (N, C, W), device=device, dtype=torch.bfloat16,
    )
    partial = torch.empty(
        (N, C, n_col_tiles), device=device, dtype=torch.float32,
    )
    sum_out = torch.empty((C,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C // BLOCK_C, N, n_col_tiles),
        _where_partial_kernel,
        (src, mask, scalar.view(1), padded, where_out, partial,
         W, PAD_LEFT, BLOCK_C, BLOCK_N),
    )
    ct.launch(
        stream,
        (C // BLOCK_C, 1, 1),
        _finalize_sum_kernel,
        (partial, sum_out, N, n_col_tiles, FINAL_BLOCK, BLOCK_C),
    )
    return padded, where_out, sum_out
