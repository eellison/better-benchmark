"""cuTile port of sum_sum_48f6a6fbbd3e: DenseNet BN-backward.

Matches Triton's 3-kernel structure:
1. `_partial_reduce_kernel` — split-K per-channel partials via `ct.sum` in-kernel.
2. `_finalize_kernel` — reduces per-channel partials via `ct.sum` and emits the
   BN-backward scalars (sum_out, scaled_dot_out, mean_term, correction_scale,
   output_scale).
3. `_epilogue_kernel` — dense bf16 gradient with slice-add.

All reductions stay inside `@ct.kernel` bodies via `ct.sum(...)` — no torch reductions.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 7.62939453125e-06
SLICE_START = 16  # channels of arg0 offset by which slice starts
EPILOGUE_BLOCK_C = 8


def _next_pow2(v: int) -> int:
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _partial_reduce_kernel(
    mask_ptr,        # bf16 [N, C, HW]
    fill_ptr,        # bf16 [1]
    rhs_ptr,         # bf16 [N, C, HW]
    activation_ptr,  # bf16 [N, C, HW]
    mean_ptr,        # f32 [C]  (from arg5 squeezed)
    partial_sum_ptr, # f32 [num_k_tiles, C]  partial sums of selected
    partial_dot_ptr, # f32 [num_k_tiles, C]  partial sums of selected * centered
    N: ct.Constant[int],
    C: ct.Constant[int],
    HW: ct.Constant[int],
    K_TOTAL: ct.Constant[int],  # N*HW
    BLOCK_K: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    """Grid: (C // BLOCK_C, num_k_tiles, 1)."""
    c_block = ct.bid(0)
    k_block = ct.bid(1)

    c_offsets = c_block * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    k_offsets = k_block * BLOCK_K + ct.arange(BLOCK_K, dtype=ct.int32)
    n_offsets = k_offsets // HW
    hw_offsets = k_offsets - n_offsets * HW
    k_mask = k_offsets < K_TOTAL

    n_2d = ct.reshape(n_offsets, (BLOCK_K, 1))
    hw_2d = ct.reshape(hw_offsets, (BLOCK_K, 1))
    c_2d = ct.reshape(c_offsets, (1, BLOCK_C))
    n_bc = ct.broadcast_to(n_2d, (BLOCK_K, BLOCK_C))
    hw_bc = ct.broadcast_to(hw_2d, (BLOCK_K, BLOCK_C))
    c_bc = ct.broadcast_to(c_2d, (BLOCK_K, BLOCK_C))

    mask_val = ct.astype(ct.gather(mask_ptr, (n_bc, c_bc, hw_bc)), ct.float32)
    rhs_val = ct.astype(ct.gather(rhs_ptr, (n_bc, c_bc, hw_bc)), ct.float32)
    fill = ct.astype(ct.load(fill_ptr, index=(0,), shape=(1,)), ct.float32)
    fill_scalar = ct.reshape(fill, ())
    selected = ct.where(mask_val <= 0.0, fill_scalar, rhs_val)

    activation = ct.astype(ct.gather(activation_ptr, (n_bc, c_bc, hw_bc)), ct.float32)
    mean = ct.astype(ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,)), ct.float32)
    mean_bc = ct.reshape(mean, (1, BLOCK_C))
    centered = activation - mean_bc
    dot = selected * centered

    k_mask_2d = ct.reshape(k_mask, (BLOCK_K, 1))
    selected_masked = ct.where(k_mask_2d, selected, 0.0)
    dot_masked = ct.where(k_mask_2d, dot, 0.0)

    partial_sum = ct.sum(selected_masked, axis=0)  # (BLOCK_C,)
    partial_dot = ct.sum(dot_masked, axis=0)  # (BLOCK_C,)

    ct.store(partial_sum_ptr, index=(k_block, c_block), tile=ct.reshape(partial_sum, (1, BLOCK_C)))
    ct.store(partial_dot_ptr, index=(k_block, c_block), tile=ct.reshape(partial_dot, (1, BLOCK_C)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,      # f32 [num_k_tiles, C]
    partial_dot_ptr,      # f32 [num_k_tiles, C]
    invstd_ptr,           # f32 [C]
    weight_ptr,           # f32 [C]
    sum_out_ptr,          # f32 [C]
    scaled_dot_out_ptr,   # f32 [C]
    mean_term_ptr,        # f32 [C]
    correction_scale_ptr, # f32 [C]
    output_scale_ptr,     # f32 [C]
    C: ct.Constant[int],
    NUM_K_TILES: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
):
    """Grid: (C, 1, 1). Per-channel column reduction using in-kernel ct.sum."""
    c = ct.bid(0)

    # Load column `c` across all row tiles (rows dim padded with zeros past NUM_K_TILES).
    tile_col_sum = ct.load(
        partial_sum_ptr, index=(0, c), shape=(BLOCK_TILES, 1),
        padding_mode=ct.PaddingMode.ZERO,
    )
    tile_col_dot = ct.load(
        partial_dot_ptr, index=(0, c), shape=(BLOCK_TILES, 1),
        padding_mode=ct.PaddingMode.ZERO,
    )
    rows = ct.arange(BLOCK_TILES, dtype=ct.int32)
    active = ct.reshape(rows < NUM_K_TILES, (BLOCK_TILES, 1))
    tile_col_sum = ct.where(active, tile_col_sum, 0.0)
    tile_col_dot = ct.where(active, tile_col_dot, 0.0)

    sum_value = ct.reshape(ct.sum(tile_col_sum), ())
    dot_value = ct.reshape(ct.sum(tile_col_dot), ())

    invstd = ct.reshape(ct.load(invstd_ptr, index=(c,), shape=(1,)), ())
    weight = ct.reshape(ct.load(weight_ptr, index=(c,), shape=(1,)), ())
    invstd_sq = invstd * invstd
    dot_mean = dot_value * SCALE
    correction_scale = dot_mean * invstd_sq
    output_scale = invstd * weight

    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(sum_value, (1,)))
    ct.store(scaled_dot_out_ptr, index=(c,), tile=ct.reshape(dot_value * invstd, (1,)))
    ct.store(mean_term_ptr, index=(c,), tile=ct.reshape(sum_value * SCALE, (1,)))
    ct.store(correction_scale_ptr, index=(c,), tile=ct.reshape(correction_scale, (1,)))
    ct.store(output_scale_ptr, index=(c,), tile=ct.reshape(output_scale, (1,)))


@ct.kernel
def _epilogue_kernel(
    sliced_ptr,          # bf16 [N, 80, HW]  arg0 flattened
    mask_ptr,            # bf16 [N, C, HW]
    fill_ptr,            # bf16 [1]
    rhs_ptr,             # bf16 [N, C, HW]
    activation_ptr,      # bf16 [N, C, HW]
    mean_ptr,            # f32 [C]
    mean_term_ptr,       # f32 [C]  (sum_1 * SCALE)
    correction_scale_ptr,# f32 [C]  (sum_2 * SCALE * rsqrt^2)
    output_scale_ptr,    # f32 [C]  (rsqrt * weight)
    out_ptr,             # bf16 [N, C, HW]
    N: ct.Constant[int],
    C: ct.Constant[int],
    HW: ct.Constant[int],
    SLICE_OFFSET_C: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
    BLOCK_C_C: ct.Constant[int],
):
    """Grid: (C // BLOCK_C_C, N * num_hw_tiles, 1)."""
    c_block = ct.bid(0)
    nk_block = ct.bid(1)
    num_hw_tiles = HW // BLOCK_HW
    n = nk_block // num_hw_tiles
    hw_block = nk_block - n * num_hw_tiles

    mask_val = ct.astype(
        ct.load(mask_ptr, index=(n, c_block, hw_block), shape=(1, BLOCK_C_C, BLOCK_HW)),
        ct.float32,
    )
    rhs_val = ct.astype(
        ct.load(rhs_ptr, index=(n, c_block, hw_block), shape=(1, BLOCK_C_C, BLOCK_HW)),
        ct.float32,
    )
    fill = ct.astype(ct.load(fill_ptr, index=(0,), shape=(1,)), ct.float32)
    fill_scalar = ct.reshape(fill, ())
    selected = ct.where(mask_val <= 0.0, fill_scalar, rhs_val)

    activation = ct.astype(
        ct.load(activation_ptr, index=(n, c_block, hw_block), shape=(1, BLOCK_C_C, BLOCK_HW)),
        ct.float32,
    )
    mean = ct.astype(ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C_C,)), ct.float32)
    mean_bc = ct.reshape(mean, (1, BLOCK_C_C, 1))
    centered = activation - mean_bc

    correction_scale = ct.astype(
        ct.load(correction_scale_ptr, index=(c_block,), shape=(BLOCK_C_C,)),
        ct.float32,
    )
    output_scale = ct.astype(
        ct.load(output_scale_ptr, index=(c_block,), shape=(BLOCK_C_C,)),
        ct.float32,
    )
    mean_term = ct.astype(
        ct.load(mean_term_ptr, index=(c_block,), shape=(BLOCK_C_C,)),
        ct.float32,
    )
    correction_scale_bc = ct.reshape(correction_scale, (1, BLOCK_C_C, 1))
    output_scale_bc = ct.reshape(output_scale, (1, BLOCK_C_C, 1))
    mean_term_bc = ct.reshape(mean_term, (1, BLOCK_C_C, 1))

    correction = centered * correction_scale_bc
    grad = (selected - correction - mean_term_bc) * output_scale_bc
    grad_bf = ct.astype(grad, ct.bfloat16)

    slice_c_block = c_block + SLICE_OFFSET_C // BLOCK_C_C
    sliced = ct.astype(
        ct.load(sliced_ptr, index=(n, slice_c_block, hw_block), shape=(1, BLOCK_C_C, BLOCK_HW)),
        ct.float32,
    )
    add_val = ct.astype(sliced + ct.astype(grad_bf, ct.float32), ct.bfloat16)

    ct.store(out_ptr, index=(n, c_block, hw_block), tile=add_val)


@oracle_impl(hardware="B200", point="b6ca30b3", BLOCK_K=1024, BLOCK_C=8, BLOCK_HW=256)
def oracle_forward(inputs, *, BLOCK_K: int, BLOCK_C: int, BLOCK_HW: int):
    arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7 = inputs
    n, c, h, w = int(arg1.shape[0]), int(arg1.shape[1]), int(arg1.shape[2]), int(arg1.shape[3])
    hw = h * w
    input_c = int(arg0.shape[1])
    k_total = n * hw
    num_k_tiles = (k_total + BLOCK_K - 1) // BLOCK_K
    device = arg1.device

    if input_c != 80 or c != 64 or h != 32 or w != 32:
        raise NotImplementedError(f"unexpected shape ({n},{input_c}->{c},{h},{w})")
    if c % BLOCK_C != 0:
        raise NotImplementedError(f"C={c} must divide BLOCK_C={BLOCK_C}")
    if hw % BLOCK_HW != 0:
        raise NotImplementedError(f"HW={hw} must divide BLOCK_HW={BLOCK_HW}")

    mean_1d = arg5.view(c).contiguous()

    partial_sum = torch.empty((num_k_tiles, c), device=device, dtype=torch.float32)
    partial_dot = torch.empty((num_k_tiles, c), device=device, dtype=torch.float32)

    mask_3d = arg1.view(n, c, hw)
    rhs_3d = arg3.view(n, c, hw)
    activation_3d = arg4.view(n, c, hw)

    fill_1 = arg2.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (c // BLOCK_C, num_k_tiles, 1),
        _partial_reduce_kernel,
        (
            mask_3d, fill_1, rhs_3d, activation_3d, mean_1d,
            partial_sum, partial_dot,
            n, c, hw, k_total, BLOCK_K, BLOCK_C,
        ),
    )

    # In-kernel finalize: matches Triton's `_finalize_kernel` with `tl.sum`.
    sum_out = torch.empty((c,), device=device, dtype=torch.float32)
    scaled_dot = torch.empty((c,), device=device, dtype=torch.float32)
    mean_term = torch.empty((c,), device=device, dtype=torch.float32)
    correction_scale = torch.empty((c,), device=device, dtype=torch.float32)
    output_scale = torch.empty((c,), device=device, dtype=torch.float32)
    block_tiles = _next_pow2(num_k_tiles)
    ct.launch(
        stream,
        (c, 1, 1),
        _finalize_kernel,
        (
            partial_sum, partial_dot, arg6.contiguous(), arg7.contiguous(),
            sum_out, scaled_dot, mean_term, correction_scale, output_scale,
            c, num_k_tiles, block_tiles,
        ),
    )

    # Epilogue kernel produces the add_out (bf16 [N, C, H, W]).
    add_out = torch.empty_strided(
        (n, c, h, w), (c * hw, hw, w, 1),
        device=device, dtype=torch.bfloat16,
    )

    mask_ep = arg1.view(n, c, hw)
    rhs_ep = arg3.view(n, c, hw)
    activation_ep = arg4.view(n, c, hw)
    sliced_ep = arg0.view(n, input_c, hw)
    add_out_3d = add_out.view(n, c, hw)

    if SLICE_START % EPILOGUE_BLOCK_C != 0:
        raise NotImplementedError(
            f"SLICE_START={SLICE_START} must be divisible by EPILOGUE_BLOCK_C={EPILOGUE_BLOCK_C}"
        )

    ct.launch(
        stream,
        (c // EPILOGUE_BLOCK_C, n * (hw // BLOCK_HW), 1),
        _epilogue_kernel,
        (
            sliced_ep, mask_ep, fill_1, rhs_ep, activation_ep,
            mean_1d, mean_term, correction_scale, output_scale, add_out_3d,
            n, c, hw, SLICE_START, BLOCK_HW, EPILOGUE_BLOCK_C,
        ),
    )

    return sum_out, scaled_dot, add_out, add_out[:, :SLICE_START, :, :]
