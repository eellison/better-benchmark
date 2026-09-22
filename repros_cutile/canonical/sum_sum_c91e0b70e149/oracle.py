"""cuTile port of sum_sum_c91e0b70e149 (COOPERATIVE_SPLIT_K): GhostNet BN
backward (slice/add/mask producer + channel reductions + dense epilogue).

Mirrors Triton's 3-kernel split-K structure: partial_reduce → finalize → epilogue.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 512
IN_CHANNELS = 120
CHANNELS = 60
HEIGHT = 28
WIDTH = 28
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = BATCH * HW  # 401408
NUMEL = ELEMENTS_PER_CHANNEL * CHANNELS
REDUCE_SCALE = 2.4912308673469386e-06


def _next_pow2(v):
    return 1 << (int(v) - 1).bit_length()


@ct.kernel
def _partial_reduce_kernel(
    wide_ptr,               # bf16 (rows, IN_C=120) channels-last flattened
    residual_ptr,           # bf16 (rows, C=60) channels-last flattened
    mask_ptr,               # bf16 (rows, C) channels-last flattened
    fill_ptr,               # bf16 (1,)
    centered_source_ptr,    # bf16 (rows, C) channels-last flattened
    mean_ptr,               # f32 (C,)
    partial_sum_ptr,        # f32 (num_chunks, C)
    partial_prod_ptr,       # f32 (num_chunks, C)
    ELEMENTS_PER_CHANNEL_: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    r_block = ct.bid(1)

    # Load wide[r, c] with per-channel stride. wide is contiguous (rows, 120).
    # Trick: use ct.load with shape (BLOCK_R, BLOCK_C) but tile-space index
    # requires row-major flat. The wide tensor is (rows, IN_C) with stride (IN_C, 1).
    # For simplicity, use gather with row_major offsets.
    rows_1d = r_block * BLOCK_R + ct.arange(BLOCK_R, dtype=ct.int32)
    cols_1d = c_block * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    rows_2d = ct.reshape(rows_1d, (BLOCK_R, 1))
    cols_2d = ct.reshape(cols_1d, (1, BLOCK_C))
    zero_i = ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.int32)
    rows_full = rows_2d + zero_i
    cols_full = cols_2d + zero_i

    active_r = rows_1d < ELEMENTS_PER_CHANNEL_
    active_c = cols_1d < CHANNELS
    active = ct.reshape(active_r, (BLOCK_R, 1)) & ct.reshape(active_c, (1, BLOCK_C))

    # wide_offsets = rows * IN_C + cols (wide has IN_C=120)
    wide_off = rows_full * IN_CHANNELS + cols_full
    narrow_off = rows_full * CHANNELS + cols_full
    zero_off = ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.int32)
    safe_wide = ct.where(active, wide_off, zero_off)
    safe_narrow = ct.where(active, narrow_off, zero_off)

    lhs_bf = ct.gather(wide_ptr, safe_wide)
    rhs_bf = ct.gather(residual_ptr, safe_narrow)
    lhs_f = ct.astype(lhs_bf, ct.float32)
    rhs_f = ct.astype(rhs_bf, ct.float32)
    # bf16 add with round-trip to match Triton default (non-inductor path).
    add_bf = ct.astype(lhs_f + rhs_f, ct.bfloat16)
    add_f = ct.astype(add_bf, ct.float32)

    mask_bf = ct.gather(mask_ptr, safe_narrow)
    mask_f = ct.astype(mask_bf, ct.float32)
    zero_f = ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.float32)
    fill_bf_1 = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_f = ct.astype(fill_bf_1, ct.float32)
    fill_f_b = ct.reshape(fill_f, (1, 1)) + zero_f
    where_value = ct.where(mask_f <= zero_f, fill_f_b, add_f)
    where_value = ct.where(active, where_value, zero_f)

    centered_source_bf = ct.gather(centered_source_ptr, safe_narrow)
    centered_source_f = ct.astype(centered_source_bf, ct.float32)
    mean = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,))
    mean_2d = ct.reshape(mean, (1, BLOCK_C)) + zero_f
    centered = centered_source_f - mean_2d
    prod = where_value * ct.where(active, centered, zero_f)

    partial_sum = ct.sum(where_value, axis=0)   # (BLOCK_C,)
    partial_prod = ct.sum(prod, axis=0)

    ct.store(partial_sum_ptr, index=(r_block, c_block),
             tile=ct.reshape(partial_sum, (1, BLOCK_C)))
    ct.store(partial_prod_ptr, index=(r_block, c_block),
             tile=ct.reshape(partial_prod, (1, BLOCK_C)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,      # f32 (num_chunks, C)
    partial_prod_ptr,     # f32 (num_chunks, C)
    invstd_ptr,           # f32 (C,)
    weight_ptr,           # f32 (C,)
    sum_out_ptr,          # f32 (C,)
    prod_out_ptr,         # f32 (C,)
    mean_term_ptr,        # f32 (C,)
    prod_coeff_ptr,       # f32 (C,)
    output_scale_ptr,     # f32 (C,)
    BLOCK_CHUNKS: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    ps = ct.load(partial_sum_ptr, index=(0, c_block),
                 shape=(BLOCK_CHUNKS, BLOCK_C),
                 padding_mode=ct.PaddingMode.ZERO)
    pp = ct.load(partial_prod_ptr, index=(0, c_block),
                 shape=(BLOCK_CHUNKS, BLOCK_C),
                 padding_mode=ct.PaddingMode.ZERO)
    sum_value = ct.sum(ps, axis=0)   # (BLOCK_C,)
    prod_value = ct.sum(pp, axis=0)
    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C,))
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C,))

    mean_term = sum_value * REDUCE_SCALE
    prod_scaled = prod_value * REDUCE_SCALE
    invstd_sq = invstd * invstd
    prod_coeff = prod_scaled * invstd_sq
    output_scale = invstd * weight
    prod_out = prod_value * invstd

    ct.store(sum_out_ptr, index=(c_block,), tile=sum_value)
    ct.store(prod_out_ptr, index=(c_block,), tile=prod_out)
    ct.store(mean_term_ptr, index=(c_block,), tile=mean_term)
    ct.store(prod_coeff_ptr, index=(c_block,), tile=prod_coeff)
    ct.store(output_scale_ptr, index=(c_block,), tile=output_scale)


@ct.kernel
def _epilogue_kernel(
    wide_ptr,               # bf16 flat
    residual_ptr,           # bf16 flat
    mask_ptr,               # bf16 flat
    fill_ptr,               # bf16 (1,)
    centered_source_ptr,    # bf16 flat
    mean_ptr,               # f32 (C,)
    mean_term_ptr,          # f32 (C,)
    prod_coeff_ptr,         # f32 (C,)
    output_scale_ptr,       # f32 (C,)
    out_ptr,                # bf16 flat (numel,) channel-major channels-last order
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    active = offsets < NUMEL
    zero_i = ct.zeros((BLOCK,), dtype=ct.int32)
    safe_off = ct.where(active, offsets, zero_i)
    channel = safe_off - (safe_off // CHANNELS) * CHANNELS
    reduce_index = safe_off // CHANNELS
    wide_off = reduce_index * IN_CHANNELS + channel

    lhs_bf = ct.gather(wide_ptr, wide_off)
    rhs_bf = ct.gather(residual_ptr, safe_off)
    lhs_f = ct.astype(lhs_bf, ct.float32)
    rhs_f = ct.astype(rhs_bf, ct.float32)
    add_bf = ct.astype(lhs_f + rhs_f, ct.bfloat16)
    add_f = ct.astype(add_bf, ct.float32)

    mask_bf = ct.gather(mask_ptr, safe_off)
    mask_f = ct.astype(mask_bf, ct.float32)
    zero_f = ct.zeros((BLOCK,), dtype=ct.float32)
    fill_bf_1 = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_f_scalar = ct.astype(fill_bf_1, ct.float32)
    fill_f_b = ct.reshape(fill_f_scalar, (1,)) + zero_f
    where_value = ct.where(mask_f <= zero_f, fill_f_b, add_f)

    centered_source_bf = ct.gather(centered_source_ptr, safe_off)
    centered_source_f = ct.astype(centered_source_bf, ct.float32)
    mean = ct.gather(mean_ptr, channel)
    centered = centered_source_f - mean

    prod_coeff = ct.gather(prod_coeff_ptr, channel)
    correction = centered * prod_coeff
    residual_v = where_value - correction

    mean_term = ct.gather(mean_term_ptr, channel)
    residual_v = residual_v - mean_term

    output_scale = ct.gather(output_scale_ptr, channel)
    out_value = residual_v * output_scale
    ct.scatter(out_ptr, (safe_off,), ct.astype(out_value, ct.bfloat16), mask=active)


@oracle_impl(hardware="B200", point="d3452c96", BLOCK_R=512, BLOCK_C=16, BLOCK_E=2048)
def oracle_forward(inputs, *, BLOCK_R: int, BLOCK_C: int, BLOCK_E: int):
    wide, residual, mask_input, fill, centered_source, mean, invstd, weight = inputs
    device = wide.device

    num_chunks = (ELEMENTS_PER_CHANNEL + BLOCK_R - 1) // BLOCK_R
    block_chunks = _next_pow2(num_chunks)

    # The four channels-last tensors [rows, C] view: get contiguous (rows, C).
    # wide is (B, IN_C=120, H, W) with strides (IN_C*HW, 1, IN_C*W, IN_C) — that's channels-last.
    # Permute (0,2,3,1) yields (B, H, W, IN_C) with contiguous strides.
    def _rows_c_view(t, cc):
        return t.permute(0, 2, 3, 1).contiguous().view(ELEMENTS_PER_CHANNEL, cc)

    wide_2d = _rows_c_view(wide, IN_CHANNELS)
    residual_2d = _rows_c_view(residual, CHANNELS)
    mask_2d = _rows_c_view(mask_input, CHANNELS)
    centered_2d = _rows_c_view(centered_source, CHANNELS)

    mean_1d = mean.view(CHANNELS)
    invstd_1d = invstd.view(CHANNELS)
    weight_1d = weight.view(CHANNELS)
    fill_1d = fill.view(1)

    partial_sum = torch.empty((num_chunks, CHANNELS), device=device, dtype=torch.float32)
    partial_prod = torch.empty((num_chunks, CHANNELS), device=device, dtype=torch.float32)
    sum_out = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    prod_out = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    mean_term = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    prod_coeff = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    output_scale = torch.empty((CHANNELS,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        ((CHANNELS + BLOCK_C - 1) // BLOCK_C, num_chunks, 1),
        _partial_reduce_kernel,
        (wide_2d.view(-1), residual_2d.view(-1), mask_2d.view(-1), fill_1d,
         centered_2d.view(-1), mean_1d, partial_sum, partial_prod,
         ELEMENTS_PER_CHANNEL, BLOCK_R, BLOCK_C),
    )
    ct.launch(
        stream,
        ((CHANNELS + BLOCK_C - 1) // BLOCK_C, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_prod, invstd_1d, weight_1d,
         sum_out, prod_out, mean_term, prod_coeff, output_scale,
         block_chunks, BLOCK_C),
    )

    # Output storage: channels-last strided [B, C, H, W]. We construct as
    # rows-major (rows, C) then permute back to NCHW channels-last stride view.
    out_nhwc = torch.empty((BATCH, HEIGHT, WIDTH, CHANNELS), device=device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        ((NUMEL + BLOCK_E - 1) // BLOCK_E, 1, 1),
        _epilogue_kernel,
        (wide_2d.view(-1), residual_2d.view(-1), mask_2d.view(-1), fill_1d,
         centered_2d.view(-1), mean_1d,
         mean_term, prod_coeff, output_scale,
         out_nhwc.view(-1),
         BLOCK_E),
    )
    out = out_nhwc.permute(0, 3, 1, 2)  # (B, C, H, W) channels-last view

    return sum_out, prod_out, out
