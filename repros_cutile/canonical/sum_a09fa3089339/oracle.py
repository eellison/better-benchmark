"""cuTile port of sum_a09fa3089339: NFNet broadcast-scale GELU-gradient.

Two-stage port mirroring Triton's split-K produce+finalize structure:
* _materialize_partial_kernel: fused broadcast-scale + in-kernel Abramowitz-
  Stegun erf polynomial + exp + bf16 rounding + per-channel partial sum.
* _final_sum_kernel: sum partials across row groups, round through bf16.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
CHANNELS = 3072
HEIGHT = 6
WIDTH = 6
HW = HEIGHT * WIDTH
ROWS = BATCH * HW
GAMMA = 1.7015043497085571
RSQRT2 = 0.7071067811865476
NORMAL_PDF_SCALE = 0.3989422804014327


def _ceil_pow2(value: int) -> int:
    return 1 << (value - 1).bit_length()


@ct.kernel
def _materialize_partial_kernel(
    scale_ptr,      # bf16 [BATCH, CHANNELS]
    rhs_ptr,        # bf16 [ROWS, CHANNELS] contiguous (channels-last flattened)
    out_ptr,        # bf16 [BATCH, CHANNELS, HW] contiguous (view of out)
    partial_ptr,    # f32 [num_groups, CHANNELS]
    ROWS_: ct.Constant[int],
    CHANNELS_: ct.Constant[int],
    HW_: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row_group = ct.bid(0)
    channel_group = ct.bid(1)

    # Load rhs tile: (ROW_BLOCK, BLOCK_C) from flattened (ROWS, CHANNELS).
    rhs = ct.load(rhs_ptr, index=(row_group, channel_group),
                  shape=(ROW_BLOCK, BLOCK_C),
                  padding_mode=ct.PaddingMode.ZERO)
    rhs_f = ct.astype(rhs, ct.float32)

    # Compute row-owned n index for each row in the tile and load scale[n, c]
    # via reshape trick: we need scale broadcast across HW dim. Simpler: since
    # rows = n * HW + hw, and HW=36 is compile-time-known, we compute n =
    # rows // HW. To gather scale, we load a (ROW_BLOCK, BLOCK_C) tile from
    # scale using row_group's n mapping via 1D gather. However, cuTile's tile
    # loads are structured — the simplest legal path is to load scale in a
    # (ROW_BLOCK, BLOCK_C) shape derived from a pre-broadcast (ROWS, CHANNELS)
    # scale tensor built in oracle_forward.
    scale = ct.load(scale_ptr, index=(row_group, channel_group),
                    shape=(ROW_BLOCK, BLOCK_C),
                    padding_mode=ct.PaddingMode.ZERO)
    scale_f = ct.astype(scale, ct.float32)
    div_bf = ct.astype(scale_f * (1.0 / 36.0), ct.bfloat16)
    div_f = ct.astype(div_bf, ct.float32)
    scaled_bf = ct.astype(div_f * GAMMA, ct.bfloat16)
    scaled_f = ct.astype(scaled_bf, ct.float32)

    # In-kernel erf via Abramowitz-Stegun 7.1.26 polynomial approximation.
    erf_arg = rhs_f * RSQRT2
    zero_f = ct.zeros((ROW_BLOCK, BLOCK_C), dtype=ct.float32)
    sign = ct.where(erf_arg >= zero_f,
                    ct.full((ROW_BLOCK, BLOCK_C), 1.0, dtype=ct.float32),
                    ct.full((ROW_BLOCK, BLOCK_C), -1.0, dtype=ct.float32))
    abs_arg = ct.where(erf_arg >= zero_f, erf_arg, -erf_arg)
    t = 1.0 / (1.0 + 0.3275911 * abs_arg)
    poly = (((((1.061405429 * t) - 1.453152027) * t + 1.421413741) * t
             - 0.284496736) * t + 0.254829592) * t
    erf_val = sign * (1.0 - poly * ct.exp(-abs_arg * abs_arg))

    cdf = (erf_val + 1.0) * 0.5
    exp_term = ct.exp((rhs_f * rhs_f) * -0.5)
    pdf_term = rhs_f * (exp_term * NORMAL_PDF_SCALE)
    derivative = cdf + pdf_term
    value_bf16 = ct.astype(scaled_f * derivative, ct.bfloat16)

    # Store out: out is viewed as (ROWS, CHANNELS) channels-last-flatten via
    # a matching torch view built in oracle_forward.
    ct.store(out_ptr, index=(row_group, channel_group), tile=value_bf16)

    # Per-channel partial sum
    row_base = row_group * ROW_BLOCK
    col_base = channel_group * BLOCK_C
    row_idx = ct.arange(ROW_BLOCK, dtype=ct.int32)
    col_idx = ct.arange(BLOCK_C, dtype=ct.int32)
    row_valid = ct.reshape((row_base + row_idx) < ROWS_, (ROW_BLOCK, 1))
    col_valid = ct.reshape((col_base + col_idx) < CHANNELS_, (1, BLOCK_C))
    mask = row_valid & col_valid

    val_f = ct.astype(value_bf16, ct.float32)
    masked = ct.where(mask, val_f, 0.0)
    partial = ct.sum(masked, axis=0)

    partial_2d = ct.reshape(partial, (1, BLOCK_C))
    ct.store(partial_ptr, index=(row_group, channel_group), tile=partial_2d)


@ct.kernel
def _final_sum_kernel(
    partial_ptr,
    sum_ptr,
    CHANNELS_: ct.Constant[int],
    NUM_GROUPS: ct.Constant[int],
    GROUP_BLOCK: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    channel_group = ct.bid(0)
    values = ct.load(partial_ptr, index=(0, channel_group),
                     shape=(GROUP_BLOCK, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
    values_f = ct.astype(values, ct.float32)

    col_base = channel_group * BLOCK_C
    group_idx = ct.arange(GROUP_BLOCK, dtype=ct.int32)
    col_idx = ct.arange(BLOCK_C, dtype=ct.int32)
    group_valid = ct.reshape(group_idx < NUM_GROUPS, (GROUP_BLOCK, 1))
    col_valid = ct.reshape((col_base + col_idx) < CHANNELS_, (1, BLOCK_C))
    mask = group_valid & col_valid

    masked = ct.where(mask, values_f, 0.0)
    total = ct.sum(masked, axis=0)
    rounded = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    ct.store(sum_ptr, index=(channel_group,), tile=rounded)


@oracle_impl(hardware="B200", point="74c4e87d", ROW_BLOCK=64, BLOCK_C=64, FINAL_BLOCK_C=32)
def oracle_forward(inputs, *, ROW_BLOCK: int, BLOCK_C: int, FINAL_BLOCK_C: int):
    arg0_1, arg1_1, *_shape_params = inputs
    device = arg0_1.device

    # rhs (arg1_1) is bf16 [BATCH, CHANNELS, HEIGHT, WIDTH] in channels-last
    # stride: stride=(C*HW, HW, W, 1). To flatten as (ROWS, CHANNELS) — which
    # is Triton's row-major access pattern — we permute (0,2,3,1) so the
    # trailing dim is the fast channel axis.
    rhs_mc = arg1_1.permute(0, 2, 3, 1).reshape(ROWS, CHANNELS).contiguous()

    # scale (arg0_1) is bf16 [BATCH, CHANNELS]. Broadcast to (ROWS, CHANNELS)
    # by repeating along HW dim.
    scale_bc = arg0_1.unsqueeze(1).expand(BATCH, HW, CHANNELS).reshape(ROWS, CHANNELS).contiguous()

    out = torch.empty_strided(
        (BATCH, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HW, HW, WIDTH, 1),
        device=device, dtype=torch.bfloat16,
    )
    # Contiguous view (BATCH*HW, CHANNELS) matching rhs_mc layout. Since out
    # is strided [C*HW, HW, W, 1] (not channels-last!), we need a view. Build
    # a separate contiguous work buffer, then copy back at the end.
    out_work = torch.empty((ROWS, CHANNELS), device=device, dtype=torch.bfloat16)

    num_groups = ct.cdiv(ROWS, ROW_BLOCK)
    partial = torch.empty((num_groups, CHANNELS), device=device, dtype=torch.float32)
    sum_out = torch.empty((CHANNELS,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_groups, ct.cdiv(CHANNELS, BLOCK_C), 1),
        _materialize_partial_kernel,
        (scale_bc, rhs_mc, out_work, partial,
         ROWS, CHANNELS, HW, ROW_BLOCK, BLOCK_C),
    )

    group_block = _ceil_pow2(num_groups)
    ct.launch(
        stream,
        (ct.cdiv(CHANNELS, FINAL_BLOCK_C), 1, 1),
        _final_sum_kernel,
        (partial, sum_out, CHANNELS, num_groups, group_block, FINAL_BLOCK_C),
    )

    # Reshape out_work (ROWS, CHANNELS) into the (BATCH, CHANNELS, HEIGHT, WIDTH)
    # layout expected by the returned tensor. out_work is contiguous with
    # row-major (BATCH, HW, CHANNELS); the target `out` is layout
    # [C*HW, HW, W, 1] — that's (BATCH, CHANNELS, HEIGHT, WIDTH) with channel-
    # major within each batch item. So we need to permute back.
    out.copy_(out_work.view(BATCH, HW, CHANNELS)
              .permute(0, 2, 1)
              .view(BATCH, CHANNELS, HEIGHT, WIDTH))

    return out, sum_out
