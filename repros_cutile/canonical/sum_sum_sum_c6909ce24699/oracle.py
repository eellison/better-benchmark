"""cuTile port of sum_sum_sum_c6909ce24699: masked-add BN-backward.

Three-kernel structure matching Triton:
  - _partial_kernel: compute where (add+where), partial channel sums
  - _finalize_kernel: reduce partial sums, compute stats
  - _epilogue_kernel: BN-backward pointwise
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
SCALE = 3.0517578125e-05


@ct.kernel
def _partial_kernel(
    arg0_ptr,        # bf16 [BATCH, C, H, W]
    arg1_ptr,        # bf16 [BATCH, C, H, W]
    arg2_ptr,        # bf16 [BATCH, C, H, W]
    scalar_ptr,      # bf16 [] (viewed as [1])
    arg4_ptr,        # bf16 [BATCH, C, H, W]
    mean_ptr,        # f32 [1, C, 1, 1]
    where_out_ptr,   # bf16 [BATCH, C, H, W]
    partial_sum_ptr, # f32 [BATCH, C]
    partial_dot_ptr, # f32 [BATCH, C]
    partial_bf16_sum_ptr,  # f32 [BATCH, C]
    C: ct.Constant[int],
    H: ct.Constant[int],
    W: ct.Constant[int],
):
    # grid = (BATCH, C)
    n = ct.bid(0)
    c = ct.bid(1)
    a0 = ct.load(arg0_ptr, index=(n, c, 0, 0), shape=(1, 1, H, W))
    a1 = ct.load(arg1_ptr, index=(n, c, 0, 0), shape=(1, 1, H, W))
    a2 = ct.load(arg2_ptr, index=(n, c, 0, 0), shape=(1, 1, H, W))
    a4 = ct.load(arg4_ptr, index=(n, c, 0, 0), shape=(1, 1, H, W))
    scalar = ct.load(scalar_ptr, index=(0,), shape=(1,))
    mean = ct.load(mean_ptr, index=(0, c, 0, 0), shape=(1, 1, 1, 1))

    add_bf16 = ct.astype(ct.astype(a0, ct.float32) + ct.astype(a1, ct.float32),
                         ct.bfloat16)
    a2_f = ct.astype(a2, ct.float32)
    zero_hw = ct.full((1, 1, H, W), 0.0, dtype=ct.float32)
    le = a2_f <= zero_hw
    scalar_val = ct.astype(scalar, ct.bfloat16)
    scalar_full = ct.reshape(scalar_val, (1, 1, 1, 1)) + ct.full((1, 1, H, W), 0.0, dtype=ct.bfloat16)
    where_bf16 = ct.where(le, scalar_full, add_bf16)
    where_f32 = ct.astype(where_bf16, ct.float32)

    ct.store(where_out_ptr, index=(n, c, 0, 0), tile=where_bf16)

    a4_f = ct.astype(a4, ct.float32)
    centered = a4_f - mean

    sum_val = ct.sum(where_f32)
    dot_val = ct.sum(where_f32 * centered)
    bf16_sum_val = ct.sum(ct.astype(where_bf16, ct.float32))

    ct.store(partial_sum_ptr, index=(n, c), tile=ct.reshape(sum_val, (1, 1)))
    ct.store(partial_dot_ptr, index=(n, c), tile=ct.reshape(dot_val, (1, 1)))
    ct.store(partial_bf16_sum_ptr, index=(n, c), tile=ct.reshape(bf16_sum_val, (1, 1)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,        # f32 [BATCH, C]
    partial_dot_ptr,        # f32 [BATCH, C]
    partial_bf16_sum_ptr,   # f32 [BATCH, C]
    invstd_ptr,             # f32 [C]
    weight_ptr,             # f32 [C]
    out_conv1_ptr,          # f32 [C]
    out_sum2_ptr,           # f32 [C]
    out_mul8_ptr,           # f32 [C]
    mean_term_ptr,          # f32 [C]
    coeff_ptr,              # f32 [C]
    out_scale_ptr,          # f32 [C]
    BATCH_C: ct.Constant[int],
    C: ct.Constant[int],
):
    # grid = (1,); one tile of shape (BATCH, C)
    tile_sum = ct.load(partial_sum_ptr, index=(0, 0), shape=(BATCH_C, C))
    tile_dot = ct.load(partial_dot_ptr, index=(0, 0), shape=(BATCH_C, C))
    tile_bf16 = ct.load(partial_bf16_sum_ptr, index=(0, 0), shape=(BATCH_C, C))

    sum_where = ct.sum(tile_sum, axis=0)
    sum_dot = ct.sum(tile_dot, axis=0)
    sum_bf16 = ct.sum(tile_bf16, axis=0)
    sum_bf16_r = ct.astype(ct.astype(sum_bf16, ct.bfloat16), ct.float32)

    invstd = ct.load(invstd_ptr, index=(0,), shape=(C,))
    weight = ct.load(weight_ptr, index=(0,), shape=(C,))

    ct.store(out_conv1_ptr, index=(0,), tile=sum_bf16_r)
    ct.store(out_sum2_ptr, index=(0,), tile=sum_where)
    ct.store(out_mul8_ptr, index=(0,), tile=sum_dot * invstd)
    ct.store(mean_term_ptr, index=(0,), tile=sum_where * SCALE)
    ct.store(coeff_ptr, index=(0,), tile=sum_dot * SCALE * invstd * invstd)
    ct.store(out_scale_ptr, index=(0,), tile=invstd * weight)


@ct.kernel
def _epilogue_kernel(
    where_out_ptr,          # bf16 [BATCH, C, H, W]
    arg4_ptr,               # bf16 [BATCH, C, H, W]
    mean_ptr,               # f32 [1, C, 1, 1]
    mean_term_ptr,          # f32 [C]
    coeff_ptr,              # f32 [C]
    out_scale_ptr,          # f32 [C]
    grad_out_ptr,           # bf16 [BATCH, C, H, W]
    C: ct.Constant[int],
    H: ct.Constant[int],
    W: ct.Constant[int],
):
    n = ct.bid(0)
    c = ct.bid(1)
    where_bf16 = ct.load(where_out_ptr, index=(n, c, 0, 0), shape=(1, 1, H, W))
    a4 = ct.load(arg4_ptr, index=(n, c, 0, 0), shape=(1, 1, H, W))
    mean = ct.load(mean_ptr, index=(0, c, 0, 0), shape=(1, 1, 1, 1))
    mean_term = ct.load(mean_term_ptr, index=(c,), shape=(1,))
    coeff = ct.load(coeff_ptr, index=(c,), shape=(1,))
    out_scale = ct.load(out_scale_ptr, index=(c,), shape=(1,))

    where_f32 = ct.astype(where_bf16, ct.float32)
    a4_f = ct.astype(a4, ct.float32)
    centered = a4_f - mean
    mt = ct.reshape(mean_term, (1, 1, 1, 1))
    co = ct.reshape(coeff, (1, 1, 1, 1))
    os = ct.reshape(out_scale, (1, 1, 1, 1))
    out = (where_f32 - centered * co - mt) * os
    ct.store(grad_out_ptr, index=(n, c, 0, 0), tile=ct.astype(out, ct.bfloat16))


@oracle_impl(hardware="B200", point="8addce5e", CHANNELS=32, HEIGHT=16, WIDTH=16)
@oracle_impl(hardware="B200", point="d10f1ba2", CHANNELS=64, HEIGHT=8, WIDTH=8)
def oracle_forward(inputs, *, CHANNELS: int, HEIGHT: int, WIDTH: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = inputs
    device = arg0_1.device
    hw = HEIGHT * WIDTH

    where_out = torch.empty_strided(
        (BATCH, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * hw, hw, WIDTH, 1),
        device=device, dtype=torch.bfloat16,
    )
    partial_sum = torch.empty((BATCH, CHANNELS), device=device, dtype=torch.float32)
    partial_dot = torch.empty((BATCH, CHANNELS), device=device, dtype=torch.float32)
    partial_bf16_sum = torch.empty((BATCH, CHANNELS), device=device, dtype=torch.float32)

    out_conv1 = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_sum2 = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_mul8 = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    mean_term = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    coeff = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_scale = torch.empty((CHANNELS,), device=device, dtype=torch.float32)

    grad_out = torch.empty_strided(
        (BATCH, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * hw, hw, WIDTH, 1),
        device=device, dtype=torch.bfloat16,
    )

    # arg3_1 is bf16[] scalar. cuTile can't load 0-D, view as [1].
    scalar_1 = arg3_1.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (BATCH, CHANNELS, 1),
        _partial_kernel,
        (arg0_1, arg1_1, arg2_1, scalar_1, arg4_1, arg5_1,
         where_out, partial_sum, partial_dot, partial_bf16_sum,
         CHANNELS, HEIGHT, WIDTH),
    )
    ct.launch(
        stream, (1, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_dot, partial_bf16_sum,
         arg6_1, arg7_1,
         out_conv1, out_sum2, out_mul8, mean_term, coeff, out_scale,
         BATCH, CHANNELS),
    )
    ct.launch(
        stream, (BATCH, CHANNELS, 1),
        _epilogue_kernel,
        (where_out, arg4_1, arg5_1, mean_term, coeff, out_scale, grad_out,
         CHANNELS, HEIGHT, WIDTH),
    )

    return where_out, out_conv1, out_sum2, out_mul8, grad_out
