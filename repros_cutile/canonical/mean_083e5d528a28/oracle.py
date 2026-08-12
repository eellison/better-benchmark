"""cuTile port of mean_083e5d528a28: MobilenetV2 BN + ReLU6 + spatial mean.

Single cuTile kernel does BN + ReLU6 + spatial mean over HW=49 in-kernel via
masked reduction over a pow2-padded BLOCK_HW dim. Matches the Triton kernel
which fuses all these ops.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bn_relu6_spatial_mean_kernel(
    mean_ptr,     # bf16 [C]
    x_ptr,        # bf16 [N*C*H*W] flat storage (channels-last physical)
    var_ptr,      # bf16 [C]
    weight_ptr,   # bf16 [C]
    bias_ptr,     # bf16 [C]
    out_ptr,      # bf16 [ROWS=N*C]
    STRIDE_N: ct.Constant[int],
    STRIDE_C: ct.Constant[int],
    STRIDE_H: ct.Constant[int],
    STRIDE_W: ct.Constant[int],
    CHANNELS: ct.Constant[int],
    WIDTH: ct.Constant[int],
    HW_: ct.Constant[int],
    ROWS: ct.Constant[int],
    BLOCK_ROWS: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    row_block = ct.bid(0)
    rows = row_block * BLOCK_ROWS + ct.arange(BLOCK_ROWS, dtype=ct.int32)
    hw = ct.arange(BLOCK_HW, dtype=ct.int32)
    row_valid = rows < ROWS
    hw_valid = hw < HW_

    n = rows // CHANNELS
    c = rows - n * CHANNELS
    h = hw // WIDTH
    w = hw - h * WIDTH

    n_2d = ct.reshape(n, (BLOCK_ROWS, 1))
    c_2d = ct.reshape(c, (BLOCK_ROWS, 1))
    h_2d = ct.reshape(h, (1, BLOCK_HW))
    w_2d = ct.reshape(w, (1, BLOCK_HW))
    offsets = (n_2d * STRIDE_N + c_2d * STRIDE_C
               + h_2d * STRIDE_H + w_2d * STRIDE_W)
    row_valid_2d = ct.reshape(row_valid, (BLOCK_ROWS, 1))
    hw_valid_2d = ct.reshape(hw_valid, (1, BLOCK_HW))
    mask = row_valid_2d & hw_valid_2d

    x = ct.astype(
        ct.gather(x_ptr, (offsets,), mask=mask),
        ct.float32,
    )
    running_mean = ct.astype(
        ct.gather(mean_ptr, (c,), mask=row_valid),
        ct.float32,
    )
    running_var = ct.astype(
        ct.gather(var_ptr, (c,), mask=row_valid),
        ct.float32,
    )
    weight = ct.astype(
        ct.gather(weight_ptr, (c,), mask=row_valid),
        ct.float32,
    )
    bias = ct.astype(
        ct.gather(bias_ptr, (c,), mask=row_valid),
        ct.float32,
    )

    rm_2d = ct.reshape(running_mean, (BLOCK_ROWS, 1))
    rv_2d = ct.reshape(running_var, (BLOCK_ROWS, 1))
    w_2d_ = ct.reshape(weight, (BLOCK_ROWS, 1))
    b_2d = ct.reshape(bias, (BLOCK_ROWS, 1))

    centered = x - rm_2d
    invstd = 1.0 / ct.sqrt(rv_2d + 1.0e-5)
    affine = centered * invstd * w_2d_ + b_2d
    rounded = ct.astype(ct.astype(affine, ct.bfloat16), ct.float32)
    zero2d = ct.full((BLOCK_ROWS, BLOCK_HW), 0.0, dtype=ct.float32)
    six2d = ct.full((BLOCK_ROWS, BLOCK_HW), 6.0, dtype=ct.float32)
    relu = ct.where(rounded < zero2d, zero2d, rounded)
    relu6 = ct.where(relu > six2d, six2d, relu)
    relu6_bf = ct.astype(ct.astype(relu6, ct.bfloat16), ct.float32)
    masked_vals = ct.where(mask, relu6_bf, zero2d)
    spatial_sum = ct.sum(masked_vals, axis=1)
    spatial_mean = spatial_sum * (1.0 / HW_)
    ct.scatter(out_ptr, (rows,),
               ct.astype(spatial_mean, ct.bfloat16),
               mask=row_valid)


@oracle_impl(hardware="B200", point="af408b42", BLOCK_ROWS=32, BLOCK_HW=64)
def oracle_forward(inputs, *, BLOCK_ROWS: int, BLOCK_HW: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape0, _shape1, shape2 = inputs
    del _shape0, _shape1

    out_shape = tuple(int(dim) for dim in shape2)
    channels = int(arg1_1.shape[1])
    hw = int(arg1_1.shape[2] * arg1_1.shape[3])
    rows = int(arg1_1.shape[0] * channels)

    out = torch.empty_strided(
        out_shape, (channels, 1), device=arg1_1.device, dtype=torch.bfloat16,
    )

    # Flat view of arg1_1 storage for offset-based access
    x_storage = arg1_1.numel()
    x_flat = torch.as_strided(arg1_1, (x_storage,), (1,))
    out_flat = out.view(rows)

    grid = ((rows + BLOCK_ROWS - 1) // BLOCK_ROWS, 1, 1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, grid, _bn_relu6_spatial_mean_kernel,
        (arg0_1, x_flat, arg2_1, arg3_1, arg4_1, out_flat,
         int(arg1_1.stride(0)), int(arg1_1.stride(1)),
         int(arg1_1.stride(2)), int(arg1_1.stride(3)),
         channels, int(arg1_1.shape[3]), hw, rows,
         BLOCK_ROWS, BLOCK_HW),
    )
    return out
