"""cuTile port of pointwise_fbfd10a2d290: ReLU + maxpool_with_offsets + le mask.

Two kernels:
  1. Mask kernel: for each spatial position, write le = (relu <= 0). This is
     equivalent to (x <= 0 OR x is NaN).
  2. Pool kernel: for each output (n, c, oh, ow), gather kernel_h * kernel_w
     values from the input (relu), find max, and record offset (kh * kw + kw).

We support 2x2 and 3x3 kernels; kernel size is a runtime parameter but the
number of gathers depends on it, so we compile-specialize via two @ct.kernel
implementations.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


def _next_power_of_2(x):
    return 1 << (int(x) - 1).bit_length()


@ct.kernel
def _mask_kernel(
    input_ptr,    # bf16 [numel]
    mask_ptr,     # b8   [numel]
    NUMEL: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    tile = ct.bid(0)
    off = ct.arange(BLOCK, dtype=ct.int32) + tile * BLOCK
    active = off < NUMEL
    x = ct.gather(input_ptr, (off,), mask=active, padding_value=ct.bfloat16(1.0))
    zero = ct.astype(0.0, ct.bfloat16)
    # relu(x) <= 0 iff x <= 0 (NaN → False)
    le = x <= zero
    ct.scatter(mask_ptr, (off,), le, mask=active)


@ct.kernel
def _pool2_kernel(
    input_ptr,    # bf16 [N, C, H_in, W_in]
    values_ptr,   # bf16 [N, C, H_out, W_out]
    offsets_ptr,  # i8   [N, C, H_out, W_out]
    TOTAL: ct.Constant[int],
    H_IN: ct.Constant[int],
    W_IN: ct.Constant[int],
    H_OUT: ct.Constant[int],
    W_OUT: ct.Constant[int],
    C_: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    tile = ct.bid(0)
    linear = ct.arange(BLOCK, dtype=ct.int32) + tile * BLOCK
    active = linear < TOTAL

    ow = linear % W_OUT
    tmp = linear // W_OUT
    oh = tmp % H_OUT
    plane = tmp // H_OUT
    c = plane % C_
    n = plane // C_

    in_h = oh * 2
    in_w = ow * 2

    x00 = ct.gather(input_ptr, (n, c, in_h, in_w), mask=active,
                    padding_value=ct.bfloat16(0.0))
    x01 = ct.gather(input_ptr, (n, c, in_h, in_w + 1), mask=active,
                    padding_value=ct.bfloat16(0.0))
    x10 = ct.gather(input_ptr, (n, c, in_h + 1, in_w), mask=active,
                    padding_value=ct.bfloat16(0.0))
    x11 = ct.gather(input_ptr, (n, c, in_h + 1, in_w + 1), mask=active,
                    padding_value=ct.bfloat16(0.0))

    # ReLU on x (via NaN-preserving max)
    zero = ct.full((BLOCK,), 0.0, dtype=ct.bfloat16)
    y00 = ct.where(x00 != x00, x00, ct.maximum(x00, zero))
    y01 = ct.where(x01 != x01, x01, ct.maximum(x01, zero))
    y10 = ct.where(x10 != x10, x10, ct.maximum(x10, zero))
    y11 = ct.where(x11 != x11, x11, ct.maximum(x11, zero))

    best = y00
    best_off = ct.zeros((BLOCK,), dtype=ct.int32)
    take = (y01 > best) | (y01 != y01)
    best = ct.where(take, y01, best)
    best_off = ct.where(take, ct.full((BLOCK,), 1, dtype=ct.int32), best_off)
    take = (y10 > best) | (y10 != y10)
    best = ct.where(take, y10, best)
    best_off = ct.where(take, ct.full((BLOCK,), 2, dtype=ct.int32), best_off)
    take = (y11 > best) | (y11 != y11)
    best = ct.where(take, y11, best)
    best_off = ct.where(take, ct.full((BLOCK,), 3, dtype=ct.int32), best_off)

    ct.scatter(values_ptr, (n, c, oh, ow), best, mask=active)
    ct.scatter(offsets_ptr, (n, c, oh, ow), ct.astype(best_off, ct.int8), mask=active)


@ct.kernel
def _pool3_kernel(
    input_ptr,
    values_ptr,
    offsets_ptr,
    TOTAL: ct.Constant[int],
    H_IN: ct.Constant[int],
    W_IN: ct.Constant[int],
    H_OUT: ct.Constant[int],
    W_OUT: ct.Constant[int],
    C_: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    tile = ct.bid(0)
    linear = ct.arange(BLOCK, dtype=ct.int32) + tile * BLOCK
    active = linear < TOTAL

    ow = linear % W_OUT
    tmp = linear // W_OUT
    oh = tmp % H_OUT
    plane = tmp // H_OUT
    c = plane % C_
    n = plane // C_

    in_h = oh * 2
    in_w = ow * 2
    zero = ct.full((BLOCK,), 0.0, dtype=ct.bfloat16)

    best = ct.full((BLOCK,), -3.3895313892515355e38, dtype=ct.bfloat16)
    best_off = ct.zeros((BLOCK,), dtype=ct.int32)
    # Unroll 3x3
    for kh in range(3):
        for kw in range(3):
            raw = ct.gather(
                input_ptr, (n, c, in_h + kh, in_w + kw),
                mask=active, padding_value=ct.bfloat16(-3.3895313892515355e38),
            )
            cand = ct.where(raw != raw, raw, ct.maximum(raw, zero))
            take = (cand > best) | (cand != cand)
            best = ct.where(take, cand, best)
            best_off = ct.where(take, ct.full((BLOCK,), kh * 3 + kw, dtype=ct.int32), best_off)

    ct.scatter(values_ptr, (n, c, oh, ow), best, mask=active)
    ct.scatter(offsets_ptr, (n, c, oh, ow), ct.astype(best_off, ct.int8), mask=active)


@oracle_impl(hardware="B200", point="27e7b058", POOL_BLOCK=256, MASK_BLOCK=1024)
@oracle_impl(hardware="B200", point="9594e3d9", POOL_BLOCK=256, MASK_BLOCK=1024)
@oracle_impl(hardware="B200", point="2ff9ef59", POOL_BLOCK=256, MASK_BLOCK=1024)
@oracle_impl(hardware="B200", point="00f3245f", POOL_BLOCK=256, MASK_BLOCK=1024)
@oracle_impl(hardware="B200", point="8d6fda54", POOL_BLOCK=256, MASK_BLOCK=1024)
@oracle_impl(hardware="B200", point="371add6b", POOL_BLOCK=256, MASK_BLOCK=1024)
def oracle_forward(inputs, *, POOL_BLOCK: int, MASK_BLOCK: int):
    arg0_1, _shape_param_0, _shape_param_1 = inputs
    kernel_h = int(_shape_param_0[0])
    kernel_w = int(_shape_param_0[1])
    stride_h = int(_shape_param_1[0])
    stride_w = int(_shape_param_1[1])
    device = arg0_1.device

    batch, channels, h_in, w_in = arg0_1.shape
    h_out = (h_in - kernel_h) // stride_h + 1
    w_out = (w_in - kernel_w) // stride_w + 1

    values = torch.empty_strided(
        (batch, channels, h_out, w_out),
        (channels * h_out * w_out, h_out * w_out, w_out, 1),
        device=device, dtype=arg0_1.dtype,
    )
    offsets = torch.empty_strided(
        (batch, channels, h_out, w_out),
        (channels * h_out * w_out, h_out * w_out, w_out, 1),
        device=device, dtype=torch.int8,
    )
    le_mask = torch.empty_strided(
        tuple(arg0_1.shape),
        tuple(arg0_1.stride()),
        device=device, dtype=torch.bool,
    )

    stream = torch.cuda.current_stream()
    total_pool = int(values.numel())
    pool_block = min(_next_power_of_2(total_pool), max(POOL_BLOCK, 4))
    pool_tiles = (total_pool + pool_block - 1) // pool_block

    if kernel_h == 2 and kernel_w == 2:
        ct.launch(
            stream, (pool_tiles, 1, 1), _pool2_kernel,
            (arg0_1, values, offsets, total_pool, h_in, w_in, h_out, w_out,
             channels, pool_block),
        )
    else:
        ct.launch(
            stream, (pool_tiles, 1, 1), _pool3_kernel,
            (arg0_1, values, offsets, total_pool, h_in, w_in, h_out, w_out,
             channels, pool_block),
        )

    # Mask kernel — flat
    numel = int(arg0_1.numel())
    input_flat = arg0_1.reshape(numel)
    mask_flat = le_mask.reshape(numel)
    mask_block = min(_next_power_of_2(numel), max(MASK_BLOCK, 4))
    mask_tiles = (numel + mask_block - 1) // mask_block
    ct.launch(
        stream, (mask_tiles, 1, 1), _mask_kernel,
        (input_flat, mask_flat, numel, mask_block),
    )
    return values, offsets, le_mask
