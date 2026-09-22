"""cuTile port of pointwise_364ae2743aa2 (NEW_PATTERN): hard-sigmoid
broadcast multiply.

Mirrors the Triton oracle: two kernels — a storage-linear FLAT kernel and a
channel/spatial-tiled kernel that reuses the (BLOCK_C,) gate values across
(BLOCK_HW,) spatial elements. Hardsigmoid arithmetic is computed inside the
kernel (matching Triton), not offloaded to torch.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _flat_hardsigmoid_broadcast_mul_kernel(
    gate_ptr,     # bf16 flat (N*C,) via gate view
    x_ptr,        # bf16 flat storage-linear (total,)
    out_ptr,      # bf16 flat storage-linear (total,)
    gate_s0: ct.Constant[int],
    gate_s1: ct.Constant[int],
    CHANNELS: ct.Constant[int],
    HW: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    c_off = offsets - (offsets // CHANNELS) * CHANNELS  # offsets % channels
    n_off = offsets // (CHANNELS * HW)

    x = ct.gather(x_ptr, offsets)
    x_f = ct.astype(x, ct.float32)

    gate_off = n_off * gate_s0 + c_off * gate_s1
    gate = ct.gather(gate_ptr, gate_off)
    gate_f = ct.astype(gate, ct.float32)
    shifted = gate_f + 3.0
    clamped = ct.minimum(ct.maximum(shifted, 0.0), 6.0)
    scale = clamped * 0.16666666666666666
    ct.scatter(out_ptr, offsets, ct.astype(x_f * scale, ct.bfloat16))


@ct.kernel
def _hardsigmoid_broadcast_mul_kernel(
    gate_ptr,     # bf16 flat
    x_ptr,        # bf16 flat
    out_ptr,      # bf16 flat
    gate_s0: ct.Constant[int],
    gate_s1: ct.Constant[int],
    x_s0: ct.Constant[int],
    x_s1: ct.Constant[int],
    x_s2: ct.Constant[int],
    x_s3: ct.Constant[int],
    out_s0: ct.Constant[int],
    out_s1: ct.Constant[int],
    out_s2: ct.Constant[int],
    out_s3: ct.Constant[int],
    CHANNELS: ct.Constant[int],
    HEIGHT: ct.Constant[int],
    WIDTH: ct.Constant[int],
    HW_TILES: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    c_block = ct.bid(0)
    nhw_tile = ct.bid(1)
    n_idx = nhw_tile // HW_TILES
    hw_base = (nhw_tile - n_idx * HW_TILES) * BLOCK_HW

    c_offsets = c_block * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    hw_offsets = hw_base + ct.arange(BLOCK_HW, dtype=ct.int32)
    h_offsets = hw_offsets // WIDTH
    w_offsets = hw_offsets - h_offsets * WIDTH

    HW = HEIGHT * WIDTH
    c_mask = c_offsets < CHANNELS
    hw_mask = hw_offsets < HW

    # Gate load: (BLOCK_C,) per program.
    gate_offsets = n_idx * gate_s0 + c_offsets * gate_s1
    gate = ct.gather(gate_ptr, gate_offsets, mask=c_mask, padding_value=0.0)
    gate_f = ct.astype(gate, ct.float32)
    shifted = gate_f + 3.0
    clamped = ct.minimum(ct.maximum(shifted, 0.0), 6.0)
    scale = clamped * 0.16666666666666666  # (BLOCK_C,)

    # 2D offsets shape (BLOCK_HW, BLOCK_C).
    h_2d = ct.reshape(h_offsets, (BLOCK_HW, 1))
    w_2d = ct.reshape(w_offsets, (BLOCK_HW, 1))
    c_2d = ct.reshape(c_offsets, (1, BLOCK_C))
    mask_2d = ct.reshape(hw_mask, (BLOCK_HW, 1)) & ct.reshape(c_mask, (1, BLOCK_C))

    x_offsets = n_idx * x_s0 + h_2d * x_s2 + w_2d * x_s3 + c_2d * x_s1
    out_offsets = n_idx * out_s0 + h_2d * out_s2 + w_2d * out_s3 + c_2d * out_s1

    x = ct.gather(x_ptr, x_offsets, mask=mask_2d, padding_value=0.0)
    x_f = ct.astype(x, ct.float32)
    scale_2d = ct.reshape(scale, (1, BLOCK_C))
    ct.scatter(out_ptr, out_offsets, ct.astype(x_f * scale_2d, ct.bfloat16),
               mask=mask_2d)


# c72422f8: (T([512,960,1,1], bf16), T([512,960,7,7], bf16, stride=(47040,1,6720,960)))
@oracle_impl(hardware="B200", point="c72422f8", BLOCK_C=128, BLOCK_HW=16, USE_FLAT=True)
# 3fb6168e: (T([512,672,1,1], bf16), T([512,672,7,7], bf16, stride=(32928,1,4704,672)))
@oracle_impl(hardware="B200", point="3fb6168e", BLOCK_C=128, BLOCK_HW=16, USE_FLAT=False)
# 33f8ec7a: (T([512,672,1,1], bf16), T([512,672,14,14], bf16, stride=(131712,1,9408,672)))
@oracle_impl(hardware="B200", point="33f8ec7a", BLOCK_C=128, BLOCK_HW=16, USE_FLAT=False)
# c6ad2c20: (T([512,480,1,1], bf16), T([512,480,14,14], bf16, stride=(94080,1,6720,480)))
@oracle_impl(hardware="B200", point="c6ad2c20", BLOCK_C=128, BLOCK_HW=16, USE_FLAT=False)
# 2e45fbca: (T([512,120,1,1], bf16), T([512,120,28,28], bf16, stride=(94080,1,3360,120)))
@oracle_impl(hardware="B200", point="2e45fbca", BLOCK_C=128, BLOCK_HW=16, USE_FLAT=False)
# c8968885: (T([512,72,1,1], bf16), T([512,72,28,28], bf16, stride=(56448,1,2016,72)))
@oracle_impl(hardware="B200", point="c8968885", BLOCK_C=128, BLOCK_HW=16, USE_FLAT=False)
def oracle_forward(inputs, *, BLOCK_C, BLOCK_HW, USE_FLAT):
    gate, x = inputs
    batch = int(x.shape[0])
    channels = int(x.shape[1])
    height = int(x.shape[2])
    width = int(x.shape[3])

    output = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )

    # Flat metadata-only views (no copy) for gather/scatter.
    gate_flat = gate.view(-1)          # contig (N*C,)
    x_flat = torch.as_strided(x, (x.numel(),), (1,))  # storage-linear
    out_flat = torch.as_strided(output, (output.numel(),), (1,))

    stream = torch.cuda.current_stream()
    if USE_FLAT:
        block = 1024
        assert x.numel() % block == 0
        ct.launch(
            stream,
            (x.numel() // block, 1, 1),
            _flat_hardsigmoid_broadcast_mul_kernel,
            (gate_flat, x_flat, out_flat,
             int(gate.stride(0)), int(gate.stride(1)),
             channels, height * width, block),
        )
        return output

    hw_tiles = ct.cdiv(height * width, BLOCK_HW)
    ct.launch(
        stream,
        (ct.cdiv(channels, BLOCK_C), batch * hw_tiles, 1),
        _hardsigmoid_broadcast_mul_kernel,
        (gate_flat, x_flat, out_flat,
         int(gate.stride(0)), int(gate.stride(1)),
         int(x.stride(0)), int(x.stride(1)), int(x.stride(2)), int(x.stride(3)),
         int(output.stride(0)), int(output.stride(1)), int(output.stride(2)), int(output.stride(3)),
         channels, height, width, hw_tiles, BLOCK_C, BLOCK_HW),
    )
    return output
