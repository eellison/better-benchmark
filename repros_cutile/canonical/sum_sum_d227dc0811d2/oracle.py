"""cuTile port of sum_sum_d227dc0811d2: NFNet pool-backward SiLU fanout with
cooperative split-K spatial reduction.

Structure:
  * `_zero_kernel` — zero the (N, C) partial accumulator.
  * `_pool_silu_spatial_split_kernel` — per (n, c_block, hw_block) tile: reads
    pool_grad at (n, c, h//2, w//2) via gather (channels-last strided), reads
    residual, activation, reduce_rhs; produces silu_out and scaled_out tiles
    via scatter (masked to skip OOB HW); reduces to per-channel `spatial_sum`
    via atomic_add into partial[n, c].
  * `_gate_row_channel_finalizer_kernel` — reduce partial over N, apply gate
    sigmoid-derivative epilogue, store gate_out/row_out, then reduce row_out
    into a final channel vector `final_out`.

Custom PTX (`add.rn.f32`/`mul.rn.f32`/etc.) is RTNE by default in cuTile.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 0.9622504486493761
MUL4 = 0.2
MUL5 = 2.0


@ct.kernel
def _zero_kernel(
    out_ptr,
    total: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    idxs = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    valid = idxs < total
    zeros = ct.zeros((BLOCK,), dtype=ct.float32)
    ct.scatter(out_ptr, idxs, zeros, mask=valid)


@ct.kernel
def _pool_silu_spatial_split_kernel(
    pool_grad_ptr,   # bf16 [N, C, pH, pW] channels-last
    residual_ptr,    # bf16 [N, C, H, W]   channels-last
    activation_ptr,  # bf16 [N, C, H, W]   channels-last
    reduce_rhs_ptr,  # bf16 [N, C, H, W]   channels-last
    silu_out_ptr,    # bf16 [N, C, H, W]   channels-last (output)
    scaled_out_ptr,  # bf16 [N, C, H, W]   channels-last (output)
    partial_ptr,     # f32  [N, C]         atomic accumulator
    C_: ct.Constant[int],
    H_: ct.Constant[int],
    W_: ct.Constant[int],
    HW_: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    n = ct.bid(0)
    c_block = ct.bid(1)
    hw_block = ct.bid(2)

    hw_off = ct.arange(BLOCK_HW, dtype=ct.int32) + hw_block * BLOCK_HW  # (BLOCK_HW,)
    c_off = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C      # (BLOCK_C,)

    c_valid = c_off < C_
    hw_valid = hw_off < HW_

    h = hw_off // W_  # (BLOCK_HW,)
    w = hw_off - h * W_  # (BLOCK_HW,)
    ph = h // 2
    pw = w // 2

    # Broadcast to (BLOCK_HW, BLOCK_C)
    n_tile = ct.full((BLOCK_HW, BLOCK_C), n, dtype=ct.int32)
    c_tile = ct.reshape(c_off, (1, BLOCK_C)) + ct.zeros((BLOCK_HW, BLOCK_C), dtype=ct.int32)
    h_tile = ct.reshape(h, (BLOCK_HW, 1)) + ct.zeros((BLOCK_HW, BLOCK_C), dtype=ct.int32)
    w_tile = ct.reshape(w, (BLOCK_HW, 1)) + ct.zeros((BLOCK_HW, BLOCK_C), dtype=ct.int32)
    ph_tile = ct.reshape(ph, (BLOCK_HW, 1)) + ct.zeros((BLOCK_HW, BLOCK_C), dtype=ct.int32)
    pw_tile = ct.reshape(pw, (BLOCK_HW, 1)) + ct.zeros((BLOCK_HW, BLOCK_C), dtype=ct.int32)

    mask2 = ct.reshape(hw_valid, (BLOCK_HW, 1)) & ct.reshape(c_valid, (1, BLOCK_C))

    pool_bf = ct.gather(pool_grad_ptr, (n_tile, c_tile, ph_tile, pw_tile),
                        mask=mask2, padding_value=0)
    residual_bf = ct.gather(residual_ptr, (n_tile, c_tile, h_tile, w_tile),
                            mask=mask2, padding_value=0)
    activation_bf = ct.gather(activation_ptr, (n_tile, c_tile, h_tile, w_tile),
                              mask=mask2, padding_value=0)
    reduce_rhs_bf = ct.gather(reduce_rhs_ptr, (n_tile, c_tile, h_tile, w_tile),
                              mask=mask2, padding_value=0)

    pool = ct.astype(pool_bf, ct.float32) * 0.25
    pool_bf16 = ct.astype(pool, ct.bfloat16)
    residual = ct.astype(residual_bf, ct.float32)
    added = residual + pool
    scaled = added * SCALE
    added_bf16 = ct.astype(residual + ct.astype(pool_bf16, ct.float32), ct.bfloat16)
    scaled_bf16 = ct.astype(ct.astype(added_bf16, ct.float32) * SCALE, ct.bfloat16)

    activation = ct.astype(activation_bf, ct.float32)
    sigmoid = 1.0 / (1.0 + ct.exp(-activation))
    silu_factor = sigmoid * (activation * (1.0 - sigmoid) + 1.0)
    silu_store_bf16 = ct.astype(scaled * silu_factor, ct.bfloat16)
    silu_bf16 = ct.astype(ct.astype(scaled_bf16, ct.float32) * silu_factor,
                          ct.bfloat16)

    mul4_bf16 = ct.astype(ct.astype(silu_bf16, ct.float32) * MUL4, ct.bfloat16)
    scaled_bf16_out = ct.astype(ct.astype(mul4_bf16, ct.float32) * MUL5,
                                ct.bfloat16)
    reduce_rhs = ct.astype(reduce_rhs_bf, ct.float32)
    reduce_val_bf16 = ct.astype(ct.astype(scaled_bf16_out, ct.float32) * reduce_rhs,
                                ct.bfloat16)

    ct.scatter(silu_out_ptr, (n_tile, c_tile, h_tile, w_tile),
               silu_store_bf16, mask=mask2)
    ct.scatter(scaled_out_ptr, (n_tile, c_tile, h_tile, w_tile),
               scaled_bf16_out, mask=mask2)

    # Sum reduce_val across BLOCK_HW to a (BLOCK_C,) tile, atomic_add to partial[n, c]
    zeros2 = ct.zeros((BLOCK_HW, BLOCK_C), dtype=ct.float32)
    masked = ct.where(mask2, ct.astype(reduce_val_bf16, ct.float32), zeros2)
    spatial_sum = ct.sum(masked, axis=0)  # (BLOCK_C,)
    # atomic_add to partial[n, c_off]
    n_out = ct.full((BLOCK_C,), n, dtype=ct.int32)
    ct.atomic_add(partial_ptr, (n_out, c_off), spatial_sum,
                  memory_order=ct.MemoryOrder.RELAXED)


@ct.kernel
def _gate_row_channel_finalizer_kernel(
    partial_ptr,   # f32 [N, C]
    gate_ptr,      # bf16 [N, C, 1, 1] channels-last (stride handled via 4D array view)
    gate_out_ptr,  # bf16 [N, C, 1, 1] channels-last
    row_out_ptr,   # bf16 [N, C, 1, 1] channels-last
    final_out_ptr, # f32  [C]
    N_: ct.Constant[int],
    C_: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    c_off = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
    n_off = ct.arange(BLOCK_N, dtype=ct.int32)
    c_valid = c_off < C_
    n_valid = n_off < N_
    mask2 = ct.reshape(n_valid, (BLOCK_N, 1)) & ct.reshape(c_valid, (1, BLOCK_C))
    n_tile = ct.reshape(n_off, (BLOCK_N, 1)) + ct.zeros((BLOCK_N, BLOCK_C), dtype=ct.int32)
    c_tile = ct.reshape(c_off, (1, BLOCK_C)) + ct.zeros((BLOCK_N, BLOCK_C), dtype=ct.int32)
    zero_i = ct.zeros((BLOCK_N, BLOCK_C), dtype=ct.int32)

    spatial_sum = ct.astype(ct.gather(partial_ptr, (n_tile, c_tile),
                                       mask=mask2, padding_value=0), ct.float32)
    gate_bf = ct.gather(gate_ptr, (n_tile, c_tile, zero_i, zero_i),
                        mask=mask2, padding_value=0)
    gate = ct.astype(gate_bf, ct.float32)
    gate_sig = ct.astype(1.0 / (1.0 + ct.exp(-gate)), ct.bfloat16)
    gate_sig_f32 = ct.astype(gate_sig, ct.float32)
    gate_deriv = gate_sig_f32 * (1.0 - gate_sig_f32)

    sum_bf16_f32 = ct.astype(ct.astype(spatial_sum, ct.bfloat16), ct.float32)
    row_bf16 = ct.astype(sum_bf16_f32 * gate_deriv, ct.bfloat16)

    ct.scatter(gate_out_ptr, (n_tile, c_tile, zero_i, zero_i),
               gate_sig, mask=mask2)
    ct.scatter(row_out_ptr, (n_tile, c_tile, zero_i, zero_i),
               row_bf16, mask=mask2)

    zeros2 = ct.zeros((BLOCK_N, BLOCK_C), dtype=ct.float32)
    row_f32 = ct.where(mask2, ct.astype(row_bf16, ct.float32), zeros2)
    total = ct.sum(row_f32, axis=0)  # (BLOCK_C,)
    total_bf16_rt = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    ct.scatter(final_out_ptr, c_off, total_bf16_rt, mask=c_valid)


def _oracle_forward_impl(inputs, *, BLOCK_HW, BLOCK_C, FINAL_BLOCK_C):
    arg0_1, _arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    device = arg2_1.device

    batch = int(arg2_1.shape[0])
    channels = int(arg2_1.shape[1])
    height = int(arg2_1.shape[2])
    width = int(arg2_1.shape[3])
    hw_size = height * width

    x_strides = tuple(arg2_1.stride())

    silu_out = torch.empty_strided(
        tuple(arg2_1.shape), x_strides, device=device, dtype=torch.bfloat16,
    )
    scaled_out = torch.empty_strided(
        tuple(arg2_1.shape), x_strides, device=device, dtype=torch.bfloat16,
    )
    gate_out = torch.empty_strided(
        tuple(arg5_1.shape), tuple(arg5_1.stride()), device=device, dtype=torch.bfloat16,
    )
    row_out = torch.empty_strided(
        tuple(arg5_1.shape), tuple(arg5_1.stride()), device=device, dtype=torch.bfloat16,
    )
    final_out = torch.empty((channels,), device=device, dtype=torch.float32)
    partial = torch.empty((batch, channels), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()

    total_partial = batch * channels
    zero_block = 1024
    ct.launch(
        stream,
        ((total_partial + zero_block - 1) // zero_block, 1, 1),
        _zero_kernel,
        (partial.view(-1), total_partial, zero_block),
    )

    ct.launch(
        stream,
        (batch, (channels + BLOCK_C - 1) // BLOCK_C,
         (hw_size + BLOCK_HW - 1) // BLOCK_HW),
        _pool_silu_spatial_split_kernel,
        (arg0_1, arg2_1, arg3_1, arg4_1, silu_out, scaled_out, partial,
         channels, height, width, hw_size, BLOCK_HW, BLOCK_C),
    )

    # Round batch up to next power of two for finalizer's BLOCK_N.
    n_pow2 = 1
    while n_pow2 < batch:
        n_pow2 *= 2
    ct.launch(
        stream,
        ((channels + FINAL_BLOCK_C - 1) // FINAL_BLOCK_C, 1, 1),
        _gate_row_channel_finalizer_kernel,
        (partial, arg5_1, gate_out, row_out, final_out,
         batch, channels, n_pow2, FINAL_BLOCK_C),
    )

    return silu_out, scaled_out, gate_out, row_out, final_out


# 2570be15: NFNet-L0 train, bf16 [128,512,28,28] channels-last fanout.
@oracle_impl(hardware="B200", point="2570be15", BLOCK_HW=128, BLOCK_C=64, FINAL_BLOCK_C=16)
# 1f32ce1f: NFNet-L0 train, bf16 [128,1536,14,14] channels-last fanout.
@oracle_impl(hardware="B200", point="1f32ce1f", BLOCK_HW=128, BLOCK_C=128, FINAL_BLOCK_C=16)
def oracle_forward(inputs, *, BLOCK_HW, BLOCK_C, FINAL_BLOCK_C):
    return _oracle_forward_impl(
        inputs, BLOCK_HW=BLOCK_HW, BLOCK_C=BLOCK_C, FINAL_BLOCK_C=FINAL_BLOCK_C,
    )
