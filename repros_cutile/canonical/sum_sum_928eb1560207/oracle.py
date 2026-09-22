"""cuTile port of sum_sum_928eb1560207: DenseNet BN-backward split-K + slice-add.

Mirrors Triton's 3-kernel plan:
1. `_partial_reduce_kernel` (grid = (C, num_blocks)): compute per-channel
   partial sum(where) and sum(where * centered) via `ct.sum(..., axis=0)`.
2. `_finalize_kernel` (grid = (C,)): reduce partials over blocks, store
   `sum_out`, `dot_tmp`, and `scale_grad = dot * invstd`.
3. `_epilogue_kernel` (grid = (C, num_blocks)): compute BN-backward dense
   output and residual slice add for c >= SLICE_START.

BLOCK_K=8192 matches Triton's setting; CHUNK_K=6272 is the effective work
per block; R_BLOCK=2048 was Triton's inner tile.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4
C = 224
H = 56
W = 56
HW = H * W          # 3136
K_TOTAL = N * HW    # 12544
SCALE = 7.971938775510203e-05
SLICE_START = 192
SLICE_C = 32
RESIDUAL_C = 256


def _ceil_pow2(value):
    return 1 << (int(value) - 1).bit_length()


NUM_BLOCKS = (K_TOTAL + 6272 - 1) // 6272   # 2
BLOCK_BLOCKS = _ceil_pow2(NUM_BLOCKS)        # 2


@ct.kernel
def _partial_reduce_kernel(
    mask_ptr,             # bf16 flat
    fill_ptr,             # bf16 scalar
    source_ptr,           # bf16 flat
    centered_source_ptr,  # f32 flat
    mean_ptr,             # f32 [C]
    partial_sum_ptr,      # f32 [num_blocks * C]
    partial_dot_ptr,      # f32 [num_blocks * C]
    K_TOTAL_: ct.Constant[int],
    HW_: ct.Constant[int],
    C_: ct.Constant[int],
    CHUNK_K_: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
):
    c = ct.bid(0)
    block = ct.bid(1)
    # cuTile: one tile that covers CHUNK_K with masking (Triton loops over
    # R_BLOCK-sized inner tiles; equivalent total work in one pass here).
    lanes = ct.arange(BLOCK_R, dtype=ct.int32)
    k_offsets = block * CHUNK_K_ + lanes
    active = (lanes < CHUNK_K_) & (k_offsets < K_TOTAL_)
    n = k_offsets // HW_
    spatial = k_offsets - n * HW_
    offsets = n * (C_ * HW_) + c * HW_ + spatial

    mask_bf = ct.gather(mask_ptr, offsets)
    mask_value = ct.astype(mask_bf, ct.float32)
    fill_scalar = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_value = ct.astype(fill_scalar, ct.float32)
    source_bf = ct.gather(source_ptr, offsets)
    source_value = ct.astype(source_bf, ct.float32)
    zero_f = ct.zeros((BLOCK_R,), dtype=ct.float32)
    grad = ct.where(mask_value <= zero_f,
                    ct.broadcast_to(fill_value, (BLOCK_R,)), source_value)
    grad = ct.where(active, grad, 0.0)

    centered_source = ct.gather(centered_source_ptr, offsets)
    mean_scalar = ct.gather(mean_ptr, ct.broadcast_to(c, (1,)))
    mean = ct.broadcast_to(mean_scalar, (BLOCK_R,))
    centered = ct.where(active, centered_source - mean, 0.0)

    product = grad * centered
    sum_grad = ct.sum(grad)
    sum_dot = ct.sum(product)

    ct.store(partial_sum_ptr, index=(block * C_ + c,),
             tile=ct.reshape(sum_grad, (1,)))
    ct.store(partial_dot_ptr, index=(block * C_ + c,),
             tile=ct.reshape(sum_dot, (1,)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,      # f32 [num_blocks * C]
    partial_dot_ptr,      # f32 [num_blocks * C]
    invstd_ptr,           # f32 [C]
    sum_out_ptr,          # f32 [C]
    dot_tmp_ptr,          # f32 [C]
    scale_grad_ptr,       # f32 [C]
    C_: ct.Constant[int],
    NUM_BLOCKS_: ct.Constant[int],
    BLOCK_BLOCKS_: ct.Constant[int],
):
    c = ct.bid(0)
    blocks = ct.arange(BLOCK_BLOCKS_, dtype=ct.int32)
    active = blocks < NUM_BLOCKS_
    offsets = blocks * C_ + c

    partial_sum = ct.gather(partial_sum_ptr, offsets)
    partial_dot = ct.gather(partial_dot_ptr, offsets)
    partial_sum = ct.where(active, partial_sum, 0.0)
    partial_dot = ct.where(active, partial_dot, 0.0)
    sum_value = ct.sum(partial_sum)
    dot_value = ct.sum(partial_dot)

    invstd_scalar = ct.gather(invstd_ptr, ct.broadcast_to(c, (1,)))
    invstd = ct.reshape(invstd_scalar, (1,))

    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(sum_value, (1,)))
    ct.store(dot_tmp_ptr, index=(c,), tile=ct.reshape(dot_value, (1,)))
    scale_grad_val = dot_value * ct.reshape(invstd, ())
    ct.store(scale_grad_ptr, index=(c,), tile=ct.reshape(scale_grad_val, (1,)))


@ct.kernel
def _epilogue_kernel(
    residual_ptr,          # bf16 [N, RESIDUAL_C, H, W] flat
    mask_ptr,              # bf16 flat
    fill_ptr,              # bf16 scalar
    source_ptr,            # bf16 flat
    centered_source_ptr,   # f32 flat
    mean_ptr,              # f32 [C]
    invstd_ptr,            # f32 [C]
    weight_ptr,            # f32 [C]
    sum_ptr,               # f32 [C]
    dot_ptr,               # f32 [C]
    dense_out_ptr,         # bf16 flat [N, C, H, W]
    add_out_ptr,           # bf16 flat [N, SLICE_C, H, W]
    K_TOTAL_: ct.Constant[int],
    HW_: ct.Constant[int],
    C_: ct.Constant[int],
    CHUNK_K_: ct.Constant[int],
    SCALE_: ct.Constant[float],
    SLICE_START_: ct.Constant[int],
    SLICE_C_: ct.Constant[int],
    RESIDUAL_C_: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    c = ct.bid(0)
    block = ct.bid(1)
    lanes = ct.arange(BLOCK_K, dtype=ct.int32)
    k_offsets = block * CHUNK_K_ + lanes
    active = (lanes < CHUNK_K_) & (k_offsets < K_TOTAL_)
    n = k_offsets // HW_
    spatial = k_offsets - n * HW_
    offsets = n * (C_ * HW_) + c * HW_ + spatial

    mask_bf = ct.gather(mask_ptr, offsets)
    mask_value = ct.astype(mask_bf, ct.float32)
    fill_scalar = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_value = ct.astype(fill_scalar, ct.float32)
    source_bf = ct.gather(source_ptr, offsets)
    source_value = ct.astype(source_bf, ct.float32)
    zero_f = ct.zeros((BLOCK_K,), dtype=ct.float32)
    grad = ct.where(mask_value <= zero_f,
                    ct.broadcast_to(fill_value, (BLOCK_K,)), source_value)

    centered_source = ct.gather(centered_source_ptr, offsets)
    mean_scalar = ct.gather(mean_ptr, ct.broadcast_to(c, (1,)))
    invstd_scalar = ct.gather(invstd_ptr, ct.broadcast_to(c, (1,)))
    weight_scalar = ct.gather(weight_ptr, ct.broadcast_to(c, (1,)))
    sum_scalar = ct.gather(sum_ptr, ct.broadcast_to(c, (1,)))
    dot_scalar = ct.gather(dot_ptr, ct.broadcast_to(c, (1,)))
    mean = ct.broadcast_to(mean_scalar, (BLOCK_K,))
    invstd_bc = ct.broadcast_to(invstd_scalar, (BLOCK_K,))
    weight_bc = ct.broadcast_to(weight_scalar, (BLOCK_K,))
    sum_bc = ct.broadcast_to(sum_scalar, (BLOCK_K,))
    dot_bc = ct.broadcast_to(dot_scalar, (BLOCK_K,))

    centered = centered_source - mean
    dot_scaled = dot_bc * SCALE_
    variance_term = dot_scaled * invstd_bc * invstd_bc
    corrected = grad - centered * variance_term
    mean_term = sum_bc * SCALE_
    centered_grad = corrected - mean_term
    output_scale = invstd_bc * weight_bc
    dense = centered_grad * output_scale
    dense_bf = ct.astype(dense, ct.bfloat16)

    ct.scatter(dense_out_ptr, offsets, dense_bf, mask=active)

    in_slice = c >= SLICE_START_
    if in_slice:
        slice_c = c - SLICE_START_
        add_offsets = n * (SLICE_C_ * HW_) + slice_c * HW_ + spatial
        residual_offsets = n * (RESIDUAL_C_ * HW_) + c * HW_ + spatial
        add_mask = active

        residual = ct.gather(residual_ptr, residual_offsets)
        residual_f = ct.astype(residual, ct.float32)
        dense_f = ct.astype(dense_bf, ct.float32)
        add_value = ct.astype(residual_f + dense_f, ct.bfloat16)
        ct.scatter(add_out_ptr, add_offsets, add_value, mask=add_mask)


@oracle_impl(hardware="B200", point="e6a7fabb", BLOCK_K=8192, CHUNK_K=6272, R_BLOCK=2048)
def oracle_forward(inputs, *, BLOCK_K: int, CHUNK_K: int, R_BLOCK: int):
    (residual, mask_input, fill, source, centered_source, mean, invstd, weight) = inputs
    device = source.device
    num_blocks = (K_TOTAL + CHUNK_K - 1) // CHUNK_K

    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    dot_tmp = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    scale_grad = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, H, W), (C * HW, HW, W, 1),
        device=device, dtype=torch.bfloat16,
    )
    add_out = torch.empty_strided(
        (N, SLICE_C, H, W), (SLICE_C * HW, HW, W, 1),
        device=device, dtype=torch.bfloat16,
    )
    partial_sum = torch.empty((num_blocks, C), device=device, dtype=torch.float32)
    partial_dot = torch.empty((num_blocks, C), device=device, dtype=torch.float32)

    def _flat(t):
        size = 1
        for s in t.shape:
            size *= s
        return t.contiguous().view(size)

    stream = torch.cuda.current_stream()
    # cuTile lacks in-kernel loops; sized the reduction tile to BLOCK_K so
    # it fully covers CHUNK_K in a single pass, mirroring the epilogue.
    ct.launch(
        stream,
        (C, num_blocks, 1),
        _partial_reduce_kernel,
        (_flat(mask_input), fill.view(1), _flat(source), _flat(centered_source),
         mean.view(C).contiguous(),
         partial_sum.view(-1), partial_dot.view(-1),
         K_TOTAL, HW, C, CHUNK_K, BLOCK_K),
    )
    ct.launch(
        stream,
        (C, 1, 1),
        _finalize_kernel,
        (partial_sum.view(-1), partial_dot.view(-1),
         invstd.view(C).contiguous(),
         sum_out, dot_tmp, scale_grad,
         C, num_blocks, BLOCK_BLOCKS),
    )
    ct.launch(
        stream,
        (C, num_blocks, 1),
        _epilogue_kernel,
        (_flat(residual), _flat(mask_input), fill.view(1), _flat(source),
         _flat(centered_source),
         mean.view(C).contiguous(), invstd.view(C).contiguous(),
         weight.view(C).contiguous(),
         sum_out, dot_tmp,
         dense_out.view(-1), add_out.view(-1),
         K_TOTAL, HW, C, CHUNK_K, SCALE,
         SLICE_START, SLICE_C, RESIDUAL_C, BLOCK_K),
    )
    return sum_out, scale_grad, dense_out, add_out
