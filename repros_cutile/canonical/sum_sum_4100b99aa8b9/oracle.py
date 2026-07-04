"""cuTile port of sum_sum_4100b99aa8b9: DenseNet BN backward tail (160 ch).

Mirrors Triton's 3-kernel structure with `ct.sum` inside the partial-reduce
and finalize kernels:
  - `_partial_reduce_kernel`: split K=(N*HW=12544) into blocks of BLOCK_K=2048,
    reduce to per-block partials for `where` and `where*centered` via
    `ct.sum(..., axis=0)`.
  - `_finalize_kernel`: sum partials over blocks per channel via `ct.sum`,
    computing final sum/scale_grad.
  - `_epilogue_kernel`: compute dense BN grad + slice-add tail.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4
C = 160
H = 56
W = 56
HW = H * W  # 3136
K_TOTAL = N * HW  # 12544
SLICE_START = 128
SLICE_C = 32
SCALE = 7.971938775510203e-05


def _ceil_pow2(value):
    return 1 << (int(value) - 1).bit_length()


@ct.kernel
def _partial_reduce_kernel(
    mask_ptr,           # bf16 [N,C,H,W] flat
    fill_ptr,           # bf16 scalar
    source_ptr,         # bf16 [N,C,H,W] flat
    centered_source_ptr, # f32 [N,C,H,W] flat
    mean_ptr,           # f32 [C]
    partial_sum_ptr,    # f32 [num_k_blocks, C]
    partial_dot_ptr,    # f32 [num_k_blocks, C]
    K_TOTAL_: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    c = ct.bid(0)
    block = ct.bid(1)
    k_rel = ct.arange(BLOCK_K, dtype=ct.int32)
    k_offsets = block * BLOCK_K + k_rel
    active = k_offsets < K_TOTAL_
    n = k_offsets // HW
    spatial = k_offsets - n * HW
    offsets = n * (C * HW) + c * HW + spatial

    mask_bf = ct.gather(mask_ptr, offsets)
    mask_value = ct.astype(mask_bf, ct.float32)
    fill_scalar = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_value = ct.astype(fill_scalar, ct.float32)
    source_bf = ct.gather(source_ptr, offsets)
    source_value = ct.astype(source_bf, ct.float32)
    selected = ct.where(mask_value <= 0.0,
                        ct.broadcast_to(fill_value, (BLOCK_K,)),
                        source_value)
    selected_active = ct.where(active, selected, 0.0)

    centered_source = ct.gather(centered_source_ptr, offsets)
    mean_scalar = ct.gather(mean_ptr, ct.broadcast_to(c, (1,)))
    mean_bc = ct.broadcast_to(mean_scalar, (BLOCK_K,))
    centered = ct.where(active, centered_source - mean_bc, 0.0)

    partial_sum = ct.sum(selected_active)  # scalar
    partial_dot = ct.sum(selected_active * centered)  # scalar

    ct.store(partial_sum_ptr, index=(block, c),
             tile=ct.reshape(partial_sum, (1, 1)))
    ct.store(partial_dot_ptr, index=(block, c),
             tile=ct.reshape(partial_dot, (1, 1)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,    # f32 [num_k_blocks, C]
    partial_dot_ptr,    # f32 [num_k_blocks, C]
    invstd_ptr,         # f32 [C]
    sum_out_ptr,        # f32 [C]
    dot_tmp_ptr,        # f32 [C]
    scale_grad_ptr,     # f32 [C]
    NUM_K_BLOCKS: ct.Constant[int],
    BLOCK_BLOCKS: ct.Constant[int],
):
    c = ct.bid(0)
    block_idx = ct.arange(BLOCK_BLOCKS, dtype=ct.int32)
    active = block_idx < NUM_K_BLOCKS

    partial_sum = ct.load(partial_sum_ptr, index=(0, c), shape=(BLOCK_BLOCKS, 1),
                          padding_mode=ct.PaddingMode.ZERO)
    partial_dot = ct.load(partial_dot_ptr, index=(0, c), shape=(BLOCK_BLOCKS, 1),
                          padding_mode=ct.PaddingMode.ZERO)
    partial_sum_1d = ct.reshape(partial_sum, (BLOCK_BLOCKS,))
    partial_dot_1d = ct.reshape(partial_dot, (BLOCK_BLOCKS,))
    ps_active = ct.where(active, partial_sum_1d, 0.0)
    pd_active = ct.where(active, partial_dot_1d, 0.0)

    sum_value = ct.sum(ps_active)  # scalar
    dot_value = ct.sum(pd_active)  # scalar

    invstd_scalar = ct.gather(invstd_ptr, ct.broadcast_to(c, (1,)))
    invstd = ct.reshape(invstd_scalar, ())

    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(sum_value, (1,)))
    ct.store(dot_tmp_ptr, index=(c,), tile=ct.reshape(dot_value, (1,)))
    ct.store(scale_grad_ptr, index=(c,), tile=ct.reshape(dot_value * invstd, (1,)))


@ct.kernel
def _epilogue_kernel(
    r0_ptr, r1_ptr, r2_ptr,
    mask_ptr,
    fill_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    sum_ptr,
    dot_ptr,
    dense_out_ptr,
    add_out_ptr,
    K_TOTAL_: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    c = ct.bid(0)
    block = ct.bid(1)
    k_rel = ct.arange(BLOCK_K, dtype=ct.int32)
    k_offsets = block * BLOCK_K + k_rel
    active = k_offsets < K_TOTAL_
    n = k_offsets // HW
    spatial = k_offsets - n * HW
    offsets = n * (C * HW) + c * HW + spatial

    mask_bf = ct.gather(mask_ptr, offsets)
    mask_value = ct.astype(mask_bf, ct.float32)
    fill_scalar = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_value = ct.astype(fill_scalar, ct.float32)
    source_bf = ct.gather(source_ptr, offsets)
    source_value = ct.astype(source_bf, ct.float32)
    selected = ct.where(mask_value <= 0.0,
                        ct.broadcast_to(fill_value, (BLOCK_K,)),
                        source_value)

    centered_source = ct.gather(centered_source_ptr, offsets)
    mean_scalar = ct.gather(mean_ptr, ct.broadcast_to(c, (1,)))
    invstd_scalar = ct.gather(invstd_ptr, ct.broadcast_to(c, (1,)))
    weight_scalar = ct.gather(weight_ptr, ct.broadcast_to(c, (1,)))
    sum_scalar = ct.gather(sum_ptr, ct.broadcast_to(c, (1,)))
    dot_scalar = ct.gather(dot_ptr, ct.broadcast_to(c, (1,)))

    mean = ct.reshape(mean_scalar, (1,))
    invstd = ct.reshape(invstd_scalar, (1,))
    weight = ct.reshape(weight_scalar, (1,))
    sum_val = ct.reshape(sum_scalar, (1,))
    dot_val = ct.reshape(dot_scalar, (1,))

    mean_bc = ct.broadcast_to(mean, (BLOCK_K,))
    centered = centered_source - mean_bc
    mean_term = sum_val * SCALE
    dot_scaled = dot_val * SCALE
    invstd_sq = invstd * invstd
    variance_term = dot_scaled * invstd_sq
    output_scale = invstd * weight

    var_bc = ct.broadcast_to(variance_term, (BLOCK_K,))
    mt_bc = ct.broadcast_to(mean_term, (BLOCK_K,))
    os_bc = ct.broadcast_to(output_scale, (BLOCK_K,))
    after_variance = selected - centered * var_bc
    after_mean = after_variance - mt_bc
    dense_bf16 = ct.astype(after_mean * os_bc, ct.bfloat16)
    ct.scatter(dense_out_ptr, offsets, dense_bf16, mask=active)

    in_slice = c >= SLICE_START
    if in_slice:
        slice_c = c - SLICE_START
        add_offsets = n * (SLICE_C * HW) + slice_c * HW + spatial

        v0 = ct.gather(r0_ptr, n * (256 * HW) + c * HW + spatial)
        v1 = ct.gather(r1_ptr, n * (224 * HW) + c * HW + spatial)
        v2 = ct.gather(r2_ptr, n * (192 * HW) + c * HW + spatial)

        def _bf16_add(a, b):
            return ct.astype(ct.astype(a, ct.float32) + ct.astype(b, ct.float32),
                             ct.bfloat16)

        residual = _bf16_add(v0, v1)
        residual = _bf16_add(residual, v2)
        add_value = _bf16_add(residual, dense_bf16)
        ct.scatter(add_out_ptr, add_offsets, add_value, mask=active)


@oracle_impl(hardware="B200", point="515bd88b", BLOCK_K=2048)
def oracle_forward(inputs, *, BLOCK_K: int):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1,
     arg7_1, arg8_1, arg9_1) = inputs
    device = arg3_1.device
    num_k_blocks = (K_TOTAL + BLOCK_K - 1) // BLOCK_K
    block_blocks = _ceil_pow2(num_k_blocks)

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
    partial_sum = torch.empty_strided(
        (num_k_blocks, C), (C, 1), device=device, dtype=torch.float32
    )
    partial_dot = torch.empty_like(partial_sum)

    def _flat(t):
        size = 1
        for s in t.shape:
            size *= s
        return t.contiguous().view(size)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (C, num_k_blocks, 1), _partial_reduce_kernel,
        (_flat(arg3_1), arg4_1.view(1), _flat(arg5_1), _flat(arg6_1),
         arg7_1.view(C).contiguous(),
         partial_sum, partial_dot,
         K_TOTAL, BLOCK_K),
    )
    ct.launch(
        stream, (C, 1, 1), _finalize_kernel,
        (partial_sum, partial_dot,
         arg8_1.view(C).contiguous(),
         sum_out, dot_tmp, scale_grad,
         num_k_blocks, block_blocks),
    )
    ct.launch(
        stream, (C, num_k_blocks, 1), _epilogue_kernel,
        (_flat(arg0_1), _flat(arg1_1), _flat(arg2_1),
         _flat(arg3_1), arg4_1.view(1), _flat(arg5_1), _flat(arg6_1),
         arg7_1.view(C).contiguous(),
         arg8_1.view(C).contiguous(),
         arg9_1.view(C).contiguous(),
         sum_out, dot_tmp,
         dense_out.view(-1), add_out.view(-1),
         K_TOTAL, BLOCK_K),
    )
    return sum_out, scale_grad, dense_out, add_out
