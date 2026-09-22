"""cuTile port of sum_sum_894f3180f3c4: MobileNet-style BN-backward tail.

Matches Triton's 3-kernel structure:
  1. partial_reduce_add: materializes bf16 add and per-tile column partials.
  2. finalize: reduces partials into per-channel sums.
  3. epilogue: writes bf16 gradient dense output using finalized per-channel
     scalars.

Reductions live in-kernel via `ct.sum`, mirroring `tl.sum` in Triton.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 9.964923469387754e-06


def _ceil_pow2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


@ct.kernel
def _partial_reduce_add_kernel(
    x0_ptr,           # bf16 [k_total, C]  (NHWC-flat)
    x1_ptr,           # bf16 [k_total, C]
    x2_ptr,           # bf16 [k_total, C]
    mean_ptr,         # f32  [C]
    add_out_ptr,      # bf16 [k_total, C]
    partial_sum_ptr,  # f32  [num_k_blocks, C]
    partial_dot_ptr,  # f32  [num_k_blocks, C]
    C_C: ct.Constant[int],
    K_TOTAL: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    k_block = ct.bid(0)
    c_block = ct.bid(1)

    x0 = ct.load(x0_ptr, index=(k_block, c_block), shape=(BLOCK_K, BLOCK_C),
                 padding_mode=ct.PaddingMode.ZERO)
    x1 = ct.load(x1_ptr, index=(k_block, c_block), shape=(BLOCK_K, BLOCK_C),
                 padding_mode=ct.PaddingMode.ZERO)
    x2 = ct.load(x2_ptr, index=(k_block, c_block), shape=(BLOCK_K, BLOCK_C),
                 padding_mode=ct.PaddingMode.ZERO)
    x0f = ct.astype(x0, ct.float32)
    x1f = ct.astype(x1, ct.float32)
    x2f = ct.astype(x2, ct.float32)
    add_bf = ct.astype(x0f + x1f, ct.bfloat16)
    add_f = ct.astype(add_bf, ct.float32)

    mean = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,),
                   padding_mode=ct.PaddingMode.ZERO)
    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    centered = x2f - mean_2d
    dot = add_f * centered

    row_idx = ct.arange(BLOCK_K, dtype=ct.int32) + k_block * BLOCK_K
    col_idx = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
    row_valid = ct.reshape(row_idx < K_TOTAL, (BLOCK_K, 1))
    col_valid = ct.reshape(col_idx < C_C, (1, BLOCK_C))
    valid = row_valid & col_valid
    zero_f = ct.zeros((BLOCK_K, BLOCK_C), dtype=ct.float32)
    add_masked = ct.where(valid, add_f, zero_f)
    dot_masked = ct.where(valid, dot, zero_f)

    ct.store(add_out_ptr, index=(k_block, c_block), tile=add_bf)
    p_sum = ct.sum(add_masked, axis=0, keepdims=True)
    p_dot = ct.sum(dot_masked, axis=0, keepdims=True)
    ct.store(partial_sum_ptr, index=(k_block, c_block), tile=p_sum)
    ct.store(partial_dot_ptr, index=(k_block, c_block), tile=p_dot)


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,      # f32 [num_k_blocks, C]
    partial_dot_ptr,      # f32 [num_k_blocks, C]
    weight_ptr,           # f32 [C]
    sum_out_ptr,          # f32 [C]
    dot_tmp_ptr,          # f32 [C]
    dot_scaled_out_ptr,   # f32 [C]
    C_C: ct.Constant[int],
    NUM_K_BLOCKS: ct.Constant[int],
    BLOCK_P: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    p_sum = ct.load(partial_sum_ptr, index=(0, c_block),
                    shape=(BLOCK_P, BLOCK_C),
                    padding_mode=ct.PaddingMode.ZERO)
    p_dot = ct.load(partial_dot_ptr, index=(0, c_block),
                    shape=(BLOCK_P, BLOCK_C),
                    padding_mode=ct.PaddingMode.ZERO)
    p_idx = ct.arange(BLOCK_P, dtype=ct.int32)
    p_valid = ct.reshape(p_idx < NUM_K_BLOCKS, (BLOCK_P, 1))
    zero_f = ct.zeros((BLOCK_P, BLOCK_C), dtype=ct.float32)
    p_sum_v = ct.where(p_valid, p_sum, zero_f)
    p_dot_v = ct.where(p_valid, p_dot, zero_f)
    total_sum = ct.sum(p_sum_v, axis=0)
    total_dot = ct.sum(p_dot_v, axis=0)

    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    ct.store(sum_out_ptr, index=(c_block,), tile=total_sum)
    ct.store(dot_tmp_ptr, index=(c_block,), tile=total_dot)
    ct.store(dot_scaled_out_ptr, index=(c_block,), tile=total_dot * weight)


@ct.kernel
def _epilogue_kernel(
    add_ptr,         # bf16 [k_total, C]
    x2_ptr,          # bf16 [k_total, C]
    mean_ptr,        # f32  [C]
    weight_ptr,      # f32  [C]
    grad_weight_ptr, # f32  [C]
    sum_ptr,         # f32  [C]
    dot_ptr,         # f32  [C]
    out_ptr,         # bf16 [k_total, C]
    C_C: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    add = ct.load(add_ptr, index=(pid,), shape=(BLOCK,))
    x2 = ct.load(x2_ptr, index=(pid,), shape=(BLOCK,))
    add_f = ct.astype(add, ct.float32)
    x2_f = ct.astype(x2, ct.float32)

    idxs = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    channel = idxs - (idxs // C_C) * C_C

    mean = ct.gather(mean_ptr, channel)
    weight = ct.gather(weight_ptr, channel)
    grad_weight = ct.gather(grad_weight_ptr, channel)
    sum_v = ct.gather(sum_ptr, channel)
    dot_v = ct.gather(dot_ptr, channel)

    centered = x2_f - mean
    dot_mean = dot_v * SCALE
    weight_sq = weight * weight
    correction_scale = dot_mean * weight_sq
    correction = centered * correction_scale
    after_corr = add_f - correction
    mean_term = sum_v * SCALE
    centered_grad = after_corr - mean_term
    output_scale = weight * grad_weight
    result = centered_grad * output_scale
    ct.store(out_ptr, index=(pid,), tile=ct.astype(result, ct.bfloat16))


_POINTS = [
    "65b876e3", "1b9feebb", "3726f4ca", "9793b43e", "399aa3e2",
    "39a9326e", "b6f518ab", "a45e6340", "6f1023fc", "b5264010",
    "315c2b3e", "0dc5b6bd", "864b3c6f", "cf15f756", "727b7028",
    "3edd6c00", "ee318906", "1592ce3d", "ebb56431", "f35ade00",
]


@oracle_impl(hardware="B200", point="65b876e3", BLOCK_K=1024, BLOCK_C=16)
@oracle_impl(hardware="B200", point="1b9feebb", BLOCK_K=1024, BLOCK_C=16)
@oracle_impl(hardware="B200", point="3726f4ca", BLOCK_K=1024, BLOCK_C=16)
@oracle_impl(hardware="B200", point="9793b43e", BLOCK_K=1024, BLOCK_C=16)
@oracle_impl(hardware="B200", point="399aa3e2", BLOCK_K=1024, BLOCK_C=16)
@oracle_impl(hardware="B200", point="39a9326e", BLOCK_K=1024, BLOCK_C=16)
@oracle_impl(hardware="B200", point="b6f518ab", BLOCK_K=1024, BLOCK_C=16)
@oracle_impl(hardware="B200", point="a45e6340", BLOCK_K=1024, BLOCK_C=16)
@oracle_impl(hardware="B200", point="6f1023fc", BLOCK_K=1024, BLOCK_C=16)
@oracle_impl(hardware="B200", point="b5264010", BLOCK_K=1024, BLOCK_C=16)
@oracle_impl(hardware="B200", point="315c2b3e", BLOCK_K=1024, BLOCK_C=16)
@oracle_impl(hardware="B200", point="0dc5b6bd", BLOCK_K=1024, BLOCK_C=16)
@oracle_impl(hardware="B200", point="864b3c6f", BLOCK_K=1024, BLOCK_C=16)
@oracle_impl(hardware="B200", point="cf15f756", BLOCK_K=1024, BLOCK_C=16)
@oracle_impl(hardware="B200", point="727b7028", BLOCK_K=1024, BLOCK_C=16)
@oracle_impl(hardware="B200", point="3edd6c00", BLOCK_K=1024, BLOCK_C=16)
@oracle_impl(hardware="B200", point="ee318906", BLOCK_K=1024, BLOCK_C=16)
@oracle_impl(hardware="B200", point="1592ce3d", BLOCK_K=1024, BLOCK_C=16)
@oracle_impl(hardware="B200", point="ebb56431", BLOCK_K=1024, BLOCK_C=16)
@oracle_impl(hardware="B200", point="f35ade00", BLOCK_K=1024, BLOCK_C=16)
def oracle_forward(inputs, *, BLOCK_K: int, BLOCK_C: int):
    arg0, arg1, arg2, arg3, arg4, arg5 = inputs
    n, c, h, w = (int(d) for d in arg0.shape)
    total = n * c * h * w
    k_total = n * h * w
    num_k_blocks = (k_total + BLOCK_K - 1) // BLOCK_K
    block_p = _ceil_pow2(num_k_blocks)

    device = arg0.device

    # Channels-last -> NHWC contiguous flat views.
    a0 = arg0.permute(0, 2, 3, 1).contiguous().view(-1)
    a1 = arg1.permute(0, 2, 3, 1).contiguous().view(-1)
    a2 = arg2.permute(0, 2, 3, 1).contiguous().view(-1)

    a0_2d = a0.view(k_total, c)
    a1_2d = a1.view(k_total, c)
    a2_2d = a2.view(k_total, c)

    mean = arg3.view(c)
    weight = arg4  # [C] f32
    grad_weight = arg5  # [C] f32

    stream = torch.cuda.current_stream()

    add_flat_2d = torch.empty((k_total, c), device=device, dtype=torch.bfloat16)
    partial_sum = torch.empty((num_k_blocks, c), device=device, dtype=torch.float32)
    partial_dot = torch.empty((num_k_blocks, c), device=device, dtype=torch.float32)

    num_c_blocks = (c + BLOCK_C - 1) // BLOCK_C
    ct.launch(
        stream,
        (num_k_blocks, num_c_blocks, 1),
        _partial_reduce_add_kernel,
        (a0_2d, a1_2d, a2_2d, mean, add_flat_2d, partial_sum, partial_dot,
         c, k_total, BLOCK_K, BLOCK_C),
    )

    sum_out = torch.empty((c,), device=device, dtype=torch.float32)
    dot_tmp = torch.empty((c,), device=device, dtype=torch.float32)
    dot_scaled_out = torch.empty((c,), device=device, dtype=torch.float32)
    ct.launch(
        stream,
        (num_c_blocks, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_dot, weight, sum_out, dot_tmp, dot_scaled_out,
         c, num_k_blocks, block_p, BLOCK_C),
    )

    add_flat = add_flat_2d.view(-1)
    EPILOGUE_BLOCK = 1024
    out_flat = torch.empty(total, device=device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        (ct.cdiv(total, EPILOGUE_BLOCK), 1, 1),
        _epilogue_kernel,
        (add_flat, a2, mean, weight, grad_weight, sum_out, dot_tmp, out_flat,
         c, EPILOGUE_BLOCK),
    )

    add_out = add_flat.view(n, h, w, c).permute(0, 3, 1, 2).contiguous(
        memory_format=torch.channels_last)
    dense_out = out_flat.view(n, h, w, c).permute(0, 3, 1, 2).contiguous(
        memory_format=torch.channels_last)

    return add_out, sum_out, dot_scaled_out, dense_out
