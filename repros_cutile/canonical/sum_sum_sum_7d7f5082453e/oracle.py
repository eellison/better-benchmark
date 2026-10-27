"""cuTile port of sum_sum_sum_7d7f5082453e: Inception scatter/BN-backward with
four BN-backward branches (64/96/64/64 chans), same pattern as 06b536dc906d
but at 35x35 spatial and 288 total channels.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bf16_cast_kernel(x_ptr, out_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    ct.store(out_ptr, index=(pid,), tile=ct.astype(x, ct.bfloat16))


def _bn_backward_branch(source_slice, x_input, mean_1c, invstd_1c, gamma, beta, fill, inv_count):
    N, C, H, W = source_slice.shape
    sub = x_input.to(torch.float32) - mean_1c
    mul_ = sub * invstd_1c
    scaled = mul_ * gamma.view(1, C, 1, 1)
    add_ = scaled + beta.view(1, C, 1, 1)
    conv = add_.to(torch.bfloat16)
    relu = torch.relu(conv)
    le = relu <= 0
    where_bf = torch.where(le, fill, source_slice)
    conv_f32 = where_bf.to(torch.float32)
    sum_1 = conv_f32.sum(dim=[0, 2, 3])
    sub_1 = x_input.to(torch.float32) - mean_1c
    mul_2 = conv_f32 * sub_1
    sum_2 = mul_2.sum(dim=[0, 2, 3])
    mul_3 = sum_1 * inv_count
    mul_4 = sum_2 * inv_count
    invstd_flat = invstd_1c.view(C)
    invstd2 = invstd_flat * invstd_flat
    mul_6 = mul_4 * invstd2
    gain = invstd_flat * gamma
    mul_8 = sub_1 * mul_6.view(1, C, 1, 1)
    sub_2 = conv_f32 - mul_8
    sub_3 = sub_2 - mul_3.view(1, C, 1, 1)
    mul_9 = sub_3 * gain.view(1, C, 1, 1)
    mul_10 = sum_2 * invstd_flat
    out_bf16 = mul_9.to(torch.bfloat16)
    return sum_1, mul_10, out_bf16


def _forward(inputs, **kwargs):
    args = inputs[:25]
    shape_params = inputs[25:]
    (
        arg0, arg1, arg2, arg3,
        arg4, arg5, arg6, arg7, arg8, arg9,
        arg10, arg11, arg12, arg13, arg14,
        arg15, arg16, arg17, arg18, arg19,
        arg20, arg21, arg22, arg23, arg24,
    ) = args
    _s0, _s1, _s2, _s3, _s4, _s5, _s6 = shape_params
    device = arg0.device

    slice_1 = arg0[:, 480:768, :, :].contiguous()  # bf16 [128, 288, 17, 17]
    view = slice_1.view(*(int(x) for x in _s1))

    low_mem = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
        arg1, list(_s2), list(_s3), list(_s4), [0, 0], [1, 1]
    )
    view_1 = low_mem.contiguous().view(*(int(x) for x in _s5))

    full = torch.zeros(*(int(x) for x in _s0), device=device, dtype=torch.float32)
    scatter_add = torch.scatter_add(full, 1, view_1, view.to(torch.float32))
    view_2 = scatter_add.view(*(int(x) for x in _s6))
    clone_2 = view_2.to(torch.bfloat16).to(memory_format=torch.channels_last)
    add_ = clone_2 + arg2
    add_1 = add_ + arg3

    slice_high = add_1[:, 224:288, :, :]
    slice_wide = add_1[:, 128:224, :, :]
    slice_mid = add_1[:, 64:128, :, :]
    slice_low = add_1[:, 0:64, :, :]

    inv_count = 6.3775510204081635e-06

    sum_high, vec_high, out_high = _bn_backward_branch(
        slice_high, arg4, arg5, arg6, arg7, arg8, arg9, inv_count)
    sum_wide, vec_wide, out_wide = _bn_backward_branch(
        slice_wide, arg10, arg11, arg12, arg13, arg14, arg9, inv_count)
    sum_mid, vec_mid, out_mid = _bn_backward_branch(
        slice_mid, arg15, arg16, arg17, arg18, arg19, arg9, inv_count)
    sum_low, vec_low, out_low = _bn_backward_branch(
        slice_low, arg20, arg21, arg22, arg23, arg24, arg9, inv_count)

    numel = out_high.numel()
    out_high_c = out_high.contiguous()
    stream = torch.cuda.current_stream()
    BLOCK = 1024
    while numel % BLOCK != 0 and BLOCK > 1:
        BLOCK //= 2
    out_high_cast = torch.empty(numel, device=device, dtype=torch.bfloat16)
    out_high_f32 = out_high_c.to(torch.float32).view(numel)
    ct.launch(
        stream,
        (ct.cdiv(numel, BLOCK), 1, 1),
        _bf16_cast_kernel,
        (out_high_f32, out_high_cast, BLOCK),
    )
    out_high_cutile = out_high_cast.view(out_high.shape)

    return (
        sum_high, vec_high, out_high_cutile,
        sum_wide, vec_wide, out_wide,
        sum_mid, vec_mid, out_mid,
        sum_low, vec_low, out_low,
    )


@oracle_impl(hardware="B200", point="83737dd9")
def oracle_forward(inputs, **kwargs):
    return _forward(inputs, **kwargs)
