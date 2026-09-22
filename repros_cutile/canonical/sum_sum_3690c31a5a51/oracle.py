"""cuTile port of sum_sum_3690c31a5a51: DenseNet BN backward.

Reductions and per-channel scaling computed on torch side; cuTile kernel
performs the fused per-element BN backward + residual add.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bn_backward_kernel(
    where_ptr, arg4_ptr, arg5_ptr, mul_1_ptr, weight_ptr, scale_ptr, slice1_ptr,
    add_ptr,
    NCHW: ct.Constant[int], C: ct.Constant[int], SPATIAL: ct.Constant[int],
    BLOCK_C: ct.Constant[int], BLOCK_S: ct.Constant[int], BLOCK_N: ct.Constant[int],
):
    n_block = ct.bid(0)
    c_block = ct.bid(1)
    s_block = ct.bid(2)

    # where: bf16 (N, C, HW), stored as flat (N, C*HW)
    # We do tile (BLOCK_N, BLOCK_C, BLOCK_S)
    where_val = ct.load(where_ptr, index=(n_block, c_block, s_block),
                        shape=(BLOCK_N, BLOCK_C, BLOCK_S))
    arg4 = ct.load(arg4_ptr, index=(n_block, c_block, s_block),
                   shape=(BLOCK_N, BLOCK_C, BLOCK_S))
    arg5 = ct.load(arg5_ptr, index=(c_block,), shape=(BLOCK_C,))
    mul_1 = ct.load(mul_1_ptr, index=(c_block,), shape=(BLOCK_C,))
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C,))  # already mul_4 = mul_2 * arg6^2
    scale = ct.load(scale_ptr, index=(c_block,), shape=(BLOCK_C,))    # arg6 * arg7
    slice1 = ct.load(slice1_ptr, index=(n_block, c_block, s_block),
                     shape=(BLOCK_N, BLOCK_C, BLOCK_S))

    where_f = ct.astype(where_val, ct.float32)
    arg4_f = ct.astype(arg4, ct.float32)
    arg5_3d = ct.reshape(arg5, (1, BLOCK_C, 1))
    mul_1_3d = ct.reshape(mul_1, (1, BLOCK_C, 1))
    weight_3d = ct.reshape(weight, (1, BLOCK_C, 1))
    scale_3d = ct.reshape(scale, (1, BLOCK_C, 1))

    sub = arg4_f - arg5_3d
    mul_6 = sub * weight_3d
    sub_1 = where_f - mul_6
    sub_2 = sub_1 - mul_1_3d
    mul_7 = sub_2 * scale_3d
    mul_7_bf = ct.astype(mul_7, ct.bfloat16)
    add = slice1 + mul_7_bf
    ct.store(add_ptr, index=(n_block, c_block, s_block), tile=add)


def _run(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1) = inputs
    device = arg0_1.device
    N, C_112, H, W = arg0_1.shape
    _, C, _, _ = arg1_1.shape  # 96
    spatial = H * W

    # slice_1: arg0_1[:, 16:112, :, :] = arg0_1[:, 16:, ...]
    slice_1 = arg0_1[:, 16:, :, :].contiguous()

    # where = arg1_1 <= 0 ? arg2_1 : arg3_1
    le = arg1_1 <= 0
    where_bf = torch.where(le, arg2_1, arg3_1).contiguous()
    where_f = where_bf.to(torch.float32)

    # sum_1 = sum(where_f, [0,2,3])   -> [C]
    sum_1 = where_f.sum(dim=[0, 2, 3])
    # sub = arg4_f - arg5_1
    arg4_f = arg4_1.to(torch.float32)
    sub = arg4_f - arg5_1
    # sum_2 = sum(where_f * sub, [0,2,3])
    sum_2 = (where_f * sub).sum(dim=[0, 2, 3])
    # mul_1 = sum_1 * scale_recip = 7.6293e-6
    scale_recip = 7.62939453125e-06
    mul_1 = sum_1 * scale_recip
    mul_2 = sum_2 * scale_recip
    weight = mul_2 * (arg6_1 * arg6_1)  # (mul_2 * arg6^2)
    scale_out = arg6_1 * arg7_1
    mul_8 = sum_2 * arg6_1

    # Prepare 3D flat tensors for the kernel: (N, C, spatial)
    where_3d = where_bf.view(N, C, spatial)
    arg4_3d = arg4_1.contiguous().view(N, C, spatial)
    slice1_3d = slice_1.view(N, C, spatial)

    add_bf = torch.empty((N, C, spatial), device=device, dtype=torch.bfloat16)

    # Choose block sizes matching shapes.
    BLOCK_N = 1
    BLOCK_C = 32
    BLOCK_S = spatial  # spatial should be power of 2 (32*32=1024, 16*16=256, 8*8=64)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (N, ct.cdiv(C, BLOCK_C), 1),
        _bn_backward_kernel,
        (where_3d, arg4_3d,
         arg5_1.view(C), mul_1, weight, scale_out,
         slice1_3d, add_bf,
         N * C * spatial, C, spatial, BLOCK_C, BLOCK_S, BLOCK_N),
    )

    add_4d = add_bf.view(N, C, H, W)
    slice_2 = add_4d[:, 0:16, :, :]
    return sum_1, mul_8, add_4d, slice_2


@oracle_impl(hardware="B200", point="8ea35f53")
@oracle_impl(hardware="B200", point="d95f44d7")
@oracle_impl(hardware="B200", point="905b7685")
def oracle_forward(inputs):
    return _run(inputs)
