"""cuTile port of sum_sum_sum_23c53e2c6899: MobileNet grouped BN backward.

Triton does the where compute + row reductions (sum_1, sum_2) + BN-backward
grouped pointwise + dense output all in one big kernel, plus a zero-init
kernel. This port uses two cuTile kernels: one to compute the where +
per-(N, channel) sum reductions, and one that does the finalize pointwise.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 64
H = 8
W = 8
HW = H * W


@ct.kernel
def _where_and_row_reduce_kernel(
    x0_ptr,       # bf16 [N, C, HW]
    x1_ptr,       # f32  [N, C, HW]
    mask_ptr,     # bool [N, C, HW]
    scalar_ptr,   # f32 [1]
    activation_ptr,  # bf16 [N, C, HW]
    where_out_ptr,   # f32 [N, C, HW]
    sum_1_ptr,       # f32 [N, C]  = sum(where * activation, dim=HW)
    sum_2_ptr,       # f32 [N, C]  = sum(where, dim=HW)
    HW_: ct.Constant[int],
):
    n = ct.bid(0)
    c = ct.bid(1)
    x0 = ct.astype(ct.load(x0_ptr, index=(n, c, 0), shape=(1, 1, HW_)),
                   ct.float32)
    x1 = ct.load(x1_ptr, index=(n, c, 0), shape=(1, 1, HW_))
    m = ct.load(mask_ptr, index=(n, c, 0), shape=(1, 1, HW_))
    s = ct.load(scalar_ptr, index=(0,), shape=(1,))
    s_scalar = ct.reshape(s, (1, 1, 1))
    act = ct.astype(ct.load(activation_ptr, index=(n, c, 0),
                            shape=(1, 1, HW_)),
                    ct.float32)
    added = x1 + x0
    where_val = ct.where(m, s_scalar, added)
    ct.store(where_out_ptr, index=(n, c, 0), tile=where_val)
    s1 = ct.sum(where_val * act)
    s2 = ct.sum(where_val)
    ct.store(sum_1_ptr, index=(n, c), tile=ct.reshape(s1, (1, 1)))
    ct.store(sum_2_ptr, index=(n, c), tile=ct.reshape(s2, (1, 1)))


def _scalar_to_1d(x):
    """Coerce a 0-D tensor to a 1-D tensor of size 1 (for cuTile load)."""
    return x.reshape(1) if x.dim() == 0 else x


@ct.kernel
def _bn_backward_finalize_kernel(
    where_ptr,       # f32 [N, 32, 2, 64]  (view of where)
    weight_ptr,      # f32 [N, 32, 2, 64]  (view of arg4 f32)
    mul3_ptr,        # f32 [N, 32, 2]  (bcast to [N, 32, 2, 64])
    mul8_ptr,        # f32 [N, 32]     (bcast to [N, 32, 2, 64])
    sub1_ptr,        # f32 [N, 32]     (bcast to [N, 32, 2, 64])
    out_ptr,         # bf16 [N, 32, 2, 64]
):
    n = ct.bid(0)
    g = ct.bid(1)  # groups 0..31
    where_tile = ct.load(where_ptr, index=(n, g, 0, 0), shape=(1, 1, 2, 64))
    weight_tile = ct.load(weight_ptr, index=(n, g, 0, 0), shape=(1, 1, 2, 64))
    mul3 = ct.load(mul3_ptr, index=(n, g, 0), shape=(1, 1, 2))  # (1,1,2)
    mul8 = ct.load(mul8_ptr, index=(n, g), shape=(1, 1))         # (1,1)
    sub1 = ct.load(sub1_ptr, index=(n, g), shape=(1, 1))         # (1,1)
    # Reshape mul3 -> (1,1,2,1) to broadcast against (1,1,2,64)
    mul3_b = ct.reshape(mul3, (1, 1, 2, 1))
    mul8_b = ct.reshape(mul8, (1, 1, 1, 1))
    sub1_b = ct.reshape(sub1, (1, 1, 1, 1))
    mul_12 = where_tile * mul3_b
    mul_13 = weight_tile * mul8_b
    add_1 = mul_12 + mul_13
    add_2 = add_1 + sub1_b
    out_bf = ct.astype(add_2, ct.bfloat16)
    ct.store(out_ptr, index=(n, g, 0, 0), tile=out_bf)


@oracle_impl(hardware="B200", point="e7e2fc4e")
def oracle_forward(inputs, **_kwargs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
     *_shape_params) = inputs
    device = arg0_1.device
    stream = torch.cuda.current_stream()

    # Compute `where` (relu backward) + per-(N,C) reductions in cuTile.
    # Reshape inputs to [N, C, HW] for the kernel.
    arg0_r = arg0_1.contiguous().view(N, C, HW)
    arg1_r = arg1_1.contiguous().view(N, C, HW)
    arg2_r = arg2_1.contiguous().view(N, C, HW)
    arg4_r = arg4_1.contiguous().view(N, C, HW)
    where_flat = torch.empty((N, C, HW), device=device, dtype=torch.float32)
    sum_1 = torch.empty((N, C), device=device, dtype=torch.float32)
    sum_2 = torch.empty((N, C), device=device, dtype=torch.float32)
    ct.launch(
        stream, (N, C, 1), _where_and_row_reduce_kernel,
        (arg0_r, arg1_r, arg2_r, _scalar_to_1d(arg3_1), arg4_r,
         where_flat, sum_1, sum_2, HW),
    )
    where = where_flat.view(N, C, H, W)
    arg4_f = arg4_1.to(torch.float32)

    unsqueeze_arg5 = arg5_1.view(1, C)
    mul_1 = sum_1 * unsqueeze_arg5
    sum_3 = mul_1.view(N, 32, 2).sum(dim=2)  # [N, 32]
    mul_2 = sum_2 * unsqueeze_arg5
    sum_4 = mul_2.view(N, 32, 2).sum(dim=2)  # [N, 32]

    unsqueeze_1 = arg6_1.view(N, 32, 1)
    view_4 = arg5_1.view(1, 32, 2)
    mul_3 = unsqueeze_1 * view_4  # [N, 32, 2]

    mul_4 = sum_4 * arg7_1
    sub = mul_4 - sum_3
    mul_5 = sub * arg6_1
    mul_6 = mul_5 * arg6_1
    mul_7 = mul_6 * arg6_1
    mul_8 = mul_7 * 0.0078125
    neg = -mul_8
    mul_9 = neg * arg7_1
    mul_10 = sum_4 * arg6_1
    mul_11 = mul_10 * 0.0078125
    sub_1 = mul_9 - mul_11

    # view_5 = where reshape to [N, 32, 2, 64]
    view_5 = where.view(N, 32, 2, 64)
    view_6 = arg4_f.view(N, 32, 2, 64)

    # Reduction outputs
    view_8 = sum_1.view(N, 32, 2)
    view_9 = sum_2.view(N, 32, 2)
    unsqueeze_7 = arg7_1.view(N, 32, 1)
    mul_14 = view_9 * unsqueeze_7
    sub_2 = view_8 - mul_14
    mul_15 = sub_2 * unsqueeze_1
    sum_5 = mul_15.sum(dim=0)  # [32, 2]
    view_10 = sum_5.view(C)
    sum_6 = sum_2.sum(dim=0)  # [C]

    # Final cuTile kernel to combine and convert
    out_bf16 = torch.empty_strided(
        (N, C, H, W), (C * HW, 1, C * W, C),
        device=device, dtype=torch.bfloat16,
    )
    out_view = torch.empty((N, 32, 2, 64), device=device, dtype=torch.bfloat16)

    ct.launch(
        stream, (N, 32, 1), _bn_backward_finalize_kernel,
        (view_5.contiguous(), view_6.contiguous(),
         mul_3.contiguous(), mul_8.contiguous(), sub_1.contiguous(),
         out_view),
    )

    # out_view is contiguous [N, 32, 2, 64]. Reshape to [N, C=64, H=8, W=8]
    # noting that _shape_param_7 is 'S', [128, 64, 8, 8] with channels_last strides.
    add_2 = out_view.view(N, C, H, W)
    out_bf16.copy_(add_2)

    return where, view_10, sum_6, out_bf16
