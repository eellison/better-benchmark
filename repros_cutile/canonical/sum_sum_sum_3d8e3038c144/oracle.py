"""cuTile port of sum_sum_sum_3d8e3038c144: functorch grouped BN backward.

Single kernel (mirroring Triton's 1 @triton.jit): each program processes one
group of BLOCK_K=16 features across BLOCK_B=128 rows.  In-kernel reductions
via ct.sum for the two axis=1 row stats (group_activation_sum,
group_where_sum), and the two axis=0 feature stats (vec, sum_value).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
GROUPS = 32
GROUP_WIDTH = 16
FEATURES = GROUPS * GROUP_WIDTH
GROUP_SCALE = 0.0625


@ct.kernel
def _grouped_bn_backward_kernel(
    grad_ptr,        # bf16 [BATCH, FEATURES]
    mask_ptr,        # b8   [BATCH, FEATURES]
    activation_ptr,  # bf16 [BATCH, FEATURES]
    weight_ptr,      # f32  [FEATURES]
    inv_ptr,         # f32  [BATCH, GROUPS]
    mean_ptr,        # f32  [BATCH, GROUPS]
    where_out_ptr,   # f32  [BATCH, FEATURES]
    vec_out_ptr,     # f32  [FEATURES]
    sum_out_ptr,     # f32  [FEATURES]
    dense_out_ptr,   # bf16 [BATCH, FEATURES]
    FEATURES_C: ct.Constant[int],
    GROUPS_C: ct.Constant[int],
    BLOCK_B: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    GROUP_SCALE_C: ct.Constant[float],
):
    group = ct.bid(0)

    grad_bf = ct.load(grad_ptr, index=(0, group), shape=(BLOCK_B, BLOCK_K))
    mask = ct.load(mask_ptr, index=(0, group), shape=(BLOCK_B, BLOCK_K))
    activation_bf = ct.load(activation_ptr, index=(0, group),
                            shape=(BLOCK_B, BLOCK_K))
    weight_1d = ct.load(weight_ptr, index=(group,), shape=(BLOCK_K,))
    inv_1d = ct.load(inv_ptr, index=(0, group), shape=(BLOCK_B, 1))
    mean_1d = ct.load(mean_ptr, index=(0, group), shape=(BLOCK_B, 1))

    grad = ct.astype(grad_bf, ct.float32)
    activation = ct.astype(activation_bf, ct.float32)
    weight_2d = ct.reshape(weight_1d, (1, BLOCK_K))

    zero_tile = ct.zeros((BLOCK_B, BLOCK_K), dtype=ct.float32)
    where_value = ct.where(mask, zero_tile, grad)
    activation_weighted = where_value * activation
    weighted_activation = activation_weighted * weight_2d
    weighted_where = where_value * weight_2d

    # axis=1 reductions per row inside the group.
    group_activation_sum = ct.sum(weighted_activation, axis=1, keepdims=True)
    group_where_sum = ct.sum(weighted_where, axis=1, keepdims=True)

    diff = group_where_sum * mean_1d - group_activation_sum
    coeff = diff * inv_1d * inv_1d * inv_1d * GROUP_SCALE_C
    neg_coeff = ct.zeros((BLOCK_B, 1), dtype=ct.float32) - coeff
    bias = neg_coeff * mean_1d - group_where_sum * inv_1d * GROUP_SCALE_C

    where_scale = inv_1d * weight_2d
    dense = where_value * where_scale + activation * coeff + bias

    vec_contrib = (activation_weighted - where_value * mean_1d) * inv_1d

    # axis=0 reductions across batch for this group's features.
    vec = ct.sum(vec_contrib, axis=0)
    sum_value = ct.sum(where_value, axis=0)

    ct.store(where_out_ptr, index=(0, group), tile=where_value)
    ct.store(vec_out_ptr, index=(group,), tile=vec)
    ct.store(sum_out_ptr, index=(group,), tile=sum_value)
    ct.store(dense_out_ptr, index=(0, group),
             tile=ct.astype(dense, ct.bfloat16))


@oracle_impl(hardware="B200", point="7e9ab656", BLOCK_B=128, BLOCK_K=16)
def oracle_forward(inputs, *, BLOCK_B: int, BLOCK_K: int):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1,
        _s0, _s1, _s2, _s3, _s4, _s5, _s6, _s7, _s8, _s9, _s10, _s11, _s12,
    ) = inputs
    device = arg0_1.device

    # View inputs as 2D (BATCH, FEATURES) for the cuTile kernel.
    grad_2d = arg0_1.view(BATCH, FEATURES)
    mask_2d = arg1_1.view(BATCH, FEATURES)
    activation_2d = arg2_1.view(BATCH, FEATURES)
    weight_1d = arg3_1.view(FEATURES)
    inv_2d = arg4_1.view(BATCH, GROUPS)
    mean_2d = arg5_1.view(BATCH, GROUPS)

    zero_out = torch.zeros((), device=device, dtype=torch.float32)
    where_out = torch.empty_strided(
        (BATCH, FEATURES, 1, 1),
        (FEATURES, 1, 1, 1),
        device=device, dtype=torch.float32,
    )
    where_out_2d = where_out.view(BATCH, FEATURES)
    vec_out = torch.empty((FEATURES,), device=device, dtype=torch.float32)
    sum_out = torch.empty((FEATURES,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (BATCH, FEATURES, 1, 1),
        (FEATURES, 1, 1, 1),
        device=device, dtype=torch.bfloat16,
    )
    dense_out_2d = dense_out.view(BATCH, FEATURES)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (GROUPS, 1, 1),
        _grouped_bn_backward_kernel,
        (grad_2d, mask_2d, activation_2d, weight_1d, inv_2d, mean_2d,
         where_out_2d, vec_out, sum_out, dense_out_2d,
         FEATURES, GROUPS, BLOCK_B, BLOCK_K, GROUP_SCALE),
    )

    return zero_out, where_out, vec_out, sum_out, dense_out
