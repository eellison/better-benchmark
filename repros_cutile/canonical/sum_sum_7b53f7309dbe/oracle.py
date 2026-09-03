"""cuTile port of sum_sum_7b53f7309dbe: phlippe DenseNet BN-backward variant.

Per (n, c) or per-c programs with different tile sizes depending on shape.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 7.62939453125e-06
SLICE = 16


@ct.kernel
def _bn_backward_kernel(
    mask_input_arr,     # bf16 [N, C, HW]
    fill_arr,           # bf16 scalar
    where_rhs_arr,      # bf16 [N, C, HW]
    centered_base_arr,  # bf16 [N, C, HW]
    mean_arr,           # f32 [C]
    invstd_arr,         # f32 [C]
    weight_arr,         # f32 [C]
    sum_out_arr,        # f32 [C]
    weight_grad_arr,    # f32 [C]
    dense_out_arr,      # bf16 [N, C, HW]
    N_: ct.Constant[int],
    HW_: ct.Constant[int],
):
    c = ct.bid(0)

    mask_input = ct.load(mask_input_arr, index=(0, c, 0), shape=(N_, 1, HW_))
    rhs = ct.load(where_rhs_arr, index=(0, c, 0), shape=(N_, 1, HW_))
    base = ct.load(centered_base_arr, index=(0, c, 0), shape=(N_, 1, HW_))
    fill_scalar = ct.load(fill_arr, index=(0,), shape=(1,))

    mask_input_f = ct.astype(mask_input, ct.float32)
    rhs_f = ct.astype(rhs, ct.float32)
    base_f = ct.astype(base, ct.float32)
    fill_f = ct.astype(fill_scalar, ct.float32)
    fill_bc = ct.reshape(fill_f, (1, 1, 1))

    where_value = ct.where(mask_input_f <= 0.0, fill_bc, rhs_f)

    mean = ct.load(mean_arr, index=(c,), shape=(1,))
    invstd = ct.load(invstd_arr, index=(c,), shape=(1,))
    weight = ct.load(weight_arr, index=(c,), shape=(1,))
    mean_bc = ct.reshape(mean, (1, 1, 1))
    invstd_bc = ct.reshape(invstd, (1, 1, 1))
    weight_bc = ct.reshape(weight, (1, 1, 1))

    centered = base_f - mean_bc

    sum_where = ct.sum(where_value)
    sum_centered = ct.sum(where_value * centered)

    mean_term = sum_where * SCALE
    dot_scaled = sum_centered * SCALE
    variance_term = dot_scaled * invstd_bc * invstd_bc
    output_scale = invstd_bc * weight_bc

    grad = (where_value - centered * variance_term - mean_term) * output_scale
    grad_bf16 = ct.astype(grad, ct.bfloat16)

    ct.store(dense_out_arr, index=(0, c, 0), tile=grad_bf16)
    ct.store(sum_out_arr, index=(c,), tile=ct.reshape(sum_where, (1,)))
    ct.store(weight_grad_arr, index=(c,),
             tile=ct.reshape(sum_centered * invstd_bc, (1,)))


@oracle_impl(hardware="B200", point="4634de77")
@oracle_impl(hardware="B200", point="efa66b2e")
@oracle_impl(hardware="B200", point="912a44db")
def oracle_forward(inputs):
    mask_input, fill, where_rhs, centered_base, mean, invstd, weight = inputs
    n = int(mask_input.shape[0])
    c = int(mask_input.shape[1])
    h = int(mask_input.shape[2])
    w = int(mask_input.shape[3])
    hw = h * w
    device = mask_input.device

    sum_out = torch.empty_strided((c,), (1,), device=device, dtype=torch.float32)
    weight_grad = torch.empty_strided((c,), (1,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (n, c, h, w),
        (c * hw, hw, w, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    mask_3d = mask_input.view(n, c, hw)
    rhs_3d = where_rhs.view(n, c, hw)
    base_3d = centered_base.view(n, c, hw)
    mean_1d = mean.view(c)
    invstd_1d = invstd.view(c)
    weight_1d = weight.view(c)
    dense_3d = dense_out.view(n, c, hw)
    fill_1d = fill.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (c, 1, 1),
        _bn_backward_kernel,
        (mask_3d, fill_1d, rhs_3d, base_3d,
         mean_1d, invstd_1d, weight_1d,
         sum_out, weight_grad, dense_3d,
         n, hw),
    )
    return sum_out, weight_grad, dense_out, dense_out[:, :SLICE, :, :]
