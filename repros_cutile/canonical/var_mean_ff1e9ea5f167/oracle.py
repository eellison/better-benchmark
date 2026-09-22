"""cuTile port of var_mean_ff1e9ea5f167: bf16 GroupNorm + affine + ReLU.

Per-group reduction: 32 groups, group_elems = channels_per_group * H * W.
Compute mean, rsqrt(var+eps), then apply affine and ReLU, return bf16 output.
Return the fp32 mean and rsqrt tensors as side outputs.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _groupnorm_affine_relu_kernel(
    x_ptr,          # bf16 [N, G, GROUP_ELEMS]
    weight_ptr,     # f32 [G, cpg]
    bias_ptr,       # f32 [G, cpg]
    mean_ptr,       # f32 [N, G]
    rsqrt_ptr,      # f32 [N, G]
    act_ptr,        # bf16 [N, G, GROUP_ELEMS]
    cpg: ct.Constant[int],
    hw: ct.Constant[int],
    GROUP_ELEMS: ct.Constant[int],
):
    batch = ct.bid(0)
    group = ct.bid(1)

    x_tile = ct.load(x_ptr, index=(batch, group, 0), shape=(1, 1, GROUP_ELEMS))
    x_f = ct.astype(x_tile, ct.float32)

    x_sum = ct.sum(x_f)
    mean = x_sum * (1.0 / GROUP_ELEMS)
    centered = x_f - mean
    var_sum = ct.sum(centered * centered)
    var = var_sum * (1.0 / GROUP_ELEMS)
    inv_std = ct.rsqrt(var + 1.0e-5)

    # Load weight/bias slice for this group: [cpg]
    w = ct.load(weight_ptr, index=(group, 0), shape=(1, cpg))
    b = ct.load(bias_ptr, index=(group, 0), shape=(1, cpg))

    # Reshape centered from (1, 1, GROUP_ELEMS) to (cpg, hw)
    centered_2d = ct.reshape(centered, (cpg, hw))
    # Broadcast w from (1, cpg) to (cpg, hw): reshape w to (cpg, 1)
    w_2d = ct.reshape(w, (cpg, 1))
    b_2d = ct.reshape(b, (cpg, 1))
    normalized = centered_2d * inv_std
    affine = normalized * w_2d + b_2d
    zero = ct.zeros((cpg, hw), dtype=ct.float32)
    act = ct.where(affine > 0.0, affine, zero)
    act_bf16 = ct.astype(act, ct.bfloat16)
    act_3d = ct.reshape(act_bf16, (1, 1, GROUP_ELEMS))
    ct.store(act_ptr, index=(batch, group, 0), tile=act_3d)

    mean_tile = ct.reshape(mean, (1, 1))
    rsqrt_tile = ct.reshape(inv_std, (1, 1))
    ct.store(mean_ptr, index=(batch, group), tile=mean_tile)
    ct.store(rsqrt_ptr, index=(batch, group), tile=rsqrt_tile)


@oracle_impl(hardware="B200", point="2744156c")
@oracle_impl(hardware="B200", point="09b1da2c")
@oracle_impl(hardware="B200", point="6756477c")
@oracle_impl(hardware="B200", point="5ebcc8cf")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1 = inputs
    n = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    width = int(arg0_1.shape[3])
    groups = int(_shape_param_0[1])
    cpg = int(_shape_param_0[2])
    hw = height * width
    group_elems = cpg * hw

    mean_out = torch.empty_strided(
        (n, groups, 1, 1),
        (groups, 1, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    rsqrt_out = torch.empty_strided(
        (n, groups, 1, 1),
        (groups, 1, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    act_out = torch.empty_strided(
        (n, channels, height, width),
        (channels * height * width, height * width, width, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    x_view = arg0_1.view(n, groups, group_elems)
    act_view = act_out.view(n, groups, group_elems)
    w_view = arg1_1.view(groups, cpg)
    b_view = arg2_1.view(groups, cpg)
    mean_view = mean_out.view(n, groups)
    rsqrt_view = rsqrt_out.view(n, groups)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n, groups, 1),
        _groupnorm_affine_relu_kernel,
        (x_view, w_view, b_view, mean_view, rsqrt_view, act_view,
         cpg, hw, group_elems),
    )
    return mean_out, rsqrt_out, act_out
