"""cuTile port of var_mean_mean_f766086bbce9 (NEW_PATTERN): GroupNorm(32) with
HW=1, affine, bf16 residual add, bf16 ReLU, and singleton-spatial mean/view.
Because HW=1, the mean over `[-1,-2]` is a no-op — the ReLU output is the
final result."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
CHANNELS = 512
GROUPS = 32
GROUP_CHANNELS = CHANNELS // GROUPS  # 16


@ct.kernel
def _groupnorm_residual_kernel(
    x_ptr,        # (BATCH, CHANNELS) bf16
    weight_ptr,   # (CHANNELS,) bf16
    bias_ptr,     # (CHANNELS,) bf16
    residual_ptr, # (BATCH, CHANNELS) bf16
    out_ptr,      # (BATCH, CHANNELS) bf16
    GROUP_ELEMS: ct.Constant[int],
):
    row = ct.bid(0)   # batch
    grp = ct.bid(1)   # group index
    # Load one group: (GROUP_ELEMS,) values from x[row, grp*GROUP_ELEMS:(grp+1)*GROUP_ELEMS]
    x = ct.load(x_ptr, index=(row, grp), shape=(1, GROUP_ELEMS))
    x_flat = ct.reshape(x, (GROUP_ELEMS,))
    x_f = ct.astype(x_flat, ct.float32)
    inv_e = 1.0 / GROUP_ELEMS
    mean = ct.sum(x_f) * inv_e
    centered = x_f - mean
    var = ct.sum(centered * centered) * inv_e
    invstd = ct.rsqrt(var + 1.0e-5)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(grp,), shape=(GROUP_ELEMS,))
    bias = ct.load(bias_ptr, index=(grp,), shape=(GROUP_ELEMS,))
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    affine = normalized * weight_f + bias_f
    affine_bf = ct.astype(affine, ct.bfloat16)

    residual = ct.load(residual_ptr, index=(row, grp), shape=(1, GROUP_ELEMS))
    residual_flat = ct.reshape(residual, (GROUP_ELEMS,))
    added = affine_bf + residual_flat
    zero_bf = ct.full(shape=(GROUP_ELEMS,), fill_value=0.0, dtype=ct.bfloat16)
    relu = ct.where(added > zero_bf, added, zero_bf)

    ct.store(out_ptr, index=(row, grp), tile=ct.reshape(relu, (1, GROUP_ELEMS)))


@oracle_impl(hardware="B200", point="cca94b79")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs

    x_2d = arg0_1.view(BATCH, CHANNELS)
    residual_2d = arg3_1.view(BATCH, CHANNELS)
    out = torch.empty_strided(
        (BATCH, CHANNELS),
        (CHANNELS, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH, GROUPS, 1),
        _groupnorm_residual_kernel,
        (x_2d, arg1_1, arg2_1, residual_2d, out, GROUP_CHANNELS),
    )
    return out.view(tuple(int(dim) for dim in _shape_param_2))
