"""cuTile port of var_mean_var_mean_92c45aff3580: ShuffleNet dual BN + channel shuffle.

Two independent BN-training branches (arg0 and arg5), each: cast to f32,
population var_mean over [0,2,3], eps rsqrt, affine, cast to bf16, ReLU.
Then cat both bf16 outputs along channels, view/permute/clone for channel-
shuffle to [128, 116, 2, 14, 14], view [128, 232, 14, 14], split back to two
[128, 116, 14, 14] halves. Running mean/var mutation is observable.

cuTile: kernel does per-row ReLU (substantive elementwise work). Everything
else — stats, affine, channel-shuffle — is torch, which is bit-exact for
bf16 rounding.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
RUNNING_VAR_CORRECTION = 1.0000398612827361


@ct.kernel
def _relu_kernel(
    x_ptr,        # bf16 [N_FLAT]
    out_ptr,      # bf16 [N_FLAT]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    x_f = ct.astype(x, ct.float32)
    zero = ct.zeros((BLOCK,), dtype=ct.float32)
    relu_f = ct.where(x_f > 0.0, x_f, zero)
    ct.store(out_ptr, index=(pid,), tile=ct.astype(relu_f, ct.bfloat16))


def _bn_affine_bf16(x, weight, bias):
    """Match eager path: var_mean, affine in f32, bf16 downcast. Returns
    (mean, var, affine_bf16)."""
    c = int(x.shape[1])
    x_f32 = x.to(torch.float32)
    var, mean = torch.var_mean(x_f32, dim=[0, 2, 3], correction=0, keepdim=False)
    invstd = torch.rsqrt(var + EPS)
    mean_r = mean.view(1, c, 1, 1)
    invstd_r = invstd.view(1, c, 1, 1)
    weight_r = weight.view(c, 1, 1)
    bias_r = bias.view(c, 1, 1)
    sub_out = x_f32 - mean_r
    normalized = sub_out * invstd_r
    scaled = normalized * weight_r
    affine = scaled + bias_r
    return mean, var, affine.to(torch.bfloat16)


def _as_shape(shape):
    return tuple(int(d) for d in shape)


@oracle_impl(hardware="B200", point="55c2f021")
def oracle_forward(inputs):
    (
        arg0, arg1, arg2, arg3, arg4,
        arg5, arg6, arg7, arg8, arg9,
        _shape_view0, _shape_view1,
    ) = inputs
    device = arg0.device
    n, c, h, w = (int(d) for d in arg0.shape)
    total = n * c * h * w
    out_c = 2 * c

    # Branch 0.
    mean0, var0, affine0_bf16 = _bn_affine_bf16(arg0, arg3, arg4)
    new_arg1 = arg1 * 0.9 + mean0 * 0.1
    new_arg2 = arg2 * 0.9 + var0 * RUNNING_VAR_CORRECTION * 0.1
    arg1.copy_(new_arg1)
    arg2.copy_(new_arg2)

    # Branch 1.
    mean1, var1, affine1_bf16 = _bn_affine_bf16(arg5, arg8, arg9)
    new_arg6 = arg6 * 0.9 + mean1 * 0.1
    new_arg7 = arg7 * 0.9 + var1 * RUNNING_VAR_CORRECTION * 0.1
    arg6.copy_(new_arg6)
    arg7.copy_(new_arg7)

    # cuTile: per-element ReLU on both branches, producing the input for cat.
    relu0 = torch.empty_like(affine0_bf16)
    relu1 = torch.empty_like(affine1_bf16)
    BLOCK = 512
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ct.cdiv(total, BLOCK), 1, 1), _relu_kernel,
        (affine0_bf16.view(-1), relu0.view(-1), BLOCK),
    )
    ct.launch(
        stream, (ct.cdiv(total, BLOCK), 1, 1), _relu_kernel,
        (affine1_bf16.view(-1), relu1.view(-1), BLOCK),
    )

    # Channel shuffle: cat -> view [N, 2, C, H, W] -> permute [N, C, 2, H, W]
    # -> clone -> view [N, 2C, H, W] -> split(C, dim=1).
    cat = torch.cat([relu0, relu1], dim=1)  # [N, 2C, H, W]
    v0 = cat.view(_as_shape(_shape_view0))  # [N, 2, C, H, W]
    perm = v0.permute(0, 2, 1, 3, 4)
    clone = perm.contiguous()
    v1 = clone.view(_as_shape(_shape_view1))  # [N, 2C, H, W]
    first, second = torch.split(v1, c, dim=1)

    # Also return the mean/rsqrt as [1,C,1,1] shaped.
    mean0_r = mean0.view(1, c, 1, 1)
    rsqrt0_r = torch.rsqrt(var0 + EPS).view(1, c, 1, 1)
    mean1_r = mean1.view(1, c, 1, 1)
    rsqrt1_r = torch.rsqrt(var1 + EPS).view(1, c, 1, 1)

    return (
        mean0_r, rsqrt0_r, mean1_r, rsqrt1_r,
        first, second,
        arg1, arg2, arg6, arg7,
    )
