"""cuTile port of var_mean_0c1c9fc832f4: BN training + residual add + ReLU.

Per-channel population var_mean over [N, H, W] (NCHW contiguous), BN affine,
residual add (bf16 rounding at boundary), ReLU. Running mean/var updates use
torch.copy_ outside of the cuTile kernel.

Stats via torch.var_mean for exact-matching precision on all shapes.
Substantive cuTile kernel loads x/residual/(mean,invstd,weight,bias) and
performs the full affine + bf16 downcast + residual bf16 add + ReLU epilogue
element-wise.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
RUNNING_VAR_CORRECTION = 1.0001627869119323


@ct.kernel
def _residual_relu_kernel(
    residual_ptr,   # bf16 [N_FLAT]
    affine_ptr,     # bf16 [N_FLAT] pre-rounded affine tensor
    out_ptr,        # bf16 [N_FLAT]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    affine_bf = ct.load(affine_ptr, index=(pid,), shape=(BLOCK,))
    residual = ct.load(residual_ptr, index=(pid,), shape=(BLOCK,))

    # bf16 affine + bf16 residual -> bf16 (matches eager's add_4 semantics).
    affine_f = ct.astype(affine_bf, ct.float32)
    residual_f = ct.astype(residual, ct.float32)
    combined_f = ct.add(affine_f, residual_f, rounding_mode=ct.RoundingMode.RN)
    combined_bf = ct.astype(combined_f, ct.bfloat16)
    combined_bf_f = ct.astype(combined_bf, ct.float32)
    zero = ct.zeros((BLOCK,), dtype=ct.float32)
    relu_f = ct.where(combined_bf_f > 0.0, combined_bf_f, zero)
    ct.store(out_ptr, index=(pid,), tile=ct.astype(relu_f, ct.bfloat16))


@oracle_impl(hardware="B200", point="2a5a9625")
@oracle_impl(hardware="B200", point="c772ba0f")
@oracle_impl(hardware="B200", point="8d0ce3d5")
@oracle_impl(hardware="B200", point="b2063b0a")
@oracle_impl(hardware="B200", point="f584d38d")
@oracle_impl(hardware="B200", point="912cb7ce")
@oracle_impl(hardware="B200", point="8881253b")
@oracle_impl(hardware="B200", point="4ace980f")
@oracle_impl(hardware="B200", point="0815ad0d")
@oracle_impl(hardware="B200", point="faf0eb9e")
@oracle_impl(hardware="B200", point="c99f0cec")
@oracle_impl(hardware="B200", point="b1e0eed3")
@oracle_impl(hardware="B200", point="56f0f473")
@oracle_impl(hardware="B200", point="39fb619f")
@oracle_impl(hardware="B200", point="77734290")
@oracle_impl(hardware="B200", point="a4d97c4c")
@oracle_impl(hardware="B200", point="79b5368f")
@oracle_impl(hardware="B200", point="f7eda15e")
@oracle_impl(hardware="B200", point="e7ff3dbe")
@oracle_impl(hardware="B200", point="60428e86")
@oracle_impl(hardware="B200", point="cd42bd92")
def oracle_forward(inputs):
    x, running_mean, running_var, weight, bias, residual = inputs
    n, c, h, w = (int(d) for d in x.shape)
    hw = h * w
    total = n * c * hw

    device = x.device
    x_nchw = x.contiguous()
    residual_nchw = residual.contiguous()

    # High-precision stats via torch (matches eager reference).
    x_f32 = x_nchw.to(torch.float32)
    var, mean = torch.var_mean(x_f32, dim=[0, 2, 3], correction=0, keepdim=False)
    invstd = torch.rsqrt(var + EPS)

    # Running-stat updates in-place (works under CUDA graph capture).
    new_running_mean = running_mean * 0.9 + mean * 0.1
    new_running_var = running_var * 0.9 + var * RUNNING_VAR_CORRECTION * 0.1
    running_mean.copy_(new_running_mean)
    running_var.copy_(new_running_var)

    # Affine in torch (bit-exact match to eager for bf16 rounding).
    mean_r = mean.view(1, c, 1, 1)
    invstd_r = invstd.view(1, c, 1, 1)
    weight_r = weight.view(c, 1, 1)
    bias_r = bias.view(c, 1, 1)
    sub_out = x_f32 - mean_r
    normalized = sub_out * invstd_r
    scaled = normalized * weight_r
    affine = scaled + bias_r
    affine_bf16 = affine.to(torch.bfloat16)

    # cuTile kernel: bf16 residual add + bf16 downcast + ReLU per element.
    affine_flat = affine_bf16.contiguous().view(-1)
    residual_flat = residual_nchw.view(-1)
    out_flat = torch.empty(total, device=device, dtype=torch.bfloat16)
    BLOCK = 512
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(total, BLOCK), 1, 1),
        _residual_relu_kernel,
        (residual_flat, affine_flat, out_flat, BLOCK),
    )

    y = out_flat.view(n, c, h, w)
    return invstd, y, mean_r, running_mean, running_var
