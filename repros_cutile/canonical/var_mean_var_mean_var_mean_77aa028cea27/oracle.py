"""cuTile port of var_mean_var_mean_var_mean_77aa028cea27: inception BN training.

4x BN training branches with in-place running-stat updates via copy_.
Reductions in torch, elementwise BN normalization step in cuTile per branch.
Concatenates 4 outputs then does avg_pool2d in torch.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 0.001
MOMENTUM = 0.1


@ct.kernel
def _bn_normalize_relu_kernel(
    x_ptr,          # bf16 [PIXELS]
    mean_ptr,       # f32 [C]
    rsqrt_ptr,      # f32 [C]
    weight_ptr,     # f32 [C]
    bias_ptr,       # f32 [C]
    out_ptr,        # bf16 [PIXELS]
    C: ct.Constant[int],
    HW: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    idx = ct.arange(BLOCK, dtype=ct.int64) + pid * BLOCK
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    x_f = ct.astype(x, ct.float32)
    c = (idx // HW) % C
    mean = ct.gather(mean_ptr, c)
    rsqrt = ct.gather(rsqrt_ptr, c)
    weight = ct.gather(weight_ptr, c)
    bias = ct.gather(bias_ptr, c)
    y = (x_f - mean) * rsqrt * weight + bias
    y_bf = ct.astype(y, ct.bfloat16)
    zero_bf = ct.zeros((BLOCK,), dtype=ct.bfloat16)
    relu_bf = ct.where(y_bf > zero_bf, y_bf, zero_bf)
    ct.store(out_ptr, index=(pid,), tile=relu_bf)


BLOCK = 1024


def _bn_branch(x, running_mean, running_var, weight, bias, device):
    """BN forward for one branch. Returns (mean, rsqrt, relu_bf16, add_mean, add_var)."""
    batch, c, h, w = x.shape
    n = batch * h * w
    n_correction = n / (n - 1.0)  # bessel correction factor
    hw = h * w
    pixels = batch * c * h * w
    x_f = x.to(torch.float32)
    var_mean = torch.ops.aten.var_mean.correction(x_f, [0, 2, 3], correction=0, keepdim=True)
    var = var_mean[0]  # [1, C, 1, 1]
    mean = var_mean[1]
    rsqrt = torch.rsqrt(var + EPS)
    mean_c = mean.squeeze()
    var_c = var.squeeze()
    rsqrt_c = rsqrt.squeeze()

    # Elementwise BN normalization + relu via cuTile
    x_flat = x.contiguous().view(-1)
    out = torch.empty_strided(
        (batch, c, h, w), (c * h * w, h * w, w, 1),
        device=device, dtype=torch.bfloat16,
    )
    out_flat = out.view(-1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ct.cdiv(pixels, BLOCK), 1, 1), _bn_normalize_relu_kernel,
        (x_flat, mean_c.contiguous(), rsqrt_c.contiguous(),
         weight.contiguous(), bias.contiguous(), out_flat,
         c, hw, BLOCK),
    )

    # Update running mean/var
    add_mean = mean_c * MOMENTUM + running_mean * 0.9
    add_var = var_c * n_correction * MOMENTUM + running_var * 0.9

    return mean, rsqrt, out, add_mean, add_var


@oracle_impl(hardware="B200", point="c9aa364e")
@oracle_impl(hardware="B200", point="d0d42d59")
@oracle_impl(hardware="B200", point="28b80430")
def oracle_forward(inputs):
    (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9,
     a10, a11, a12, a13, a14, a15, a16, a17, a18, a19) = inputs
    device = a0.device

    mean0, rsqrt0, relu0, add_mean0, add_var0 = _bn_branch(a0, a1, a2, a3, a4, device)
    mean1, rsqrt1, relu1, add_mean1, add_var1 = _bn_branch(a5, a6, a7, a8, a9, device)
    mean2, rsqrt2, relu2, add_mean2, add_var2 = _bn_branch(a10, a11, a12, a13, a14, device)
    mean3, rsqrt3, relu3, add_mean3, add_var3 = _bn_branch(a15, a16, a17, a18, a19, device)

    cat = torch.cat([relu0, relu1, relu2, relu3], dim=1)
    avg_pool = torch.ops.aten.avg_pool2d.default(cat, [3, 3], [1, 1], [1, 1])

    # In-place copy_ for running stats
    copy0 = torch.ops.aten.copy_.default(a1, add_mean0)
    copy1 = torch.ops.aten.copy_.default(a2, add_var0)
    copy2 = torch.ops.aten.copy_.default(a6, add_mean1)
    copy3 = torch.ops.aten.copy_.default(a7, add_var1)
    copy4 = torch.ops.aten.copy_.default(a11, add_mean2)
    copy5 = torch.ops.aten.copy_.default(a12, add_var2)
    copy6 = torch.ops.aten.copy_.default(a16, add_mean3)
    copy7 = torch.ops.aten.copy_.default(a17, add_var3)

    return (mean0, rsqrt0, mean1, rsqrt1, mean2, rsqrt2, mean3, rsqrt3,
            cat, avg_pool,
            copy0, copy1, copy2, copy3, copy4, copy5, copy6, copy7)
