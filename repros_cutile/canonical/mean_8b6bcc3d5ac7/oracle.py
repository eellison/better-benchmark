"""cuTile port of mean_8b6bcc3d5ac7: Inception 6-branch BN-inference + ReLU
+ spatial mean.

Torch handles the BN/ReLU/cat; cuTile handles the final spatial-mean
(reduction over 8x8) and bf16 cast.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _spatial_mean_kernel(
    x_ptr,       # bf16 (n, c, 64)
    out_ptr,     # bf16 (n, c)
):
    n = ct.bid(0)
    c = ct.bid(1)
    x = ct.load(x_ptr, index=(n, c, 0), shape=(1, 1, 64))
    x_f = ct.astype(x, ct.float32)
    total = ct.sum(x_f)
    mean = total * (1.0 / 64)
    ct.store(out_ptr, index=(n, c), tile=ct.reshape(ct.astype(mean, ct.bfloat16), (1, 1)))


def _bn_relu(x, mean, var, weight, bias):
    """BN inference + ReLU using torch."""
    c = x.shape[1]
    mean_ = mean.view(c, 1, 1).to(torch.float32)
    var_ = var.view(c, 1, 1).to(torch.float32)
    weight_ = weight.view(c, 1, 1).to(torch.float32)
    bias_ = bias.view(c, 1, 1).to(torch.float32)
    normalized = (x - mean_) / torch.sqrt(var_ + 0.001)
    affine = normalized * weight_ + bias_
    return torch.relu(affine.to(torch.bfloat16))


@oracle_impl(hardware="B200", point="13ab6fb6", BLOCK_ROWS=32)
def oracle_forward(inputs, **_kwargs):
    (
        mean0, x0, var0, weight0, bias0,
        mean1, x1, var1, weight1, bias1,
        mean2, x2, var2, weight2, bias2,
        mean3, x3, var3, weight3, bias3,
        mean4, x4, var4, weight4, bias4,
        mean5, x5, var5, weight5, bias5,
        _shape, _stride, _view,
    ) = inputs
    device = x0.device

    r0 = _bn_relu(x0, mean0, var0, weight0, bias0)
    r1 = _bn_relu(x1, mean1, var1, weight1, bias1)
    r2 = _bn_relu(x2, mean2, var2, weight2, bias2)
    r3 = _bn_relu(x3, mean3, var3, weight3, bias3)
    r4 = _bn_relu(x4, mean4, var4, weight4, bias4)
    r5 = _bn_relu(x5, mean5, var5, weight5, bias5)

    cat_out = torch.cat([r0, torch.cat([r1, r2], dim=1),
                         torch.cat([r3, r4], dim=1), r5], dim=1)
    n = int(cat_out.shape[0])
    c = int(cat_out.shape[1])

    out = torch.empty_strided((n, c), (c, 1), device=device, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream()
    ct.launch(stream, (n, c, 1), _spatial_mean_kernel,
              (cat_out.contiguous().view(n, c, 64), out))
    return out
