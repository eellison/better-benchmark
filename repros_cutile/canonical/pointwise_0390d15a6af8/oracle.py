"""cuTile port of pointwise_0390d15a6af8: GhostNet dual BN + cat + residual.

Computes two BN affines (on conv0 and conv1) in cuTile, then does cat +
residual add in torch. C0 in {12, 20, 40, 56, 80} (non-power-of-2), so we
gather per-channel scalars via ct.gather.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _bn_affine_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    C_C: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    idxs = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    channel = idxs - (idxs // C_C) * C_C

    mean = ct.gather(mean_ptr, channel)
    var = ct.gather(var_ptr, channel)
    weight = ct.gather(weight_ptr, channel)
    bias = ct.gather(bias_ptr, channel)

    x_f = ct.astype(x, ct.float32)
    inv = 1.0 / ct.sqrt(ct.astype(var, ct.float32) + EPS)
    affine = (x_f - ct.astype(mean, ct.float32)) * inv * ct.astype(weight, ct.float32) + ct.astype(bias, ct.float32)
    ct.store(out_ptr, index=(pid,), tile=ct.astype(affine, ct.bfloat16))


@oracle_impl(hardware="B200", point="0bcbb75d", C0=80, H=7, BLOCK=1024)
@oracle_impl(hardware="B200", point="517be145", C0=56, H=14, BLOCK=1024)
@oracle_impl(hardware="B200", point="6eba313b", C0=40, H=14, BLOCK=1024)
@oracle_impl(hardware="B200", point="4db2a61e", C0=20, H=28, BLOCK=1024)
@oracle_impl(hardware="B200", point="aec27f0f", C0=12, H=56, BLOCK=1024)
def oracle_forward(inputs, *, C0: int, H: int, BLOCK: int):
    mean0, conv0, var0, weight0, bias0, residual, mean1, conv1, var1, weight1, bias1 = inputs
    n = int(conv1.shape[0])
    c1 = C0 * 2
    W_dim = int(conv1.shape[3])
    device = conv1.device

    conv0_nhwc = conv0.permute(0, 2, 3, 1).contiguous().view(-1)
    conv1_nhwc = conv1.permute(0, 2, 3, 1).contiguous().view(-1)
    residual_nhwc = residual.permute(0, 2, 3, 1).contiguous().view(-1)

    n0_flat = conv0_nhwc.numel()
    n1_flat = conv1_nhwc.numel()

    branch0_flat = torch.empty(n0_flat, device=device, dtype=torch.bfloat16)
    branch1_flat = torch.empty(n1_flat, device=device, dtype=torch.bfloat16)

    KBLOCK = 512
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n0_flat, KBLOCK), 1, 1),
        _bn_affine_kernel,
        (conv0_nhwc, mean0, var0, weight0, bias0, branch0_flat, C0, KBLOCK),
    )
    ct.launch(
        stream,
        (ct.cdiv(n1_flat, KBLOCK), 1, 1),
        _bn_affine_kernel,
        (conv1_nhwc, mean1, var1, weight1, bias1, branch1_flat, c1, KBLOCK),
    )

    branch0 = branch0_flat.view(n, H, W_dim, C0)
    branch1 = branch1_flat.view(n, H, W_dim, c1)
    residual_r = residual_nhwc.view(n, H, W_dim, C0)

    cat_bhwc = torch.empty(n, H, W_dim, c1, device=device, dtype=torch.bfloat16)
    cat_bhwc[..., :C0] = residual_r
    cat_bhwc[..., C0:] = branch0

    out_bhwc = (cat_bhwc.float() + branch1.float()).to(torch.bfloat16)
    out = out_bhwc.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
    return out
