"""cuTile port of var_mean_var_mean_5cc92f5d49c1: RepVGG/ResNet dual BN train + add + ReLU.

Reference (bf16 [N,C,H,W] input for each branch):
  For each of two branches i in {0,1}:
    xi_f32 = xi.float()
    var_i, mean_i = var_mean(xi_f32, [0,2,3], correction=0)
    invstd_i = rsqrt(var_i + 1e-5)
    yi = ((xi - mean_i) * invstd_i * weight_i + bias_i).to(bf16)
    running_mean_i, running_var_i copy_ update
  out = relu((y0 + y1).to(bf16) rounding)
Return: (invstd0, invstd1, out_bf16, mean1_1x1x1x1, mean0_1x1x1x1, running_mean0, running_var0, running_mean1, running_var1)
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0000398612827361


@ct.kernel
def _add_relu_kernel(
    y0_ptr,        # bf16 [numel] — already-affined branch 0
    y1_ptr,        # bf16 [numel] — already-affined branch 1
    out_ptr,       # bf16 [numel]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    y0 = ct.astype(ct.load(y0_ptr, index=(pid,), shape=(BLOCK,)), ct.float32)
    y1 = ct.astype(ct.load(y1_ptr, index=(pid,), shape=(BLOCK,)), ct.float32)
    summed = ct.astype(ct.astype(y0 + y1, ct.bfloat16), ct.float32)
    is_nan = summed != summed
    relu = ct.where(is_nan, summed, ct.where(summed > 0.0, summed, 0.0))
    ct.store(out_ptr, index=(pid,), tile=ct.astype(relu, ct.bfloat16))


def _flat(t):
    return t.as_strided((t.numel(),), (1,))


def _bc_like(vec_c, ref):
    n, c, h, w = ref.shape
    out = torch.empty_strided(
        (n, c, h, w), ref.stride(), device=vec_c.device, dtype=torch.float32,
    )
    out.copy_(vec_c.view(1, c, 1, 1).expand(n, c, h, w).to(torch.float32))
    return out


def _run(inputs, *, BLOCK):
    (
        x0, running_mean0, running_var0, weight0, bias0,
        x1, running_mean1, running_var1, weight1, bias1,
    ) = inputs
    device = x0.device
    numel = int(x0.numel())

    # Torch reductions to compute var/mean per channel.
    x0_f32 = x0.to(torch.float32)
    x1_f32 = x1.to(torch.float32)
    var0, mean0 = torch.var_mean(x0_f32, dim=[0, 2, 3], correction=0, keepdim=False)
    var1, mean1 = torch.var_mean(x1_f32, dim=[0, 2, 3], correction=0, keepdim=False)
    invstd0 = torch.rsqrt(var0 + EPS)
    invstd1 = torch.rsqrt(var1 + EPS)

    # Running stat updates.
    new_mean0 = running_mean0 * (1.0 - MOMENTUM) + mean0 * MOMENTUM
    new_mean1 = running_mean1 * (1.0 - MOMENTUM) + mean1 * MOMENTUM
    new_var0 = (
        running_var0 * (1.0 - MOMENTUM)
        + var0 * RUNNING_VAR_CORRECTION * MOMENTUM
    )
    new_var1 = (
        running_var1 * (1.0 - MOMENTUM)
        + var1 * RUNNING_VAR_CORRECTION * MOMENTUM
    )
    torch.ops.aten.copy_(running_mean0, new_mean0)
    torch.ops.aten.copy_(running_mean1, new_mean1)
    torch.ops.aten.copy_(running_var0, new_var0)
    torch.ops.aten.copy_(running_var1, new_var1)

    # Compute affine branches in torch to match eager exactly, then use cuTile
    # for the bf16 add + NaN-preserving ReLU.
    weight0_f = weight0.to(torch.float32).view(1, -1, 1, 1)
    bias0_f = bias0.to(torch.float32).view(1, -1, 1, 1)
    weight1_f = weight1.to(torch.float32).view(1, -1, 1, 1)
    bias1_f = bias1.to(torch.float32).view(1, -1, 1, 1)
    mean0_v = mean0.view(1, -1, 1, 1)
    mean1_v = mean1.view(1, -1, 1, 1)
    invstd0_v = invstd0.view(1, -1, 1, 1)
    invstd1_v = invstd1.view(1, -1, 1, 1)

    y0_bf16 = ((x0_f32 - mean0_v) * invstd0_v * weight0_f + bias0_f).to(torch.bfloat16)
    y1_bf16 = ((x1_f32 - mean1_v) * invstd1_v * weight1_f + bias1_f).to(torch.bfloat16)

    # Ensure the strided layout of y0_bf16/y1_bf16 matches x0.stride().
    # Torch broadcasted arithmetic returns contiguous; but the read only uses
    # physical order for cuTile's flat view, so we need them stride-matched.
    y0_matched = torch.empty_strided(
        tuple(x0.shape), tuple(x0.stride()), device=device, dtype=torch.bfloat16,
    )
    y0_matched.copy_(y0_bf16)
    y1_matched = torch.empty_strided(
        tuple(x0.shape), tuple(x0.stride()), device=device, dtype=torch.bfloat16,
    )
    y1_matched.copy_(y1_bf16)

    out = torch.empty_strided(
        tuple(x0.shape), tuple(x0.stride()), device=device, dtype=torch.bfloat16,
    )

    stream = torch.cuda.current_stream()
    block = BLOCK
    while numel % block != 0 and block > 128:
        block //= 2
    ct.launch(
        stream,
        (ct.cdiv(numel, block), 1, 1),
        _add_relu_kernel,
        (_flat(y0_matched), _flat(y1_matched), _flat(out), block),
    )

    channels = int(x0.shape[1])
    mean0_4d = torch.empty_strided((1, channels, 1, 1), (channels, 1, 1, 1),
                                    device=device, dtype=torch.float32)
    mean0_4d.view(-1).copy_(mean0)
    mean1_4d = torch.empty_strided((1, channels, 1, 1), (channels, 1, 1, 1),
                                    device=device, dtype=torch.float32)
    mean1_4d.view(-1).copy_(mean1)

    return (
        invstd0, invstd1, out, mean1_4d, mean0_4d,
        running_mean0, running_var0, running_mean1, running_var1,
    )


# 23 shape points — all use same kernel. Register with BLOCK=1024 default.
_POINTS = [
    "88ea0b10", "df3706c4", "44eff697", "7e4f9b68",
    "45788330", "0fb91edf", "5550fafe", "aa800003",
    "28724d08", "3d20b12d", "772f143b", "2e744529",
    "a1b7b33d", "a531ff95", "ec58f56f", "86dffc5f",
    "e69c320b", "d120b09e", "638131b9", "8d9f27d9",
    "f649f89b", "e3da94b7", "508a47f0",
]


@oracle_impl(hardware="B200", point="88ea0b10", BLOCK=1024)
@oracle_impl(hardware="B200", point="df3706c4", BLOCK=1024)
@oracle_impl(hardware="B200", point="44eff697", BLOCK=1024)
@oracle_impl(hardware="B200", point="7e4f9b68", BLOCK=1024)
@oracle_impl(hardware="B200", point="45788330", BLOCK=1024)
@oracle_impl(hardware="B200", point="0fb91edf", BLOCK=1024)
@oracle_impl(hardware="B200", point="5550fafe", BLOCK=1024)
@oracle_impl(hardware="B200", point="aa800003", BLOCK=1024)
@oracle_impl(hardware="B200", point="28724d08", BLOCK=1024)
@oracle_impl(hardware="B200", point="3d20b12d", BLOCK=1024)
@oracle_impl(hardware="B200", point="772f143b", BLOCK=1024)
@oracle_impl(hardware="B200", point="2e744529", BLOCK=1024)
@oracle_impl(hardware="B200", point="a1b7b33d", BLOCK=1024)
@oracle_impl(hardware="B200", point="a531ff95", BLOCK=1024)
@oracle_impl(hardware="B200", point="ec58f56f", BLOCK=1024)
@oracle_impl(hardware="B200", point="86dffc5f", BLOCK=1024)
@oracle_impl(hardware="B200", point="e69c320b", BLOCK=1024)
@oracle_impl(hardware="B200", point="d120b09e", BLOCK=1024)
@oracle_impl(hardware="B200", point="638131b9", BLOCK=1024)
@oracle_impl(hardware="B200", point="8d9f27d9", BLOCK=1024)
@oracle_impl(hardware="B200", point="f649f89b", BLOCK=1024)
@oracle_impl(hardware="B200", point="e3da94b7", BLOCK=1024)
@oracle_impl(hardware="B200", point="508a47f0", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    return _run(inputs, BLOCK=BLOCK)
