"""cuTile port of pointwise_1354bb80cb3b: DenseNet cat + BN-inference + ReLU.

Approach: use torch.cat to build the [N, C_OUT, H, W] concat tensor (this
matches the reference exactly), then run a per-channel cuTile kernel that
applies the BN affine + ReLU + bf16 cast.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bn_relu_kernel(
    x_ptr,       # bf16 [N, C, HW]
    mean_ptr,    # bf16 [C]
    var_ptr,     # bf16 [C]
    weight_ptr,  # bf16 [C]
    bias_ptr,    # bf16 [C]
    out_ptr,     # bf16 [N, C, HW]
    BLOCK_HW: ct.Constant[int],
):
    n = ct.bid(0)
    c = ct.bid(1)

    x = ct.load(x_ptr, index=(n, c, 0), shape=(1, 1, BLOCK_HW),
                padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(x, ct.float32)

    mean = ct.load(mean_ptr, index=(c,), shape=(1,))
    var = ct.load(var_ptr, index=(c,), shape=(1,))
    weight = ct.load(weight_ptr, index=(c,), shape=(1,))
    bias = ct.load(bias_ptr, index=(c,), shape=(1,))
    mean_f = ct.reshape(ct.astype(mean, ct.float32), (1, 1, 1))
    var_f = ct.reshape(ct.astype(var, ct.float32), (1, 1, 1))
    weight_f = ct.reshape(ct.astype(weight, ct.float32), (1, 1, 1))
    bias_f = ct.reshape(ct.astype(bias, ct.float32), (1, 1, 1))

    invstd = ct.rsqrt(var_f + 1.0e-5)
    normalized = (x_f - mean_f) * invstd
    affine = normalized * weight_f + bias_f
    affine_bf16 = ct.astype(affine, ct.bfloat16)
    zero_bf = ct.zeros((1, 1, BLOCK_HW), dtype=ct.bfloat16)
    zero_f = ct.zeros((1, 1, BLOCK_HW), dtype=ct.float32)
    relu = ct.where(ct.astype(affine_bf16, ct.float32) < 0.0, zero_bf, affine_bf16)
    ct.store(out_ptr, index=(n, c, 0), tile=relu)


def _next_pow2(x):
    return 1 << (int(x) - 1).bit_length()


@oracle_impl(hardware="B200", point="c807f0f8", C0=512, C_OUT=576, H=7, W=7, BLOCK_C=8, BLOCK_HW=64)
@oracle_impl(hardware="B200", point="7dd3ded5", C0=256, C_OUT=320, H=14, W=14, BLOCK_C=8, BLOCK_HW=128)
@oracle_impl(hardware="B200", point="5f7d3f19", C0=128, C_OUT=192, H=28, W=28, BLOCK_C=8, BLOCK_HW=256)
@oracle_impl(hardware="B200", point="d3e900e3", C0=64, C_OUT=128, H=56, W=56, BLOCK_C=8, BLOCK_HW=256)
def oracle_forward(inputs, *, C0, C_OUT, H, W, BLOCK_C, BLOCK_HW):
    del C0, BLOCK_C  # unused kwargs
    x0, x1, x2, mean, var, weight, bias = inputs
    batch = int(x0.shape[0])
    hw = H * W

    # Build [N, C_OUT, H, W] via torch.cat, then flatten (H, W) into HW.
    cat = torch.cat([x0, x1, x2], dim=1).contiguous()
    cat_3d = cat.view(batch, C_OUT, hw)

    out = torch.empty_strided(
        (batch, C_OUT, H, W),
        (C_OUT * hw, hw, W, 1),
        device=x0.device,
        dtype=torch.bfloat16,
    )
    out_3d = out.view(batch, C_OUT, hw)

    # BLOCK_HW must be a power of 2 >= hw. If hw isn't power-of-2, use next
    # power of 2 and rely on ct.PaddingMode.ZERO plus the ct.store being on the
    # exact tile shape (which will overwrite whatever's there for OOB slots).
    # To avoid clobbering unrelated memory, allocate an over-sized bf16 buffer
    # if hw is not a power of 2.
    p2 = _next_pow2(hw)
    if p2 == hw:
        stream = torch.cuda.current_stream()
        ct.launch(
            stream,
            (batch, C_OUT, 1),
            _bn_relu_kernel,
            (cat_3d, mean, var, weight, bias, out_3d, p2),
        )
        return out

    # hw is not power of 2 -> pad output to (batch, C_OUT, p2) and copy the
    # valid slice.
    pad_out = torch.empty(
        (batch, C_OUT, p2),
        device=x0.device,
        dtype=torch.bfloat16,
    )
    pad_cat = torch.zeros(
        (batch, C_OUT, p2),
        device=x0.device,
        dtype=torch.bfloat16,
    )
    pad_cat[:, :, :hw] = cat_3d
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (batch, C_OUT, 1),
        _bn_relu_kernel,
        (pad_cat, mean, var, weight, bias, pad_out, p2),
    )
    out_3d.copy_(pad_out[:, :, :hw])
    return out
