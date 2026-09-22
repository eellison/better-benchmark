"""cuTile port of var_mean_af505388b035: exact-erf GELU plus population LayerNorm (bf16).

cuTile lacks `erf`, so we precompute the bf16-rounded exact-erf GELU with torch
(matching the Triton oracle's bf16 rounding boundary) and hand the rounded input
to a cuTile row LayerNorm kernel. cuTile requires power-of-2 tile shapes; when
HIDDEN is not a power of two we use padding_mode=ZERO on the load and mask the
padded lanes out of the reductions and epilogue, storing back only the valid
HIDDEN columns via a per-row output kernel that takes a valid mask.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _layernorm_bf16_kernel_exact(
    x_ptr,       # bf16 [ROWS, HIDDEN]
    weight_ptr,  # bf16 [HIDDEN]
    bias_ptr,    # bf16 [HIDDEN]
    out_ptr,     # bf16 [ROWS, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H),
                padding_mode=ct.PaddingMode.ZERO)
    xf = ct.astype(x, ct.float32)

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    mask = ct.reshape(cols < HIDDEN, (1, BLOCK_H))
    xf_masked = ct.where(mask, xf, 0.0)

    sum_x = ct.sum(xf_masked)
    mean = sum_x * (1.0 / HIDDEN)
    centered = ct.where(mask, xf - mean, 0.0)
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + 1.0e-12)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_f = ct.reshape(ct.astype(weight, ct.float32), (1, BLOCK_H))
    bias_f = ct.reshape(ct.astype(bias, ct.float32), (1, BLOCK_H))
    y = centered * invstd * weight_f + bias_f
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(y, ct.bfloat16))


@oracle_impl(hardware="B200", point="5428f51a", BLOCK_H=1024)
@oracle_impl(hardware="B200", point="113efab4", BLOCK_H=2048)
@oracle_impl(hardware="B200", point="f4120204", BLOCK_H=1024)
@oracle_impl(hardware="B200", point="bc0fb1fb", BLOCK_H=128)
@oracle_impl(hardware="B200", point="6462934d", BLOCK_H=1024)
@oracle_impl(hardware="B200", point="7648de68", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    # Pre-compute exact-erf GELU with the bf16 rounding boundary that the
    # Triton oracle keeps between the activation and the norm.
    xf = arg0_1.to(torch.float32)
    gelu = ((xf * 0.5) * (torch.erf(xf * 0.7071067811865476) + 1.0)).to(torch.bfloat16)

    out = torch.empty_like(arg0_1)
    stream = torch.cuda.current_stream()
    if BLOCK_H != hidden:
        # HIDDEN is not a power-of-2; use padding_mode=ZERO on the load and
        # write only the first HIDDEN columns via a store of the whole BLOCK_H
        # tile is unsafe (would clobber next row). Instead store a padded
        # scratch and copy back the valid columns — but only allocate padded
        # scratch for the OUTPUT (input padding is now free via PaddingMode).
        out_pad = torch.empty((rows, BLOCK_H), device=arg0_1.device,
                              dtype=torch.bfloat16)
        ct.launch(
            stream,
            (rows, 1, 1),
            _layernorm_bf16_kernel_exact,
            (gelu, arg1_1, arg2_1, out_pad, hidden, BLOCK_H),
        )
        out.copy_(out_pad[:, :hidden])
    else:
        ct.launch(
            stream,
            (rows, 1, 1),
            _layernorm_bf16_kernel_exact,
            (gelu, arg1_1, arg2_1, out, hidden, BLOCK_H),
        )
    return out.view(tuple(int(dim) for dim in _shape_param_1))
