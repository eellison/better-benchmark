"""cuTile port of var_mean_3a2d56cdea00 (SCHEDULER_FUSION): residual +
LayerNorm with two returned outputs (residual bf16 tensor + normed bf16).

Ports the Triton `_residual_layernorm_two_output_kernel`. HIDDEN can be
non-power-of-2 (768) — we round the tile column count up to the next power
of 2 and mask out-of-bounds cols to 0 for the reduction. Since HIDDEN=768
and 1024 are exact multiples in these shapes, and BLOCK_H covers them, the
sum masking is straightforward.

For HIDDEN=768 we mask cols>=768 to 0 in the sum; the mean divides by
HIDDEN, so the padding-zeros don't perturb.

The variance uses `sum(x - mean)**2 / HIDDEN` in the Triton version.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _residual_layernorm_two_output_kernel(
    flat_ptr,      # bf16 [rows, HIDDEN]
    residual_ptr,  # bf16 [rows, HIDDEN]
    weight_ptr,    # bf16 [HIDDEN]
    bias_ptr,      # bf16 [HIDDEN]
    add_out_ptr,   # bf16 [rows, HIDDEN]
    norm_out_ptr,  # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_N),
                   padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_N),
                       padding_mode=ct.PaddingMode.ZERO)
    flat_f = ct.astype(flat, ct.float32)
    resid_f = ct.astype(residual, ct.float32)
    x = flat_f + resid_f
    added_bf16 = ct.astype(x, ct.bfloat16)
    ct.store(add_out_ptr, index=(row, 0), tile=added_bf16)

    # Reduction: mask cols >= HIDDEN to 0.
    col_idx = ct.arange(BLOCK_N, dtype=ct.int32)
    col_mask = ct.reshape(col_idx, (1, BLOCK_N)) < HIDDEN
    x_added = ct.astype(added_bf16, ct.float32)
    x_masked = ct.where(col_mask, x_added, 0.0)
    sum_x = ct.sum(x_masked, axis=1, keepdims=True)
    mean = sum_x * (1.0 / HIDDEN)
    centered = x_added - mean
    centered_sq = ct.where(col_mask, centered * centered, 0.0)
    variance = ct.sum(centered_sq, axis=1, keepdims=True) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + 1.0e-5)

    weight = ct.astype(ct.load(weight_ptr, index=(0,), shape=(BLOCK_N,),
                                padding_mode=ct.PaddingMode.ZERO), ct.float32)
    bias = ct.astype(ct.load(bias_ptr, index=(0,), shape=(BLOCK_N,),
                              padding_mode=ct.PaddingMode.ZERO), ct.float32)
    weight_2d = ct.reshape(weight, (1, BLOCK_N))
    bias_2d = ct.reshape(bias, (1, BLOCK_N))
    y = centered * invstd * weight_2d + bias_2d
    ct.store(norm_out_ptr, index=(row, 0), tile=ct.astype(y, ct.bfloat16))


def _resolve_shape(shape, numel):
    dims = [int(dim) for dim in shape]
    unknown = -1
    known = 1
    for idx, dim in enumerate(dims):
        if dim == -1:
            unknown = idx
        else:
            known *= dim
    if unknown >= 0:
        dims[unknown] = numel // known
    return tuple(dims)


def _next_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="63bebcf6", BLOCK_N=1024)
@oracle_impl(hardware="B200", point="1cea4d76", BLOCK_N=1024)
@oracle_impl(hardware="B200", point="f2c837cd", BLOCK_N=2048)
def oracle_forward(inputs, *, BLOCK_N):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    add_shape = tuple(int(dim) for dim in _shape_param_0)
    norm_shape = _resolve_shape(_shape_param_1, arg0_1.numel())

    add_out = torch.empty_strided(
        add_shape,
        (add_shape[1] * add_shape[2], add_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        norm_shape,
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    add_out_2d = add_out.view(rows, hidden)
    residual_2d = arg1_1.view(rows, hidden)
    block_n = _next_power_of_2(hidden)
    if BLOCK_N < block_n:
        block_n = BLOCK_N
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _residual_layernorm_two_output_kernel,
        (arg0_1, residual_2d, arg2_1, arg3_1, add_out_2d, norm_out,
         hidden, block_n),
    )
    return (add_out, norm_out)
