"""cuTile port of var_mean_8577bcc46cb1 (SCHEDULER_FUSION): BEiT scale/residual
LayerNorm per row of `[ROWS, HIDDEN=768]`.

  scaled_bf16 = arg1 * arg0
  add_bf16 = arg2 + scaled_bf16     (returned)
  x_f = add_bf16.to(f32)
  mean = sum(x_f) / HIDDEN
  var = mean(x^2) - mean^2
  invstd = rsqrt(var + 1e-6)
  final = ((x - mean) * invstd * weight + bias).to(bf16)
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HIDDEN = 768


@ct.kernel
def _beit_layernorm_kernel(
    scaled_ptr,    # bf16 [ROWS, HIDDEN] flat
    scale_ptr,     # bf16 [HIDDEN]
    residual_ptr,  # bf16 [ROWS, HIDDEN]
    weight_ptr,    # bf16 [HIDDEN]
    bias_ptr,      # bf16 [HIDDEN]
    add_out_ptr,   # bf16 [ROWS, HIDDEN]
    final_ptr,     # bf16 [ROWS, HIDDEN]
    HIDDEN_: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    scaled_src = ct.load(scaled_ptr, index=(row, 0), shape=(1, BLOCK_H))
    scale = ct.load(scale_ptr, index=(0,), shape=(BLOCK_H,))
    scale_f = ct.astype(scale, ct.float32)
    scaled_bf16 = ct.astype(
        ct.astype(scaled_src, ct.float32) * ct.reshape(scale_f, (1, BLOCK_H)),
        ct.bfloat16,
    )
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    add_bf16 = ct.astype(
        ct.astype(residual, ct.float32) + ct.astype(scaled_bf16, ct.float32),
        ct.bfloat16,
    )
    ct.store(add_out_ptr, index=(row, 0), tile=add_bf16)

    x = ct.astype(add_bf16, ct.float32)
    inv_h = 1.0 / HIDDEN_
    x_sum = ct.sum(x, axis=1, keepdims=True)
    x_sq_sum = ct.sum(x * x, axis=1, keepdims=True)
    mean = x_sum * inv_h
    variance = x_sq_sum * inv_h - mean * mean
    centered = x - mean
    invstd = ct.rsqrt(variance + 1.0e-6)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_f = ct.astype(ct.reshape(weight, (1, BLOCK_H)), ct.float32)
    bias_f = ct.astype(ct.reshape(bias, (1, BLOCK_H)), ct.float32)
    affine = centered * invstd * weight_f + bias_f
    final = ct.astype(affine, ct.bfloat16)
    ct.store(final_ptr, index=(row, 0), tile=final)


@oracle_impl(hardware="B200", point="e781f0b8", BLOCK_H=1024, ROW_BLOCK=4)
def oracle_forward(inputs, *, BLOCK_H, ROW_BLOCK):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _s0, _s1 = inputs
    view_shape = tuple(int(dim) for dim in _s0)
    flat_shape = tuple(int(dim) for dim in _s1)
    rows = view_shape[0] * view_shape[1]
    hidden = int(arg0_1.shape[1])

    scaled_2d = arg0_1.view(rows, hidden)
    residual_2d = arg2_1.view(rows, hidden)

    add_out = torch.empty(
        view_shape, device=arg0_1.device, dtype=torch.bfloat16
    )
    final = torch.empty(
        flat_shape, device=arg0_1.device, dtype=torch.bfloat16
    )
    add_out_2d = add_out.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _beit_layernorm_kernel,
        (scaled_2d, arg1_1, residual_2d, arg3_1, arg4_1, add_out_2d, final, hidden, BLOCK_H),
    )
    return add_out, final
