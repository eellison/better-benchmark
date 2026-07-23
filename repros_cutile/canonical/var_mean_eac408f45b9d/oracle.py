"""cuTile port of var_mean_eac408f45b9d: GPT-J triple-residual LayerNorm.

Ports the Triton `_triple_residual_layernorm_bf16_kernel`. Per row:
  add0 = (x0 + x1).bf16
  add1 = (add0 + residual).bf16   (returned)
  mean = sum(add1_f32) / HIDDEN
  var = sum(add1^2) / HIDDEN - mean^2  (correction=0)
  invstd = rsqrt(var + 1e-5)
  norm = ((add1_f32 - mean) * invstd) * weight + bias -> bf16
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _triple_residual_layernorm_bf16_kernel(
    x0_ptr,        # bf16 [rows, HIDDEN]
    x1_ptr,        # bf16 [rows, HIDDEN]
    residual_ptr,  # bf16 [rows, HIDDEN]
    weight_ptr,    # bf16 [HIDDEN]
    bias_ptr,      # bf16 [HIDDEN]
    add_out_ptr,   # bf16 [rows, HIDDEN]
    norm_out_ptr,  # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    x0 = ct.load(x0_ptr, index=(row, 0), shape=(1, BLOCK_H))
    x1 = ct.load(x1_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))

    x0_f = ct.astype(x0, ct.float32)
    x1_f = ct.astype(x1, ct.float32)
    residual_f = ct.astype(residual, ct.float32)

    add0_bf16 = ct.astype(x0_f + x1_f, ct.bfloat16)
    add1_bf16 = ct.astype(ct.astype(add0_bf16, ct.float32) + residual_f, ct.bfloat16)
    ct.store(add_out_ptr, index=(row, 0), tile=add1_bf16)

    x = ct.astype(add1_bf16, ct.float32)
    mean = ct.sum(x) * (1.0 / HIDDEN)
    var = ct.sum(x * x) * (1.0 / HIDDEN) - mean * mean
    var = ct.where(var > 0.0, var, 0.0)
    invstd = ct.rsqrt(var + 1.0e-5)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(ct.astype(weight, ct.float32), (1, BLOCK_H))
    bias_2d = ct.reshape(ct.astype(bias, ct.float32), (1, BLOCK_H))

    y = (x - mean) * invstd * weight_2d + bias_2d
    ct.store(norm_out_ptr, index=(row, 0), tile=ct.astype(y, ct.bfloat16))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="d9611874", BLOCK_H=4096)
def oracle_forward(inputs, *, BLOCK_H):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, _shape1, shape2, shape3, shape4, shape5 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    add_shape = _as_shape(shape0)

    add_out = torch.empty_strided(
        add_shape,
        (add_shape[1] * add_shape[2], add_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    norm_base = torch.empty_strided(
        add_shape,
        (add_shape[1] * add_shape[2], add_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    # add_out is [1, 128, 4096]; view as [128, 4096] for the kernel.
    add_out_2d = add_out.view(rows, hidden)
    norm_out_2d = norm_base.view(rows, hidden)
    # residual is [1, 128, 4096]; view as [128, 4096].
    residual_2d = arg2_1.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1),
        _triple_residual_layernorm_bf16_kernel,
        (arg0_1, arg1_1, residual_2d, arg3_1, arg4_1, add_out_2d, norm_out_2d, hidden, BLOCK_H),
    )
    return (
        add_out,
        norm_base.view(_as_shape(shape2)),
        norm_base.view(_as_shape(shape3)),
        norm_base.view(_as_shape(shape4)),
        norm_base.view(_as_shape(shape5)),
    )
