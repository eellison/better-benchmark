"""cuTile port of var_mean_3a55f9ba3f1d: XLNet bf16 residual + LayerNorm row.

Ports the Triton `_xlnet_residual_layernorm_kernel` to cuda.tile. The eager
Repro computes bf16 add + f32 var_mean + normalize + bf16 store. cuTile's
default f32 arithmetic is IEEE round-to-nearest-even, matching the Triton
inline PTX `add.rn.f32` semantics — so we drop the PTX and use plain
arithmetic. The Triton oracle emits an fp32/bf16 blend gated by |y|>2.5, but
the eager reference only uses the fp32 path; we match it directly.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _xlnet_residual_layernorm_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    flat_f = ct.astype(flat, ct.float32)
    resid_f = ct.astype(residual, ct.float32)
    # bf16 add: eager Repro's add is bf16 → then converted to fp32
    add = ct.astype(ct.astype(flat_f + resid_f, ct.bfloat16), ct.float32)

    inv_h = 1.0 / HIDDEN
    mean = ct.sum(add, axis=1, keepdims=True) * inv_h
    centered = add - mean
    variance = ct.sum(centered * centered, axis=1, keepdims=True) * inv_h
    invstd = ct.rsqrt(variance + 1.0e-12)

    weight = ct.astype(
        ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,)), ct.float32)
    bias = ct.astype(
        ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,)), ct.float32)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))

    normalized = centered * invstd
    y = normalized * weight_2d + bias_2d
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(y, ct.bfloat16))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="c4c9bec6", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        shape0,
        shape1,
        shape2,
        shape3,
    ) = inputs
    base_shape = _shape_tuple(shape0)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])

    out = torch.empty_strided(
        base_shape,
        _contiguous_stride(base_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    # Reshape residual (bf16 [B, S, H]) into [rows, H]
    out_2d = out.view(rows, hidden)
    residual_2d = arg1_1.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _xlnet_residual_layernorm_kernel,
        (arg0_1, residual_2d, arg2_1, arg3_1, out_2d, hidden, BLOCK_H),
    )

    flat0 = out.view(_shape_tuple(shape1)).squeeze(0)
    flat1 = out.view(_shape_tuple(shape2)).squeeze(0)
    flat2 = out.view(_shape_tuple(shape3)).squeeze(0)
    return out, flat0, flat1, flat2, out[-base_shape[0]:]
