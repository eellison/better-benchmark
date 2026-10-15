"""cuTile port of sum_29c472896711: hard-swish mask-scale + column-sum.

Ports the Triton _hardswish_materialize_sum_kernel: for each column tile,
walk M=32 rows, gate by mask, apply hard-swish-like factor, store bf16
product, and reduce down to fp32 column sum. Also emit a scalar zero tensor.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


M = 32
N = 1280


@ct.kernel
def _zero_scalar_kernel(full_ptr):
    ct.store(full_ptr, index=(0,),
             tile=ct.zeros(shape=(1,), dtype=ct.float32))


@ct.kernel
def _hardswish_materialize_sum_kernel(
    mask_ptr,     # bool [M, N]
    scale_src_ptr,  # bf16 [M, N]
    gate_ptr,     # bf16 [M, N]
    out_ptr,      # bf16 [M, N]
    sum_ptr,      # f32 [N]
    M_: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    col_tile = ct.bid(0)
    mask_tile = ct.load(mask_ptr, index=(0, col_tile), shape=(M_, BLOCK_N))
    scale_src = ct.astype(
        ct.load(scale_src_ptr, index=(0, col_tile), shape=(M_, BLOCK_N)),
        ct.float32,
    )
    gate = ct.astype(
        ct.load(gate_ptr, index=(0, col_tile), shape=(M_, BLOCK_N)),
        ct.float32,
    )

    zeros_f = ct.full(shape=(M_, BLOCK_N), fill_value=0.0, dtype=ct.float32)
    mask_zero = ct.full(shape=(M_, BLOCK_N), fill_value=0, dtype=ct.bool_)
    mask_nonzero = mask_tile != mask_zero
    one_25 = ct.full(shape=(M_, BLOCK_N), fill_value=1.25, dtype=ct.float32)
    scaled_f32 = ct.where(mask_nonzero, scale_src * one_25, zeros_f)
    scaled_eager = ct.astype(ct.astype(scaled_f32, ct.bfloat16), ct.float32)
    third = ct.full(shape=(M_, BLOCK_N), fill_value=1.0 / 3.0, dtype=ct.float32)
    half = ct.full(shape=(M_, BLOCK_N), fill_value=0.5, dtype=ct.float32)
    factor = gate * third + half
    three = ct.full(shape=(M_, BLOCK_N), fill_value=3.0, dtype=ct.float32)
    neg_three = ct.full(shape=(M_, BLOCK_N), fill_value=-3.0, dtype=ct.float32)
    gated = scaled_f32 * factor
    value = ct.where(gate < three, gated, scaled_f32)
    value = ct.where(gate <= neg_three, zeros_f, value)
    value_bf16 = ct.astype(value, ct.bfloat16)

    gated_eager = scaled_eager * factor
    value_eager = ct.where(gate < three, gated_eager, scaled_eager)
    value_eager = ct.where(gate <= neg_three, zeros_f, value_eager)
    value_sum_bf16 = ct.astype(value_eager, ct.bfloat16)

    ct.store(out_ptr, index=(0, col_tile), tile=value_bf16)
    reduced = ct.sum(ct.astype(value_sum_bf16, ct.float32),
                     axis=0, keepdims=False)
    # Round through bf16
    reduced_bf16 = ct.astype(reduced, ct.bfloat16)
    reduced_out = ct.astype(reduced_bf16, ct.float32)
    ct.store(sum_ptr, index=(col_tile,), tile=reduced_out)


@oracle_impl(hardware="B200", point="c7ec772c", BLOCK_N=8)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, _shape_param_0 = inputs
    n = int(_shape_param_0[0])
    m = int(arg1_1.shape[0])

    full = torch.empty_strided((), (), device=arg1_1.device, dtype=torch.float32)
    out = torch.empty_strided((m, n), (n, 1), device=arg1_1.device,
                              dtype=torch.bfloat16)
    sum_out = torch.empty_strided((n,), (1,), device=arg1_1.device,
                                  dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(stream, (1, 1, 1), _zero_scalar_kernel, (full.view(1),))
    ct.launch(
        stream, (n // BLOCK_N, 1, 1), _hardswish_materialize_sum_kernel,
        (arg0_1, arg1_1, arg2_1, out, sum_out, m, BLOCK_N),
    )
    return full, out, out.permute(1, 0), sum_out
