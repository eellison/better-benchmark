"""cuTile port of var_mean_393456b916a6 (SCHEDULER_FUSION): residual-add
LayerNorm row kernel with three alias views of the normalized bf16 output."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _residual_layernorm_alias_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    norm_out_ptr,
    HIDDEN: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    ADD_IN_FP32: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_N))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_N))
    # Match eager: bf16 add, then f32 cast for the rest of the arithmetic.
    add_bf16 = flat + residual
    x = ct.astype(add_bf16, ct.float32)

    inv_hidden = 1.0 / HIDDEN
    row_sum = ct.sum(x)
    mean = row_sum * inv_hidden
    centered = x - mean
    var = ct.sum(centered * centered) * inv_hidden
    invstd = ct.rsqrt(var + 1.0e-12)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_N,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_N,))
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_N))
    bias_2d = ct.reshape(bias_f, (1, BLOCK_N))
    y = centered * invstd * weight_2d + bias_2d
    ct.store(norm_out_ptr, index=(row, 0), tile=ct.astype(y, ct.bfloat16))


@oracle_impl(hardware="B200", point="d4701d13", BLOCK_N=4096, ADD_IN_FP32=0)
@oracle_impl(hardware="B200", point="63bebcf6", BLOCK_N=1024, ADD_IN_FP32=0)
@oracle_impl(hardware="B200", point="107053b2", BLOCK_N=2048, ADD_IN_FP32=1)
@oracle_impl(hardware="B200", point="82b81ff3", BLOCK_N=1024, ADD_IN_FP32=0)
@oracle_impl(hardware="B200", point="94a8a62c", BLOCK_N=256, ADD_IN_FP32=0)
def oracle_forward(inputs, *, BLOCK_N, ADD_IN_FP32):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2, shape3 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    norm_shape = tuple(int(dim) for dim in shape0)

    # residual is (batch, seq, hidden) contiguous; reshape to (rows, hidden).
    residual_2d = arg1_1.view(rows, hidden)

    norm_out = torch.empty_strided(
        norm_shape,
        (norm_shape[1] * norm_shape[2], norm_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    # Cast norm_out's underlying storage as (rows, hidden) for cuTile store.
    norm_flat = norm_out.view(rows, hidden)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _residual_layernorm_alias_kernel,
        (arg0_1, residual_2d, arg2_1, arg3_1, norm_flat, hidden, BLOCK_N, ADD_IN_FP32),
    )
    return (
        norm_out,
        norm_out.view(tuple(int(dim) for dim in shape1)),
        norm_out.view(tuple(int(dim) for dim in shape2)),
        norm_out.view(tuple(int(dim) for dim in shape3)),
    )
