"""cuTile port of sum_d6239b8319dc: scalar zero + where + channel sum.

Ports the Triton `_masked_where_channel_sum_kernel` — produces a scalar bf16
zero, a bf16 where(pred<=0, 0, values) output, and an fp32 channel sum.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _masked_where_channel_sum_kernel(
    pred_ptr,       # bf16 [n, c]
    value_ptr,      # bf16 [n, c]
    where_out_ptr,  # bf16 [n, c]
    sum_out_ptr,    # fp32 [c]
    n_size: ct.Constant[int],
    c_size: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    pid_c = ct.bid(0)
    pred = ct.load(pred_ptr, index=(0, pid_c), shape=(BLOCK_N, BLOCK_C))
    values = ct.load(value_ptr, index=(0, pid_c), shape=(BLOCK_N, BLOCK_C))
    pred_f = ct.astype(pred, ct.float32)
    values_f = ct.astype(values, ct.float32)
    zero = ct.zeros((BLOCK_N, BLOCK_C), dtype=ct.float32)
    selected = ct.where(pred_f <= 0.0, zero, values_f)
    selected_bf16 = ct.astype(selected, ct.bfloat16)
    ct.store(where_out_ptr, index=(0, pid_c), tile=selected_bf16)
    totals = ct.sum(ct.astype(selected_bf16, ct.float32), axis=0)
    ct.store(sum_out_ptr, index=(pid_c,), tile=totals)


@ct.kernel
def _scalar_zero_kernel(out_ptr):
    ct.store(out_ptr, index=(0,), tile=ct.zeros(shape=(1,), dtype=ct.bfloat16))


def _oracle_forward_impl(inputs, *, BLOCK_N: int, BLOCK_C: int):
    arg0_1, arg1_1 = inputs
    n_size = int(arg0_1.shape[0])
    c_size = int(arg0_1.shape[1])

    full = torch.empty_strided((), (), device=arg0_1.device, dtype=torch.bfloat16)
    where = torch.empty_strided(
        tuple(arg1_1.shape),
        tuple(arg1_1.stride()),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    converted_sum = torch.empty_strided(
        (c_size,),
        (1,),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    # Views: [N, C, 1, 1] -> [N, C]
    pred_2d = arg0_1.view(n_size, c_size)
    val_2d = arg1_1.view(n_size, c_size)
    where_2d = where.view(n_size, c_size)

    stream = torch.cuda.current_stream()
    ct.launch(stream, (1, 1, 1), _scalar_zero_kernel, (full.view(1),))
    ct.launch(
        stream,
        (ct.cdiv(c_size, BLOCK_C), 1, 1),
        _masked_where_channel_sum_kernel,
        (pred_2d, val_2d, where_2d, converted_sum, n_size, c_size, BLOCK_N, BLOCK_C),
    )
    return full, where, converted_sum


# f0206885: (T([128,768,1,1], bf16), T([128,768,1,1], bf16))
@oracle_impl(hardware="B200", point="f0206885", BLOCK_N=128, BLOCK_C=8)
# 32b702a1: (T([512,240,1,1], bf16), T([512,240,1,1], bf16))
@oracle_impl(hardware="B200", point="32b702a1", BLOCK_N=512, BLOCK_C=8)
# 09bcbc35: (T([128,384,1,1], bf16), T([128,384,1,1], bf16))
@oracle_impl(hardware="B200", point="09bcbc35", BLOCK_N=128, BLOCK_C=8)
# f4357794: (T([32,240,1,1], bf16), T([32,240,1,1], bf16))
@oracle_impl(hardware="B200", point="f4357794", BLOCK_N=32, BLOCK_C=16)
def oracle_forward(inputs, *, BLOCK_N: int, BLOCK_C: int):
    return _oracle_forward_impl(inputs, BLOCK_N=BLOCK_N, BLOCK_C=BLOCK_C)
