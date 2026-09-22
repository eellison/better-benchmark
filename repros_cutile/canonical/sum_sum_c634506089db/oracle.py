"""cuTile port of sum_sum_c634506089db: MT5 LayerNorm-backward with dropout mask.

Reference: LN-backward pattern with per-row and per-column reductions plus
a dropout-masked scaling at the final bf16 output. All reduction dims are
pow2. Uses cuTile for elementwise producer kernels, torch for reductions.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _add3_kernel(
    a_ptr, b_ptr, c_ptr, out_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    a = ct.astype(ct.load(a_ptr, index=(pid,), shape=(BLOCK,)), ct.float32)
    b = ct.astype(ct.load(b_ptr, index=(pid,), shape=(BLOCK,)), ct.float32)
    c = ct.astype(ct.load(c_ptr, index=(pid,), shape=(BLOCK,)), ct.float32)
    ct.store(out_ptr, index=(pid,), tile=a + b + c)


@ct.kernel
def _ln_bw_epilogue_kernel(
    add1_ptr,    # f32[NUMEL]
    weight_ptr,  # f32[HIDDEN]
    arg4_ptr,    # f32[NUMEL]
    arg5_ptr,    # f32[ROWS]
    arg6_ptr,    # f32[NUMEL]
    arg7_ptr,    # b8[NUMEL]
    sum2_ptr,    # f32[ROWS]
    add3_out_ptr,# f32[NUMEL]
    mul10_ptr,   # bf16[NUMEL]
    HIDDEN_: ct.Constant[int],
    INV_HIDDEN: ct.Constant[float],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    idx = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    row = idx // HIDDEN_
    col = idx - row * HIDDEN_

    add1 = ct.load(add1_ptr, index=(pid,), shape=(BLOCK,))
    weight = ct.gather(weight_ptr, col)
    arg4 = ct.load(arg4_ptr, index=(pid,), shape=(BLOCK,))
    arg5 = ct.gather(arg5_ptr, row)
    arg6 = ct.load(arg6_ptr, index=(pid,), shape=(BLOCK,))
    arg7 = ct.load(arg7_ptr, index=(pid,), shape=(BLOCK,))
    sum2 = ct.gather(sum2_ptr, row)

    mul = add1 * weight
    mul_4 = mul * arg5
    add_2 = arg6 + mul_4
    pow_1 = arg5 * arg5 * arg5
    mul_6 = sum2 * (-0.5) * pow_1
    mul_8 = arg4 * 2.0 * mul_6 * INV_HIDDEN
    add_3 = add_2 + mul_8
    ct.store(add3_out_ptr, index=(pid,), tile=add_3)

    add_3_bf16 = ct.astype(add_3, ct.bfloat16)
    arg7_bf16 = ct.astype(ct.astype(arg7, ct.float32), ct.bfloat16)
    scaled = ct.astype(ct.astype(arg7_bf16, ct.float32) * 1.1111111111111112, ct.bfloat16)
    scaled_f32 = ct.astype(scaled, ct.float32)
    mul_10 = ct.astype(ct.astype(add_3_bf16, ct.float32) * scaled_f32, ct.bfloat16)
    ct.store(mul10_ptr, index=(pid,), tile=mul_10)


def _run(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
     *_shape) = inputs
    device = arg0_1.device
    # Infer shape from arg4 which is f32[batch, seq, hidden].
    batch, seq, hidden = arg4_1.shape
    rows = batch * seq
    numel = rows * hidden
    BLOCK = 1024

    add1_flat = torch.empty(numel, device=device, dtype=torch.float32)
    a_flat = arg0_1.view(numel)
    b_flat = arg1_1.view(numel)
    c_flat = arg2_1.view(numel)
    stream = torch.cuda.current_stream()
    ct.launch(stream, (ct.cdiv(numel, BLOCK), 1, 1), _add3_kernel,
              (a_flat, b_flat, c_flat, add1_flat, BLOCK))

    add_1 = add1_flat.view(batch, seq, hidden)
    mul = add_1 * arg3_1
    mul_1 = arg4_1 * arg5_1
    mul_2 = add_1 * mul_1
    sum_1 = mul_2.sum(dim=[0, 1], keepdim=True, dtype=torch.float32)
    view_3 = sum_1.view(hidden)
    mul_3 = mul * arg4_1
    sum_2 = mul_3.sum(dim=[2], keepdim=True)

    add3_flat = torch.empty(numel, device=device, dtype=torch.float32)
    mul10_flat = torch.empty(numel, device=device, dtype=torch.bfloat16)
    weight_f = arg3_1.contiguous()
    arg4_flat = arg4_1.contiguous().view(numel)
    arg5_flat = arg5_1.contiguous().view(rows)
    arg6_flat = arg6_1.contiguous().view(numel)
    arg7_flat = arg7_1.contiguous().view(numel)
    sum2_flat = sum_2.reshape(rows)

    ct.launch(stream, (ct.cdiv(numel, BLOCK), 1, 1), _ln_bw_epilogue_kernel,
              (add1_flat, weight_f, arg4_flat, arg5_flat, arg6_flat,
               arg7_flat, sum2_flat, add3_flat, mul10_flat,
               hidden, 1.0 / hidden, BLOCK))

    add_3 = add3_flat.view(batch, seq, hidden)
    view_4 = mul10_flat.view(rows, hidden)
    permute = view_4.permute(1, 0)

    return view_3, add_3, view_4, permute


@oracle_impl(hardware="B200", point="b1beb32b", ROW_TILE=4, FINAL_BLOCK_COLS=8)
@oracle_impl(hardware="B200", point="04d5ba81", ROW_TILE=4, FINAL_BLOCK_COLS=8)
def oracle_forward(inputs, **_kwargs):
    return _run(inputs)
