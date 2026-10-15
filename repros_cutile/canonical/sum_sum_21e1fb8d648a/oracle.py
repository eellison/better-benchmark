"""cuTile port of sum_sum_21e1fb8d648a: ALBERT LN-backward dual row-sum.

Ports the Triton `_albert_ln_backward_dual_sum_kernel` — for each row of
`[ROWS=4096, HIDDEN=4096]`:
  add_val = residual + mm.to(f32)
  scaled = add_val * weight
  sum_scaled, sum_scaled_normed = reductions over HIDDEN
  epilogue = row_scale * (scaled*HIDDEN - sum_scaled - normed*sum_scaled_normed)
Emits: (add_val f32, epilogue f32, epilogue.bf16, epilogue.bf16.permute(1,0)).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _albert_ln_backward_dual_sum_kernel(
    mm_bf16_ptr,      # bf16 [ROWS, HIDDEN]
    residual_ptr,     # f32  [ROWS, HIDDEN]
    weight_ptr,       # f32  [HIDDEN]
    normed_ptr,       # f32  [ROWS, HIDDEN]
    row_scale_ptr,    # f32  [ROWS]
    add_out_ptr,      # f32  [ROWS, HIDDEN]
    epilogue_out_ptr, # f32  [ROWS, HIDDEN]
    bf16_out_ptr,     # bf16 [ROWS, HIDDEN]
    HIDDEN_C: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)

    mm = ct.astype(
        ct.load(mm_bf16_ptr, index=(row, 0), shape=(1, BLOCK_N)), ct.float32
    )
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_N))
    add_val = residual + mm
    ct.store(add_out_ptr, index=(row, 0), tile=add_val)

    weight_1d = ct.load(weight_ptr, index=(0,), shape=(BLOCK_N,))
    weight = ct.reshape(weight_1d, (1, BLOCK_N))
    normed = ct.load(normed_ptr, index=(row, 0), shape=(1, BLOCK_N))
    scaled = add_val * weight

    sum_scaled = ct.sum(scaled, axis=1, keepdims=True)  # (1, 1)
    scaled_normed = scaled * normed
    sum_scaled_normed = ct.sum(scaled_normed, axis=1, keepdims=True)

    row_scale = ct.reshape(
        ct.load(row_scale_ptr, index=(row,), shape=(1,)), (1, 1)
    )
    term0 = scaled * float(HIDDEN_C)
    term1 = term0 - sum_scaled
    term2 = normed * sum_scaled_normed
    epilogue = row_scale * (term1 - term2)

    ct.store(epilogue_out_ptr, index=(row, 0), tile=epilogue)
    ct.store(bf16_out_ptr, index=(row, 0), tile=ct.astype(epilogue, ct.bfloat16))


@oracle_impl(hardware="B200", point="6542053c", BLOCK_N=4096)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape_param_0, _shape_param_1 = inputs
    del _shape_param_0

    out_add = torch.empty_strided(
        tuple(int(dim) for dim in arg1_1.shape),
        tuple(int(stride) for stride in arg1_1.stride()),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    out_epilogue = torch.empty_strided(
        tuple(int(dim) for dim in arg3_1.shape),
        tuple(int(stride) for stride in arg3_1.stride()),
        device=arg3_1.device,
        dtype=torch.float32,
    )
    out_bf16 = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_1),
        (int(_shape_param_1[1]), 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    rows = int(arg1_1.numel() // arg1_1.shape[-1])
    hidden = int(arg1_1.shape[-1])

    # 2D views for cuTile
    arg0_2d = arg0_1.view(rows, hidden)
    arg1_2d = arg1_1.view(rows, hidden)
    arg3_2d = arg3_1.view(rows, hidden)
    arg4_1d = arg4_1.view(rows)
    out_add_2d = out_add.view(rows, hidden)
    out_epilogue_2d = out_epilogue.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _albert_ln_backward_dual_sum_kernel,
        (arg0_2d, arg1_2d, arg2_1, arg3_2d, arg4_1d,
         out_add_2d, out_epilogue_2d, out_bf16,
         hidden, BLOCK_N),
    )

    return out_add, out_epilogue, out_bf16, out_bf16.permute(1, 0)
