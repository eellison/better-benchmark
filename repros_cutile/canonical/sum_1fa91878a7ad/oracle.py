"""cuTile port of sum_1fa91878a7ad (SCHEDULER_FUSION): GPT-J bf16 row sum + fma epilogue.

For each row of shape [N_ROWS, N_COLS]:
  product = bf16-to-f32(x) * y
  row_sum = sum(product, axis=1)
  fma = -y * row_sum + product     (broadcast row_sum over cols)
  rounded_bf16 = fma.to(bf16).to(f32)
  out = (rounded * 0.0625).to(bf16)   # divide by 16
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bf16_row_sum_fma_div_kernel(
    x_ptr,           # bf16 [N_ROWS, N_COLS]
    y_ptr,           # f32 [N_ROWS, N_COLS]
    out_ptr,         # bf16 [N_ROWS, N_COLS]
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_block = ct.bid(0)
    x = ct.load(x_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    y = ct.load(y_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))

    x_f = ct.astype(x, ct.float32)
    product = x_f * y
    row_sum = ct.sum(product, axis=1, keepdims=True)  # [BLOCK_M, 1]
    fma = -y * row_sum + product
    rounded = ct.astype(ct.astype(fma, ct.bfloat16), ct.float32)
    out = ct.astype(rounded * 0.0625, ct.bfloat16)
    ct.store(out_ptr, index=(row_block, 0), tile=out)


@oracle_impl(hardware="B200", point="cdab8152", BLOCK_M=2, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1 = inputs
    out_shape = tuple(int(dim) for dim in _shape_param_1)
    n_cols = int(arg1_1.shape[-1])
    n_rows = arg1_1.numel() // n_cols

    out = torch.empty_strided(
        out_shape,
        (out_shape[1] * out_shape[2], out_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    # arg0_1 is bf16[16, 128, 128] — view as [16*128, 128] = [n_rows, n_cols].
    x_2d = arg0_1.view(n_rows, n_cols)
    # arg1_1 is f32[1, 16, 128, 128] — view as [n_rows, n_cols].
    y_2d = arg1_1.view(n_rows, n_cols)
    out_2d = out.view(n_rows, n_cols)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_rows, BLOCK_M), 1, 1),
        _bf16_row_sum_fma_div_kernel,
        (x_2d, y_2d, out_2d, BLOCK_M, BLOCK_N),
    )
    return out
