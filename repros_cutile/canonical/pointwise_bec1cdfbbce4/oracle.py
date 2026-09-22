"""cuTile port of pointwise_bec1cdfbbce4: Lennard-Jones outer-broadcast tanh.

Takes f32 inputs `arg0[16]`, `arg1[16,1]`, `arg2[128,1]` and computes bf16
`arg2_bf16 = to_bf16(arg2)` and bf16 `tanh(bf16(arg2*arg1 + arg0))` as a
`[128,16]` outer product.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 128
COLS = 16


@ct.kernel
def _outer_tanh_kernel(
    arg0_ptr,   # (16,) f32
    arg1_ptr,   # (16,) f32 (flattened from [16,1])
    arg2_ptr,   # (128,) f32 (flattened from [128,1])
    arg2_bf16_out_ptr,  # (128,) bf16
    tanh_out_ptr,       # (128, 16) bf16
    ROWS_C: ct.Constant[int],
    COLS_C: ct.Constant[int],
):
    # Full outer product in one tile.
    arg2_f32 = ct.load(arg2_ptr, index=(0,), shape=(ROWS_C,))
    arg2_bf16 = ct.astype(arg2_f32, ct.bfloat16)
    ct.store(arg2_bf16_out_ptr, index=(0,), tile=arg2_bf16)

    arg1_f32 = ct.load(arg1_ptr, index=(0,), shape=(COLS_C,))
    arg0_f32 = ct.load(arg0_ptr, index=(0,), shape=(COLS_C,))
    arg1_bf16 = ct.astype(arg1_f32, ct.bfloat16)
    arg0_bf16 = ct.astype(arg0_f32, ct.bfloat16)

    # Broadcast: arg2_bf16 (128,) reshape to (128,1); arg1_bf16 (16,) reshape to (1,16)
    arg2_2d = ct.reshape(ct.astype(arg2_bf16, ct.float32), (ROWS_C, 1))
    arg1_2d = ct.reshape(ct.astype(arg1_bf16, ct.float32), (1, COLS_C))
    arg0_2d = ct.reshape(ct.astype(arg0_bf16, ct.float32), (1, COLS_C))

    product = arg2_2d * arg1_2d
    affine = product + arg0_2d
    affine_bf16 = ct.astype(affine, ct.bfloat16)
    result_f32 = ct.astype(affine_bf16, ct.float32)
    # tanh in f32 then cast to bf16
    result_bf16 = ct.astype(ct.tanh(result_f32), ct.bfloat16)
    ct.store(tanh_out_ptr, index=(0, 0), tile=result_bf16)


@oracle_impl(hardware="B200", point="1b1f1ebc", BLOCK_M=8)
def oracle_forward(inputs, *, BLOCK_M: int):
    arg0_1, arg1_1, arg2_1 = inputs
    arg2_bf16 = torch.empty_strided(
        tuple(arg2_1.shape), tuple(arg2_1.stride()),
        device=arg2_1.device, dtype=torch.bfloat16,
    )
    tanh = torch.empty_strided(
        (ROWS, COLS), (COLS, 1),
        device=arg2_1.device, dtype=torch.bfloat16,
    )

    # Flatten to 1D for cuTile loads
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (1, 1, 1),
        _outer_tanh_kernel,
        (arg0_1.view(16), arg1_1.view(16), arg2_1.view(128),
         arg2_bf16.view(128), tanh, ROWS, COLS),
    )
    return arg2_bf16, tanh
