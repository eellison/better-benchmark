"""cuTile port of pointwise_435f6504efa7: MobileBERT row affine `(addmm+res)*scale+bias`.

Ports the Triton `_row_affine_kernel`. Each intermediate is round-tripped
through bf16 (rtne) to match the Repro's bf16 boundaries: `(addmm + res)`,
`(*scale)`, `(+bias)`. cuTile's default astype rounding is RTNE, matching the
Triton `_round_bf16_to_fp32` bit-hack used in the original.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _row_affine_kernel(
    addmm_ptr,     # (rows, n_cols) bf16
    residual_ptr,  # (rows, n_cols) bf16
    scale_ptr,     # (n_cols,) bf16
    bias_ptr,      # (n_cols,) bf16
    out_ptr,       # (rows, n_cols) bf16
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_block = ct.bid(0)

    addmm = ct.load(addmm_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    residual = ct.load(residual_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    scale = ct.load(scale_ptr, index=(0,), shape=(BLOCK_N,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_N,))

    addmm_f = ct.astype(addmm, ct.float32)
    residual_f = ct.astype(residual, ct.float32)
    # Broadcast scale/bias across rows by reshaping to (1, BLOCK_N).
    scale_2d = ct.astype(ct.reshape(scale, (1, BLOCK_N)), ct.float32)
    bias_2d = ct.astype(ct.reshape(bias, (1, BLOCK_N)), ct.float32)

    added = ct.astype(ct.astype(addmm_f + residual_f, ct.bfloat16), ct.float32)
    scaled = ct.astype(ct.astype(added * scale_2d, ct.bfloat16), ct.float32)
    out = ct.astype(scaled + bias_2d, ct.bfloat16)
    ct.store(out_ptr, index=(row_block, 0), tile=out)


# 09b2e78e: (T([32768,512], bf16), T([256,128,512], bf16), T([512], bf16), T([512], bf16), S([256,128,512]), S([32768,512]))
@oracle_impl(hardware="B200", point="09b2e78e", BLOCK_M=1, BLOCK_N=512)
# d2ddc3c7: (T([32768,128], bf16), T([256,128,128], bf16), T([128], bf16), T([128], bf16), S([256,128,128]), S([32768,128]))
@oracle_impl(hardware="B200", point="d2ddc3c7", BLOCK_M=8, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1 = inputs
    n_cols = int(arg0_1.shape[1])
    n_rows = int(arg0_1.shape[0])
    output = torch.empty_strided(
        tuple(int(dim) for dim in shape0),
        (int(shape0[1]) * n_cols, n_cols, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    # arg1_1 is (256, 128, 512) reshape to (rows, cols) for a 2D view.
    residual_2d = arg1_1.view(n_rows, n_cols)
    # output is [B, S, D] -> view as (rows, cols)
    out_2d = output.view(n_rows, n_cols)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows // BLOCK_M, 1, 1),
        _row_affine_kernel,
        (arg0_1, residual_2d, arg2_1, arg3_1, out_2d, BLOCK_M, BLOCK_N),
    )
    return output.view(tuple(int(dim) for dim in shape1))
