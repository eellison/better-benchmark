"""cuTile port of pointwise_f2ae34e9b8f1: Longformer three-input bf16 residual add.

NEW_PATTERN: three bf16 [8192, 768] inputs are viewed as [1024, 8, 768],
permuted [1, 0, 2] to [8, 1024, 768], summed in fp32 with a fp32 [8, 1024, 768]
residual, and the fp32 result is stored contiguously.

BLOCK_N=256 divides 768 evenly (three tiles per row), and cuTile requires
tile shapes to be powers of two.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _longformer_three_bf16_residual_kernel(
    a_ptr,        # bf16 [1024, 8, 768]
    b_ptr,        # bf16 [1024, 8, 768]
    c_ptr,        # bf16 [1024, 8, 768]
    residual_ptr, # f32 [8, 1024, 768]
    out_ptr,      # f32 [8, 1024, 768]
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    batch = ct.bid(0)   # 0..8
    seq_b = ct.bid(1)   # 0..1024/BLOCK_M
    d_b = ct.bid(2)     # 0..768/BLOCK_N

    a = ct.load(a_ptr, index=(seq_b, batch, d_b), shape=(BLOCK_M, 1, BLOCK_N))
    b = ct.load(b_ptr, index=(seq_b, batch, d_b), shape=(BLOCK_M, 1, BLOCK_N))
    c = ct.load(c_ptr, index=(seq_b, batch, d_b), shape=(BLOCK_M, 1, BLOCK_N))
    a_f = ct.astype(a, ct.float32)
    b_f = ct.astype(b, ct.float32)
    c_f = ct.astype(c, ct.float32)
    sum_abc = a_f + b_f + c_f  # (BLOCK_M, 1, BLOCK_N)
    sum_abc = ct.reshape(sum_abc, (1, BLOCK_M, BLOCK_N))

    residual = ct.load(residual_ptr, index=(batch, seq_b, d_b), shape=(1, BLOCK_M, BLOCK_N))
    out = residual + sum_abc
    ct.store(out_ptr, index=(batch, seq_b, d_b), tile=out)


@oracle_impl(hardware="B200", point="b940b015", BLOCK_M=4, BLOCK_N=256)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs

    a3 = arg0_1.view(1024, 8, 768)
    b3 = arg1_1.view(1024, 8, 768)
    c3 = arg2_1.view(1024, 8, 768)

    output = torch.empty_strided(
        (8, 1024, 768),
        (786432, 768, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    grid = (8, 1024 // BLOCK_M, 768 // BLOCK_N)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        grid,
        _longformer_three_bf16_residual_kernel,
        (a3, b3, c3, arg3_1, output, BLOCK_M, BLOCK_N),
    )
    return output
