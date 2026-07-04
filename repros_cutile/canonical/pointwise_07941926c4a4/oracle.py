"""cuTile port of pointwise_07941926c4a4: ALBERT scaled attention-score layout clone.

Reads bf16[8,64,64,512] view of the input, permutes to [8,512,64,64] with a
0.3535533905932738 scalar multiply (bf16 rounded), and materializes as bf16[4096,4096].
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _scaled_score_layout_clone_kernel(
    src,       # bf16 [8, 64, 64, 512]
    dst,       # bf16 [8, 512, 64, 64]
    BLOCK_D: ct.Constant[int],
):
    b = ct.bid(0)
    k = ct.bid(1)
    d_tile = ct.bid(2)

    # Load [1, 1, 64, BLOCK_D] tile from src
    x = ct.load(src, index=(b, k, 0, d_tile), shape=(1, 1, 64, BLOCK_D))
    # Reshape to (64, BLOCK_D)
    x = ct.reshape(x, (64, BLOCK_D))
    # Transpose to (BLOCK_D, 64)
    x_t = ct.transpose(x)
    # Scale in fp32 to match bf16 rounding boundary
    x_f = ct.astype(x_t, ct.float32) * 0.3535533905932738
    x_bf = ct.astype(x_f, ct.bfloat16)
    # Reshape for store: (1, BLOCK_D, 1, 64)
    x_bf = ct.reshape(x_bf, (1, BLOCK_D, 1, 64))
    ct.store(dst, index=(b, d_tile, k, 0), tile=x_bf)


@oracle_impl(hardware="B200", point="6274ca22")
def oracle_forward(inputs):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    output_shape = tuple(int(dim) for dim in _shape_param_2)  # [4096, 4096]
    out = torch.empty_strided(
        output_shape,
        (output_shape[1], 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    # Views: input [8, 64, 64, 512], output [8, 512, 64, 64]
    src4 = arg0_1.view(8, 64, 64, 512)
    dst4 = out.view(8, 512, 64, 64)

    BLOCK_D = 64
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (8, 64, 512 // BLOCK_D),
        _scaled_score_layout_clone_kernel,
        (src4, dst4, BLOCK_D),
    )
    return out, out.permute(1, 0)
