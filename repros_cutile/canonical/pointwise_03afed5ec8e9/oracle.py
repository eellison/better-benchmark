"""cuTile port of pointwise_03afed5ec8e9: bf16 -> f32 head layout clone.

Reshape bf16 [4096, 2048] as [32, 128, 16, 128], permute to [32, 16, 128, 128],
convert to f32, and materialize as f32 [512, 128, 128].
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
HEADS = 16
SEQ = 128
DIM = 128


@ct.kernel
def _head_layout_cast_kernel(
    input_ptr,     # bf16 [B, S, H, D]
    output_ptr,    # f32  [B, H, S, D]
    BLOCK_D: ct.Constant[int],
):
    b = ct.bid(0)
    h = ct.bid(1)
    s = ct.bid(2)
    # Load a [1, 1, 1, D] slice from [B, S, H, D] input at position (b, s, h, 0)
    values = ct.load(input_ptr, index=(b, s, h, 0), shape=(1, 1, 1, BLOCK_D))
    values_f = ct.astype(values, ct.float32)
    # Store to [B, H, S, D] output at position (b, h, s, 0), reshaping to (1,1,1,D)
    values_reshaped = ct.reshape(values_f, (1, 1, 1, BLOCK_D))
    ct.store(output_ptr, index=(b, h, s, 0), tile=values_reshaped)


@oracle_impl(
    hardware="B200",
    point="af0c9f46",
    BLOCK_D=128,
)
def oracle_forward(inputs, *, BLOCK_D):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    del _shape_param_0, _shape_param_1

    # Input arg0_1: bf16 [4096, 2048] -> view as [B, S, H, D]
    input_4d = arg0_1.view(BATCH, SEQ, HEADS, DIM)
    # Output shape from _shape_param_3 = [512, 128, 128]; underlying is [B, H, S, D]
    output = torch.empty_strided(
        (BATCH * HEADS, SEQ, DIM),
        (SEQ * DIM, DIM, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    output_4d = output.view(BATCH, HEADS, SEQ, DIM)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH, HEADS, SEQ),
        _head_layout_cast_kernel,
        (input_4d, output_4d, BLOCK_D),
    )
    return output
