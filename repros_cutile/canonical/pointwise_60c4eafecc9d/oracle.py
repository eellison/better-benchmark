"""cuTile port of pointwise_60c4eafecc9d (SCHEDULER_FUSION): ConvBert
cast/add/mul + returned split-head permute view over the input projection.

The Repro's activation is bf16[32, 384, 512]; it's permuted to [32, 512, 384]
and added to projection[32, 512, 384] (view of [16384, 384]).

We use a permuted view (strided) for activation so cuTile can load contiguous tiles.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
CHANNELS = 384
SEQ = 512
N_ROWS = BATCH * SEQ


@ct.kernel
def _cast_add_mul_kernel(
    activation_permuted_ptr,  # bf16 [16384, 384] (permuted view of [32, 384, 512])
    bias_ptr,                 # f32 [384]
    projection_ptr,           # bf16 [16384, 384]
    out_ptr,                  # bf16 [16384, 384]
    BLOCK_M: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row_block = ct.bid(0)
    col_block = ct.bid(1)

    activation = ct.load(
        activation_permuted_ptr,
        index=(row_block, col_block),
        shape=(BLOCK_M, BLOCK_C),
    )
    projection = ct.load(
        projection_ptr,
        index=(row_block, col_block),
        shape=(BLOCK_M, BLOCK_C),
    )
    bias = ct.load(bias_ptr, index=(col_block,), shape=(BLOCK_C,))
    bias_2d = ct.reshape(bias, (1, BLOCK_C))
    bias_f = ct.astype(bias_2d, ct.float32)

    activation_f = ct.astype(activation, ct.float32)
    rounded = ct.astype(
        ct.astype(activation_f + bias_f, ct.bfloat16), ct.float32
    )
    projection_f = ct.astype(projection, ct.float32)
    out = ct.astype(rounded * projection_f, ct.bfloat16)
    ct.store(out_ptr, index=(row_block, col_block), tile=out)


@oracle_impl(hardware="B200", point="06a51e7f", BLOCK_M=16, BLOCK_C=128)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_C):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_1

    # arg0_1 is bf16[32, 384, 512] (i.e. activation with channels dim 1 and seq dim 2).
    # We need `activation.permute(0, 2, 1)` shape [32, 512, 384], viewed as [16384, 384].
    # Use as_strided to get the permuted contiguous-looking 2D view.
    activation_permuted = arg0_1.permute(0, 2, 1).contiguous().view(N_ROWS, CHANNELS)

    # arg1_1 is f32[384, 1] - flatten to f32[384].
    bias_flat = arg1_1.view(CHANNELS)

    # arg2_1 is bf16[16384, 384] — already the layout we need.
    projection = arg2_1

    # Build head-view (permute alias) — this is a metadata output.
    batch = int(_shape_param_0[0])
    seq = int(_shape_param_0[1])
    channels = int(_shape_param_0[2])
    head_view = arg2_1.as_strided(
        (batch, channels // 64, seq, 64),
        (seq * channels, 64, channels, 1),
    )
    out = torch.empty_like(arg2_1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(N_ROWS, BLOCK_M), ct.cdiv(CHANNELS, BLOCK_C), 1),
        _cast_add_mul_kernel,
        (activation_permuted, bias_flat, projection, out, BLOCK_M, BLOCK_C),
    )
    return out, head_view
