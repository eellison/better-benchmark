"""cuTile port of pointwise_f2a03dbd04dd: scaled attention-key permute.

Input arg0_1 is contiguous [B*S, H*D] (logical [B, S, H, D]).
Output is contiguous [B*H, D, S] (logical [B, H, D, S]) with values scaled by
1/sqrt(8) = 0.3535533905932738.

Directly ports the Triton `_scaled_key_layout_kernel`: a 2D grid over
(ynumel = B*H*D, S) that reads via flat pointer offsets and writes to the
contiguous output. Uses scatter to allow larger tile sizes with masking.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 0.3535533905932738


@ct.kernel
def _scaled_key_layout_kernel(
    input_ptr,   # bf16 flat storage of arg0_1 (contiguous)
    output_ptr,  # bf16 flat storage of output (contiguous [ynumel, S])
    S_: ct.Constant[int],
    HIDDEN_: ct.Constant[int],   # H * D
    YNUMEL_: ct.Constant[int],   # B * H * D
    BLOCK_Y: ct.Constant[int],
    BLOCK_X: ct.Constant[int],
):
    y_block = ct.bid(1)
    x_block = ct.bid(0)
    y = y_block * BLOCK_Y + ct.arange(BLOCK_Y, dtype=ct.int32)
    x = x_block * BLOCK_X + ct.arange(BLOCK_X, dtype=ct.int32)

    y_2d = ct.reshape(y, (BLOCK_Y, 1))
    x_2d = ct.reshape(x, (1, BLOCK_X))

    valid = (y_2d < YNUMEL_) & (x_2d < S_)

    inner = y_2d - (y_2d // HIDDEN_) * HIDDEN_
    batch = y_2d // HIDDEN_
    input_offsets = inner + HIDDEN_ * (x_2d + S_ * batch)
    output_offsets = x_2d + S_ * y_2d

    values_bf = ct.gather(input_ptr, input_offsets)
    values = ct.astype(values_bf, ct.float32) * SCALE
    values_bf16 = ct.astype(values, ct.bfloat16)
    ct.scatter(output_ptr, output_offsets, values_bf16, mask=valid)


@oracle_impl(hardware="B200", point="d87997ca", BLOCK_Y=64, BLOCK_X=64)
@oracle_impl(hardware="B200", point="d20f46e2", BLOCK_Y=64, BLOCK_X=64)
@oracle_impl(hardware="B200", point="bd432928", BLOCK_Y=16, BLOCK_X=64)
@oracle_impl(hardware="B200", point="1a8eaeba", BLOCK_Y=16, BLOCK_X=64)
@oracle_impl(hardware="B200", point="b8160d07", BLOCK_Y=16, BLOCK_X=64)
@oracle_impl(hardware="B200", point="ad7b2a2c", BLOCK_Y=16, BLOCK_X=64)
@oracle_impl(hardware="B200", point="3ab46e72", BLOCK_Y=16, BLOCK_X=64)
def oracle_forward(inputs, *, BLOCK_Y: int, BLOCK_X: int):
    arg0_1, _s0, _s1, s2, s3 = inputs
    B = int(s2[0])
    H = int(s2[1])
    D = int(s2[2])
    S = int(s2[3])
    output = torch.empty_strided(
        tuple(int(dim) for dim in s3),
        (D * S, S, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )

    ynumel = B * H * D
    hidden = H * D
    src_flat = arg0_1.view(-1)
    dst_flat = output.view(-1)

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(S, BLOCK_X), ct.cdiv(ynumel, BLOCK_Y), 1)
    ct.launch(
        stream, grid, _scaled_key_layout_kernel,
        (src_flat, dst_flat, S, hidden, ynumel, BLOCK_Y, BLOCK_X),
    )
    return output
