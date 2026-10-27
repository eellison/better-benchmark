"""cuTile port of sum_sum_sum_e39690a4a03d: MobileBERT multi-add + column reductions.

Structure matches Triton reference:
  Kernel 1 (_token_partials_and_side_kernel): compute add2 = arg1 + arg0 + arg2 + arg3
    plus side_f32 = add2 * arg5, side_bf16 = to_bf16(side_f32), mul = add2 * arg4,
    then reduce along the token axis into per-batch partial buffers of shape
    (BATCH, HIDDEN) for out0 (sum add2), out1 (sum mul), out3 (sum side_bf16 as f32).
  Kernel 2 (_finalize_batch_partials_kernel): reduce partials along the BATCH axis
    to yield the final [HIDDEN] column sums.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 256
TOKENS = 128
HIDDEN = 512


@ct.kernel
def _token_partials_and_side_kernel(
    arg0_ptr,     # bf16 [BATCH*TOKENS, HIDDEN]
    arg1_ptr,     # f32  [BATCH, TOKENS, HIDDEN]
    arg2_ptr,     # bf16 [BATCH*TOKENS, HIDDEN]
    arg3_ptr,     # bf16 [BATCH*TOKENS, HIDDEN]
    arg4_ptr,     # f32  [BATCH, TOKENS, HIDDEN]
    scale_ptr,    # f32  [HIDDEN]
    side_f32_ptr, # f32  [BATCH*TOKENS, HIDDEN]
    side_bf16_ptr,# bf16 [BATCH*TOKENS, HIDDEN]
    partial0_ptr, # f32  [BATCH, HIDDEN]  = sum over tokens of add2
    partial1_ptr, # f32  [BATCH, HIDDEN]  = sum over tokens of mul
    partial3_ptr, # f32  [BATCH, HIDDEN]  = sum over tokens of side_bf16 as f32
    BLOCK_C: ct.Constant[int],
):
    b = ct.bid(0)       # batch dim
    col_block = ct.bid(1)

    # Load scale for this col block.
    scale = ct.load(scale_ptr, index=(col_block,), shape=(BLOCK_C,))
    scale_2d = ct.reshape(scale, (1, BLOCK_C))

    # Compute the (TOKENS, BLOCK_C) tile for this (b, col_block).
    # arg1, arg4 are 3D [BATCH, TOKENS, HIDDEN].
    # arg0, arg2, arg3, side_* are 2D [BATCH*TOKENS, HIDDEN].
    # We treat 2D via a load with a start row = b * TOKENS. But cuTile 2D
    # loads use tile-space coords; index=(b, col_block) with shape=(TOKENS, BLOCK_C)
    # loads rows [b*TOKENS : (b+1)*TOKENS], cols [col_block*BLOCK_C : ...].
    # For that to work, TOKENS must divide BATCH*TOKENS (it does) and BLOCK_C
    # must divide HIDDEN.
    arg0 = ct.load(arg0_ptr, index=(b, col_block), shape=(TOKENS, BLOCK_C))
    arg2 = ct.load(arg2_ptr, index=(b, col_block), shape=(TOKENS, BLOCK_C))
    arg3 = ct.load(arg3_ptr, index=(b, col_block), shape=(TOKENS, BLOCK_C))
    # For 3D loads, index=(b, 0, col_block), shape=(1, TOKENS, BLOCK_C).
    arg1 = ct.load(arg1_ptr, index=(b, 0, col_block), shape=(1, TOKENS, BLOCK_C))
    arg1_2d = ct.reshape(arg1, (TOKENS, BLOCK_C))
    arg4 = ct.load(arg4_ptr, index=(b, 0, col_block), shape=(1, TOKENS, BLOCK_C))
    arg4_2d = ct.reshape(arg4, (TOKENS, BLOCK_C))

    x = (
        arg1_2d
        + ct.astype(arg0, ct.float32)
        + ct.astype(arg2, ct.float32)
        + ct.astype(arg3, ct.float32)
    )
    mul = x * arg4_2d
    side_f32 = x * scale_2d
    side_bf16 = ct.astype(side_f32, ct.bfloat16)
    side_bf16_f32 = ct.astype(side_bf16, ct.float32)

    ct.store(side_f32_ptr, index=(b, col_block), tile=side_f32)
    ct.store(side_bf16_ptr, index=(b, col_block), tile=side_bf16)

    # Per-batch column partial sums (reduce token dim, axis=0).
    p0 = ct.sum(x, axis=0)                # shape (BLOCK_C,)
    p1 = ct.sum(mul, axis=0)
    p3 = ct.sum(side_bf16_f32, axis=0)
    p0_2d = ct.reshape(p0, (1, BLOCK_C))
    p1_2d = ct.reshape(p1, (1, BLOCK_C))
    p3_2d = ct.reshape(p3, (1, BLOCK_C))
    ct.store(partial0_ptr, index=(b, col_block), tile=p0_2d)
    ct.store(partial1_ptr, index=(b, col_block), tile=p1_2d)
    ct.store(partial3_ptr, index=(b, col_block), tile=p3_2d)


@ct.kernel
def _finalize_batch_partials_kernel(
    partial0_ptr, # f32 [BATCH, HIDDEN]
    partial1_ptr, # f32 [BATCH, HIDDEN]
    partial3_ptr, # f32 [BATCH, HIDDEN]
    out0_ptr,     # f32 [HIDDEN]
    out1_ptr,     # f32 [HIDDEN]
    out3_ptr,     # f32 [HIDDEN]
    BLOCK_C: ct.Constant[int],
    BATCH_C: ct.Constant[int],
):
    col_block = ct.bid(0)
    # Load full BATCH x BLOCK_C tile.
    p0 = ct.load(partial0_ptr, index=(0, col_block), shape=(BATCH_C, BLOCK_C))
    p1 = ct.load(partial1_ptr, index=(0, col_block), shape=(BATCH_C, BLOCK_C))
    p3 = ct.load(partial3_ptr, index=(0, col_block), shape=(BATCH_C, BLOCK_C))
    s0 = ct.sum(p0, axis=0)
    s1 = ct.sum(p1, axis=0)
    s3 = ct.sum(p3, axis=0)
    # For out3: round to bf16 then back to f32 (matches Triton's finalize).
    s3_bf = ct.astype(s3, ct.bfloat16)
    s3_out = ct.astype(s3_bf, ct.float32)
    ct.store(out0_ptr, index=(col_block,), tile=s0)
    ct.store(out1_ptr, index=(col_block,), tile=s1)
    ct.store(out3_ptr, index=(col_block,), tile=s3_out)


@oracle_impl(hardware="B200", point="2394756c", xblock=64)
def oracle_forward(inputs, *, xblock):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1,
        _shape_param_0, _shape_param_1, _shape_param_2,
        shape3, shape4, shape5, shape6,
    ) = inputs

    rows = BATCH * TOKENS  # 32768
    full_shape = (BATCH, TOKENS, HIDDEN)
    device = arg1_1.device

    # 2D views for the bf16 inputs (arg0, arg2, arg3).
    arg0_2d = arg0_1.view(rows, HIDDEN)
    arg2_2d = arg2_1.view(rows, HIDDEN)
    arg3_2d = arg3_1.view(rows, HIDDEN)

    # Output allocations.
    side_f32 = torch.empty_strided(
        full_shape, (TOKENS * HIDDEN, HIDDEN, 1),
        device=device, dtype=torch.float32,
    )
    side_bf16 = torch.empty_strided(
        (rows, HIDDEN), (HIDDEN, 1), device=device, dtype=torch.bfloat16,
    )
    partial0 = torch.empty((BATCH, HIDDEN), device=device, dtype=torch.float32)
    partial1 = torch.empty((BATCH, HIDDEN), device=device, dtype=torch.float32)
    partial3 = torch.empty((BATCH, HIDDEN), device=device, dtype=torch.float32)
    out0 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out1 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out3 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)

    BLOCK_C = 128
    if HIDDEN % BLOCK_C != 0:
        raise NotImplementedError("BLOCK_C does not divide HIDDEN")

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH, HIDDEN // BLOCK_C, 1),
        _token_partials_and_side_kernel,
        (
            arg0_2d, arg1_1, arg2_2d, arg3_2d, arg4_1, arg5_1,
            side_f32.view(rows, HIDDEN), side_bf16,
            partial0, partial1, partial3,
            BLOCK_C,
        ),
    )
    ct.launch(
        stream,
        (HIDDEN // BLOCK_C, 1, 1),
        _finalize_batch_partials_kernel,
        (partial0, partial1, partial3, out0, out1, out3, BLOCK_C, BATCH),
    )

    # Reshape outputs to match declared shapes.
    out0_v = out0.view(tuple(int(d) for d in shape3))
    out1_v = out1.view(tuple(int(d) for d in shape4))
    view_5 = side_bf16.view(tuple(int(d) for d in shape5))
    permute = view_5.permute(1, 0)
    out3_v = out3.view(tuple(int(d) for d in shape6))

    return out0_v, side_f32, out1_v, view_5, permute, out3_v
