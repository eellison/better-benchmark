"""cuTile port of sum_sum_sum_565b9b0299d1 (COOPERATIVE_SPLIT_K): MobileBERT
multi-output reduction.

Producer: x = fp32(lhs) + fp32(rhs); side_bf16 = bf16(x * scale).
Column reductions (all [HIDDEN=128]):
  out0 = sum_rows(x)
  out1 = sum_rows(x * product_rhs)
  out2 = sum_rows(side_bf16 to f32), then bf16-round the result.

Two-stage: (1) per-row-block partial sums for out0/out1/out2 + side_bf16 store,
(2) final per-column reduction across chunks.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 32768
HIDDEN = 128
BLOCK_R = 128  # rows per program


@ct.kernel
def _producer_partials_kernel(
    lhs_ptr,          # bf16 [ROWS, HIDDEN]
    rhs_ptr,          # bf16 [ROWS, HIDDEN]
    product_rhs_ptr,  # bf16 [ROWS, HIDDEN]
    scale_ptr,        # f32 [HIDDEN]
    side_bf16_ptr,    # bf16 [ROWS, HIDDEN]
    partial0_ptr,     # f32 [NUM_CHUNKS, HIDDEN]  sum(x)
    partial1_ptr,     # f32 [NUM_CHUNKS, HIDDEN]  sum(x * product_rhs)
    partial2_ptr,     # f32 [NUM_CHUNKS, HIDDEN]  sum(f32(side_bf16))
    HIDDEN_C: ct.Constant[int],
    BLOCK_R_C: ct.Constant[int],
):
    chunk = ct.bid(0)
    lhs = ct.astype(
        ct.load(lhs_ptr, index=(chunk, 0), shape=(BLOCK_R_C, HIDDEN_C)),
        ct.float32,
    )
    rhs = ct.astype(
        ct.load(rhs_ptr, index=(chunk, 0), shape=(BLOCK_R_C, HIDDEN_C)),
        ct.float32,
    )
    product_rhs = ct.astype(
        ct.load(product_rhs_ptr, index=(chunk, 0), shape=(BLOCK_R_C, HIDDEN_C)),
        ct.float32,
    )
    scale = ct.astype(
        ct.load(scale_ptr, index=(0,), shape=(HIDDEN_C,)),
        ct.float32,
    )
    scale_2d = ct.reshape(scale, (1, HIDDEN_C))

    x = lhs + rhs
    side_bf16 = ct.astype(x * scale_2d, ct.bfloat16)
    ct.store(side_bf16_ptr, index=(chunk, 0), tile=side_bf16)

    partial0 = ct.sum(x, axis=0)
    partial1 = ct.sum(x * product_rhs, axis=0)
    partial2 = ct.sum(ct.astype(side_bf16, ct.float32), axis=0)
    ct.store(partial0_ptr, index=(chunk, 0), tile=ct.reshape(partial0, (1, HIDDEN_C)))
    ct.store(partial1_ptr, index=(chunk, 0), tile=ct.reshape(partial1, (1, HIDDEN_C)))
    ct.store(partial2_ptr, index=(chunk, 0), tile=ct.reshape(partial2, (1, HIDDEN_C)))


@ct.kernel
def _finalize_partials_kernel(
    partial0_ptr,     # f32 [NUM_CHUNKS, HIDDEN]
    partial1_ptr,     # f32 [NUM_CHUNKS, HIDDEN]
    partial2_ptr,     # f32 [NUM_CHUNKS, HIDDEN]
    out0_ptr,         # f32 [HIDDEN]
    out1_ptr,         # f32 [HIDDEN]
    out2_ptr,         # f32 [HIDDEN]
    HIDDEN_C: ct.Constant[int],
    NUM_CHUNKS: ct.Constant[int],
):
    # One program covers the entire (NUM_CHUNKS, HIDDEN) partials.
    p0 = ct.load(partial0_ptr, index=(0, 0), shape=(NUM_CHUNKS, HIDDEN_C))
    p1 = ct.load(partial1_ptr, index=(0, 0), shape=(NUM_CHUNKS, HIDDEN_C))
    p2 = ct.load(partial2_ptr, index=(0, 0), shape=(NUM_CHUNKS, HIDDEN_C))
    sum0 = ct.sum(p0, axis=0)
    sum1 = ct.sum(p1, axis=0)
    sum2 = ct.sum(p2, axis=0)
    sum2_rounded = ct.astype(ct.astype(sum2, ct.bfloat16), ct.float32)
    ct.store(out0_ptr, index=(0,), tile=sum0)
    ct.store(out1_ptr, index=(0,), tile=sum1)
    ct.store(out2_ptr, index=(0,), tile=sum2_rounded)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="02a109be", xblock=32)
def oracle_forward(inputs, *, xblock: int):
    (
        lhs, rhs, product_rhs, scale,
        _shape0, _shape1, sum0_shape_param, _shape3,
        sum1_shape_param, flat_shape_param, sum2_shape_param,
    ) = inputs
    del xblock
    device = lhs.device
    flat_shape = _shape_tuple(flat_shape_param)
    sum0_shape = _shape_tuple(sum0_shape_param)
    sum1_shape = _shape_tuple(sum1_shape_param)
    sum2_shape = _shape_tuple(sum2_shape_param)

    num_chunks = ROWS // BLOCK_R
    side_bf16 = torch.empty_strided(
        flat_shape, (flat_shape[1], 1), device=device, dtype=torch.bfloat16
    )
    partial0 = torch.empty((num_chunks, HIDDEN), device=device, dtype=torch.float32)
    partial1 = torch.empty((num_chunks, HIDDEN), device=device, dtype=torch.float32)
    partial2 = torch.empty((num_chunks, HIDDEN), device=device, dtype=torch.float32)
    out0 = torch.empty_strided(sum0_shape, (1,), device=device, dtype=torch.float32)
    out1 = torch.empty_strided(sum1_shape, (1,), device=device, dtype=torch.float32)
    out2 = torch.empty_strided(sum2_shape, (1,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_chunks, 1, 1),
        _producer_partials_kernel,
        (lhs, rhs, product_rhs, scale, side_bf16, partial0, partial1, partial2,
         HIDDEN, BLOCK_R),
    )
    ct.launch(
        stream,
        (1, 1, 1),
        _finalize_partials_kernel,
        (partial0, partial1, partial2, out0, out1, out2,
         HIDDEN, num_chunks),
    )
    return out0, out1, side_bf16, side_bf16.permute(1, 0), out2
