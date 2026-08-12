"""cuTile port of sum_sum_sum_51593d0552e5: MobileBERT multi-output LN
backward + column reductions.

Two cuTile kernels mirroring the Triton reference:
  1) _partials_and_sides_kernel: elementwise chain + per-batch partial sums
     over TOKENS + bf16 side outputs.
  2) _finalize_kernel: sum partials over BATCH; round outs 4 & 9 through bf16.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 256
TOKENS = 128
HIDDEN = 128
FINAL_BLOCK_C = 32


@ct.kernel
def _partials_and_sides_kernel(
    arg0_ptr,        # bf16 [BATCH*TOKENS, HIDDEN]
    arg1_ptr,        # f32  [BATCH*TOKENS, HIDDEN] (view of [B,T,H])
    arg2_ptr,        # bf16 [BATCH*TOKENS, HIDDEN]
    arg3_ptr,        # f32  [HIDDEN]
    arg4_ptr,        # f32  [HIDDEN]
    arg5_ptr,        # bf16 [BATCH*TOKENS, HIDDEN]
    arg6_ptr,        # f32  [HIDDEN]
    side0_ptr,       # bf16 [BATCH*TOKENS, HIDDEN]
    side1_ptr,       # bf16 [BATCH*TOKENS, HIDDEN]
    p0_ptr,          # f32  [BATCH, HIDDEN]
    p1_ptr,
    p4_ptr,
    p5_ptr,
    p6_ptr,
    p9_ptr,
    TOKENS_: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
):
    b = ct.bid(0)
    arg0 = ct.astype(ct.load(arg0_ptr, index=(b, 0),
                             shape=(TOKENS_, HIDDEN_)), ct.float32)
    arg1 = ct.load(arg1_ptr, index=(b, 0), shape=(TOKENS_, HIDDEN_))
    arg2 = ct.astype(ct.load(arg2_ptr, index=(b, 0),
                             shape=(TOKENS_, HIDDEN_)), ct.float32)
    arg5 = ct.astype(ct.load(arg5_ptr, index=(b, 0),
                             shape=(TOKENS_, HIDDEN_)), ct.float32)
    arg3_1d = ct.load(arg3_ptr, index=(0,), shape=(HIDDEN_,))
    arg4_1d = ct.load(arg4_ptr, index=(0,), shape=(HIDDEN_,))
    arg6_1d = ct.load(arg6_ptr, index=(0,), shape=(HIDDEN_,))
    arg3 = ct.reshape(arg3_1d, (1, HIDDEN_))
    arg4 = ct.reshape(arg4_1d, (1, HIDDEN_))
    arg6 = ct.reshape(arg6_1d, (1, HIDDEN_))

    add = arg1 + arg0
    add_1 = arg2 * arg3 + arg4
    add_2 = arg5 + add_1
    mul_1 = add * add_2
    mul_2 = add * arg6
    side0 = ct.astype(mul_2, ct.bfloat16)
    mul_3 = mul_2 * arg2
    mul_4 = mul_2 * arg3
    side1 = ct.astype(mul_4, ct.bfloat16)

    ct.store(side0_ptr, index=(b, 0), tile=side0)
    ct.store(side1_ptr, index=(b, 0), tile=side1)

    p0 = ct.sum(add, axis=0, keepdims=True)
    p1 = ct.sum(mul_1, axis=0, keepdims=True)
    p4 = ct.sum(ct.astype(side0, ct.float32), axis=0, keepdims=True)
    p5 = ct.sum(mul_2, axis=0, keepdims=True)
    p6 = ct.sum(mul_3, axis=0, keepdims=True)
    p9 = ct.sum(ct.astype(side1, ct.float32), axis=0, keepdims=True)

    ct.store(p0_ptr, index=(b, 0), tile=p0)
    ct.store(p1_ptr, index=(b, 0), tile=p1)
    ct.store(p4_ptr, index=(b, 0), tile=p4)
    ct.store(p5_ptr, index=(b, 0), tile=p5)
    ct.store(p6_ptr, index=(b, 0), tile=p6)
    ct.store(p9_ptr, index=(b, 0), tile=p9)


@ct.kernel
def _finalize_kernel(
    p0_ptr, p1_ptr, p4_ptr, p5_ptr, p6_ptr, p9_ptr,
    out0_ptr, out1_ptr, out4_ptr, out5_ptr, out6_ptr, out9_ptr,
    BATCH_: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_blk = ct.bid(0)
    p0 = ct.load(p0_ptr, index=(0, c_blk), shape=(BATCH_, BLOCK_C))
    p1 = ct.load(p1_ptr, index=(0, c_blk), shape=(BATCH_, BLOCK_C))
    p4 = ct.load(p4_ptr, index=(0, c_blk), shape=(BATCH_, BLOCK_C))
    p5 = ct.load(p5_ptr, index=(0, c_blk), shape=(BATCH_, BLOCK_C))
    p6 = ct.load(p6_ptr, index=(0, c_blk), shape=(BATCH_, BLOCK_C))
    p9 = ct.load(p9_ptr, index=(0, c_blk), shape=(BATCH_, BLOCK_C))

    out0 = ct.sum(p0, axis=0)
    out1 = ct.sum(p1, axis=0)
    out4 = ct.sum(p4, axis=0)
    out5 = ct.sum(p5, axis=0)
    out6 = ct.sum(p6, axis=0)
    out9 = ct.sum(p9, axis=0)

    out4_bf = ct.astype(ct.astype(out4, ct.bfloat16), ct.float32)
    out9_bf = ct.astype(ct.astype(out9, ct.bfloat16), ct.float32)

    ct.store(out0_ptr, index=(c_blk,), tile=out0)
    ct.store(out1_ptr, index=(c_blk,), tile=out1)
    ct.store(out4_ptr, index=(c_blk,), tile=out4_bf)
    ct.store(out5_ptr, index=(c_blk,), tile=out5)
    ct.store(out6_ptr, index=(c_blk,), tile=out6)
    ct.store(out9_ptr, index=(c_blk,), tile=out9_bf)


@oracle_impl(hardware="B200", point="fd972c32", XBLOCK=32)
def oracle_forward(inputs, **_kwargs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1,
     s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10) = inputs
    device = arg0_1.device

    rows = BATCH * TOKENS
    # arg0/arg2/arg5 are already (rows, HIDDEN). arg1_1 is (BATCH, TOKENS, HIDDEN)
    # — view as 2D for the kernel (metadata-only).
    arg1_2d = arg1_1.view(rows, HIDDEN)

    side0_2d = torch.empty((rows, HIDDEN), device=device, dtype=torch.bfloat16)
    side1_2d = torch.empty((rows, HIDDEN), device=device, dtype=torch.bfloat16)
    p0 = torch.empty((BATCH, HIDDEN), device=device, dtype=torch.float32)
    p1 = torch.empty((BATCH, HIDDEN), device=device, dtype=torch.float32)
    p4 = torch.empty((BATCH, HIDDEN), device=device, dtype=torch.float32)
    p5 = torch.empty((BATCH, HIDDEN), device=device, dtype=torch.float32)
    p6 = torch.empty((BATCH, HIDDEN), device=device, dtype=torch.float32)
    p9 = torch.empty((BATCH, HIDDEN), device=device, dtype=torch.float32)

    out0 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out1 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out4 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out5 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out6 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out9 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (BATCH, 1, 1),
        _partials_and_sides_kernel,
        (arg0_1, arg1_2d, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1,
         side0_2d, side1_2d, p0, p1, p4, p5, p6, p9,
         TOKENS, HIDDEN),
    )
    ct.launch(
        stream, (HIDDEN // FINAL_BLOCK_C, 1, 1),
        _finalize_kernel,
        (p0, p1, p4, p5, p6, p9,
         out0, out1, out4, out5, out6, out9,
         BATCH, FINAL_BLOCK_C),
    )

    side0 = side0_2d.view(*tuple(int(x) for x in s5))
    side1 = side1_2d.view(*tuple(int(x) for x in s9))

    view_1 = out0.view(*tuple(int(x) for x in s1))
    view_4 = out1.view(*tuple(int(x) for x in s4))
    view_5 = side0
    permute = view_5.permute(1, 0)
    ct_3 = out4.view(*tuple(int(x) for x in s6))
    view_7 = out5.view(*tuple(int(x) for x in s7))
    view_8 = out6.view(*tuple(int(x) for x in s8))
    view_9 = side1
    permute_1 = view_9.permute(1, 0)
    ct_6 = out9.view(*tuple(int(x) for x in s10))

    return (view_1, view_4, view_5, permute, ct_3, view_7, view_8, view_9,
            permute_1, ct_6)
