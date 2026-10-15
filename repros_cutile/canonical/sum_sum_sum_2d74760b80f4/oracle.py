"""cuTile port of sum_sum_sum_2d74760b80f4: dm_nfnet SE spatial reduce.

Reference (channels-last, non-pow2 spatial: 24x24, 12x12, 6x6):
  mul_1     = bf16(bf16(arg0 * 0.9805806756909201) * 1.7015043497085571)
  sigmoid   = bf16(sigmoid(arg1))                                [N, C, 1, 1]
  mul_2     = bf16(arg2 * sigmoid)                                bf16 [N, C, H, W]
  mul_3     = bf16(mul_2 * 2.0)
  mul_4     = bf16(mul_3 * arg3)                                  (arg3: scalar)
  mul_5     = bf16(mul_4 * 0.2)
  add       = bf16(mul_5 + arg4)
  gelu_grad = gelu_deriv(f32(add))
  mul_12    = f32(mul_1) * gelu_grad
  add_3     = bf16(arg5 + bf16(mul_12))                            bf16 [N, C, H, W]
  mul_13    = bf16(add_3 * 0.2)
  mul_14    = bf16(mul_13 * mul_3)                                  bf16 [N, C, H, W]
  sum_1     = sum(mul_14 as f32, all)                              f32 scalar
  mul_15    = bf16(mul_13 * arg3)
  mul_16    = bf16(mul_15 * 2.0)                                   bf16 [N, C, H, W]
  mul_17    = bf16(mul_16 * arg2)
  sum_2     = sum(mul_17 as f32, [2, 3], keepdim, dtype=f32)      f32 [N, C, 1, 1]
  sum_2_bf16= bf16(sum_2)
  sigmoid_grad = sigmoid * (1 - sigmoid)
  gate      = bf16(sum_2_bf16 * sigmoid_grad)                     bf16 [N, C, 1, 1]
  sum_3     = sum(f32(gate), [0, 2, 3])                            f32 [C]
  sum_4     = sum(f32(add_3), [0, 2, 3])                           f32 [C]
  Returns (sigmoid, add_3, sum_1, mul_16, gate, sum_3, sum_4).

Port plan: torch handles the elementwise producer (GELU is easily done in
torch on bf16 inputs cast to f32). One cuTile kernel does the SE per-(n,c)
spatial reduction (sum_2 -> gate) — that's the substantive work: reducing
[N, C, H, W] to [N, C, 1, 1] with a bf16 rounding boundary. torch does the
final channel/scalar reductions and the sigmoid.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


def _make_reduce_kernel(BLOCK_HW, BLOCK_C, HW):

    @ct.kernel
    def _reduce_nc(
        mul16_ptr,     # bf16 [N, HW, C]  (NHWC-flat: N * HW rows of C)
        a2_ptr,        # bf16 [N, HW, C]
        add3_ptr,      # bf16 [N, HW, C]
        gate_ptr,      # bf16 [N, C]
        sum2_ptr,      # f32  [N, C]  (kept for optional consumers)
        add3_sum_ptr,  # f32  [N, C]  partial for sum_4
        scalar_part_ptr,  # f32 [N, C_TILES]  partial for sum_1
        sigmoid_ptr,   # bf16 [N, C]
        C_: ct.Constant[int],
    ):
        # Grid: (N, cdiv(C, BLOCK_C))
        n = ct.bid(0)
        c_tile = ct.bid(1)

        cols = ct.arange(BLOCK_C, dtype=ct.int32) + c_tile * BLOCK_C
        col_mask = cols < C_

        # Load sigmoid slice (per-(n, c))
        sig = ct.load(sigmoid_ptr, index=(n, c_tile), shape=(1, BLOCK_C),
                      padding_mode=ct.PaddingMode.ZERO)

        acc_sum2 = ct.zeros((BLOCK_C,), dtype=ct.float32)
        acc_add3 = ct.zeros((BLOCK_C,), dtype=ct.float32)
        acc_scalar_local = ct.zeros((BLOCK_C,), dtype=ct.float32)

        num_hw_tiles = ct.cdiv(HW, BLOCK_HW)
        # Static loop
        for hw_tile in ct.static_iter(range(num_hw_tiles)):
            mul16 = ct.load(mul16_ptr, index=(n, hw_tile, c_tile),
                             shape=(1, BLOCK_HW, BLOCK_C),
                             padding_mode=ct.PaddingMode.ZERO)
            a2 = ct.load(a2_ptr, index=(n, hw_tile, c_tile),
                           shape=(1, BLOCK_HW, BLOCK_C),
                           padding_mode=ct.PaddingMode.ZERO)
            add3 = ct.load(add3_ptr, index=(n, hw_tile, c_tile),
                            shape=(1, BLOCK_HW, BLOCK_C),
                            padding_mode=ct.PaddingMode.ZERO)
            hw_idx = ct.arange(BLOCK_HW, dtype=ct.int32) + hw_tile * BLOCK_HW
            hw_mask = hw_idx < HW
            mask = ct.reshape(hw_mask, (1, BLOCK_HW, 1)) & \
                   ct.reshape(col_mask, (1, 1, BLOCK_C))

            mul16_f = ct.astype(mul16, ct.float32)
            a2_f = ct.astype(a2, ct.float32)
            add3_f = ct.astype(add3, ct.float32)
            # mul_17 = bf16(mul_16 * a2). But we've already computed mul_16
            # in torch; here we just reduce mul_17 = bf16(mul_16 * a2).
            mul17 = ct.astype(mul16_f * a2_f, ct.bfloat16)
            mul17_f = ct.astype(mul17, ct.float32)
            valid = mul17_f * ct.astype(mask, ct.float32)
            acc_sum2 = acc_sum2 + ct.sum(valid, axis=(0, 1))

            # add_3 partial along spatial (dim=2,3): sum_4 needs it per-(N,C).
            add3_valid = add3_f * ct.astype(mask, ct.float32)
            acc_add3 = acc_add3 + ct.sum(add3_valid, axis=(0, 1))

            # sum_1 partial: sum(mul_14, all), where mul_14 = bf16(mul_13 * mul_3).
            # We don't have mul_13/mul_3 here; torch has already computed mul_14.
            # Skip in this kernel.
            _ = acc_scalar_local  # unused (torch will compute sum_1)

        # sum2_bf = bf16(acc_sum2)
        sum2_bf = ct.astype(acc_sum2, ct.bfloat16)
        sig_f = ct.astype(sig, ct.float32)
        # sig is (1, BLOCK_C); reshape acc_sum2 to (1, BLOCK_C)
        sum2_bf_2d = ct.reshape(sum2_bf, (1, BLOCK_C))
        sum2_bf_f = ct.astype(sum2_bf_2d, ct.float32)
        sig_grad = sig_f * (1.0 - sig_f)
        gate_f = sum2_bf_f * sig_grad
        gate_bf = ct.astype(gate_f, ct.bfloat16)
        ct.store(gate_ptr, index=(n, c_tile), tile=gate_bf)
        ct.store(sum2_ptr, index=(n, c_tile),
                 tile=ct.reshape(acc_sum2, (1, BLOCK_C)))
        ct.store(add3_sum_ptr, index=(n, c_tile),
                 tile=ct.reshape(acc_add3, (1, BLOCK_C)))
        ct.store(scalar_part_ptr, index=(n, c_tile),
                 tile=ct.reshape(acc_scalar_local, (1, BLOCK_C)))

    return _reduce_nc


# Preinstantiate for each shape.
_K_512_24 = _make_reduce_kernel(BLOCK_HW=64, BLOCK_C=32, HW=576)
_K_1536_12 = _make_reduce_kernel(BLOCK_HW=16, BLOCK_C=32, HW=144)
_K_1536_6 = _make_reduce_kernel(BLOCK_HW=4, BLOCK_C=32, HW=36)


def _run(inputs, *, C, H, W, BLOCK_C, kernel):
    arg0, arg1, arg2, arg3, arg4, arg5 = inputs
    device = arg0.device
    N = int(arg0.shape[0])
    HW = H * W

    # Torch handles the elementwise producer.
    # mul_1 = bf16(bf16(a0 * 0.98...) * 1.70...)
    a0_f = arg0.to(torch.float32)
    mul_pre = (a0_f * 0.9805806756909201).to(torch.bfloat16)
    mul_1 = (mul_pre.to(torch.float32) * 1.7015043497085571).to(torch.bfloat16)

    sigmoid = torch.sigmoid(arg1.to(torch.float32)).to(torch.bfloat16)  # [N, C, 1, 1]
    mul_2 = (arg2 * sigmoid).to(torch.bfloat16)   # broadcast
    mul_3 = (mul_2.to(torch.float32) * 2.0).to(torch.bfloat16)
    scalar_bf = arg3.to(torch.float32).to(torch.bfloat16)
    mul_4 = (mul_3.to(torch.float32) * scalar_bf.to(torch.float32)).to(torch.bfloat16)
    mul_5 = (mul_4.to(torch.float32) * 0.2).to(torch.bfloat16)
    add = (mul_5.to(torch.float32) + arg4.to(torch.float32)).to(torch.bfloat16)

    add_f = add.to(torch.float32)
    # gelu = 0.5 * (1 + erf(x/sqrt(2))) + x * (1/sqrt(2*pi)) * exp(-x^2/2)
    gelu_grad = 0.5 * (torch.erf(add_f * 0.7071067811865476) + 1.0) + \
                add_f * 0.3989422804014327 * torch.exp(add_f * add_f * -0.5)
    mul_12 = mul_1.to(torch.float32) * gelu_grad
    convert_2 = mul_12.to(torch.bfloat16)
    add_3 = (arg5.to(torch.float32) + convert_2.to(torch.float32)).to(torch.bfloat16)

    mul_13 = (add_3.to(torch.float32) * 0.2).to(torch.bfloat16)
    mul_14 = (mul_13.to(torch.float32) * mul_3.to(torch.float32)).to(torch.bfloat16)
    sum_1 = mul_14.to(torch.float32).sum()

    mul_15 = (mul_13.to(torch.float32) * scalar_bf.to(torch.float32)).to(torch.bfloat16)
    mul_16 = (mul_15.to(torch.float32) * 2.0).to(torch.bfloat16)

    # cuTile kernel: reduce mul_17 = bf16(mul_16 * arg2) over [2, 3] per (n, c),
    # then combine with sigmoid_grad to produce gate. Also reduce add_3 over
    # spatial (partial for sum_4). Sum_3 is finalized in torch.

    # Prepare NHWC contig views for cuTile.
    mul16_nhwc = mul_16.permute(0, 2, 3, 1).contiguous()   # [N, H, W, C]
    a2_nhwc = arg2.permute(0, 2, 3, 1).contiguous()         # [N, H, W, C]
    add3_nhwc = add_3.permute(0, 2, 3, 1).contiguous()      # [N, H, W, C]
    sigmoid_2d = sigmoid.view(N, C)                          # bf16

    # Reshape to (N, HW, C) contiguous.
    mul16_flat = mul16_nhwc.view(N, HW, C)
    a2_flat = a2_nhwc.view(N, HW, C)
    add3_flat = add3_nhwc.view(N, HW, C)

    # Kernel outputs.
    C_TILES = (C + BLOCK_C - 1) // BLOCK_C
    gate_2d = torch.empty((N, C), device=device, dtype=torch.bfloat16)
    sum2_partial = torch.empty((N, C), device=device, dtype=torch.float32)
    add3_partial = torch.empty((N, C), device=device, dtype=torch.float32)
    scalar_partial = torch.empty((N, C), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (N, C_TILES, 1), kernel,
        (mul16_flat, a2_flat, add3_flat,
         gate_2d, sum2_partial, add3_partial, scalar_partial,
         sigmoid_2d, C),
    )

    # gate_4d = bf16 [N, C, 1, 1]
    gate_bf16 = gate_2d.view(N, C, 1, 1).contiguous()

    # sum_3 = sum(gate as f32, [0, 2, 3]) = sum along N.
    sum_3 = gate_bf16.to(torch.float32).sum(dim=[0, 2, 3])

    # sum_4 = sum(add_3 as f32, [0, 2, 3]) — sum across N then already-per-(n,c)
    # sum. add3_partial is per-(n, c) spatial sum.
    sum_4 = add3_partial.sum(dim=0)

    return sigmoid, add_3, sum_1, mul_16, gate_bf16, sum_3, sum_4


@oracle_impl(hardware="B200", point="adbc9760")
def oracle_forward_adbc(inputs):
    return _run(inputs, C=512, H=24, W=24, BLOCK_C=32, kernel=_K_512_24)


@oracle_impl(hardware="B200", point="bfd27ee1")
def oracle_forward_bfd(inputs):
    return _run(inputs, C=1536, H=12, W=12, BLOCK_C=32, kernel=_K_1536_12)


@oracle_impl(hardware="B200", point="9d70a1e6")
def oracle_forward(inputs):
    return _run(inputs, C=1536, H=6, W=6, BLOCK_C=32, kernel=_K_1536_6)
