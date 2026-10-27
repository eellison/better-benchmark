"""cuTile port of pointwise_fb4ca2518d23: NFNet gate/residual SiLU (channels-last).

Two outputs: (mul2 + residual) in bf16, and silu(add) * scale in bf16.
gate is [N,C,1,1] broadcasting per-channel across the NHWC image.
Layout is NHWC — treat memory as flat [N, H*W, C] with C innermost.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


FINAL_SCALE = 0.9622504486493761


@ct.kernel
def _nfnet_gate_silu_kernel(
    gate_ptr,      # bf16 [N, C]
    x_ptr,         # bf16 [N, HW, C]  (channels-last memory view)
    residual_ptr,  # bf16 [N, HW, C]
    add_out_ptr,   # bf16 [N, HW, C]
    silu_out_ptr,  # bf16 [N, HW, C]
    C: ct.Constant[int],
    FINAL_SCALE_: ct.Constant[float],
    BLOCK_C: ct.Constant[int],
):
    n = ct.bid(0)
    hw = ct.bid(1)
    c_block = ct.bid(2)

    gate = ct.load(gate_ptr, index=(n, c_block), shape=(1, BLOCK_C))
    x = ct.load(x_ptr, index=(n, hw, c_block), shape=(1, 1, BLOCK_C))
    residual = ct.load(residual_ptr, index=(n, hw, c_block), shape=(1, 1, BLOCK_C))

    gate_f = ct.astype(gate, ct.float32)
    # bf16 sigmoid: cast to fp32, sigmoid, cast back
    gate_sigmoid_bf16 = ct.astype(1.0 / (1.0 + ct.exp(-gate_f)), ct.bfloat16)

    x_f = ct.astype(x, ct.float32)
    residual_f = ct.astype(residual, ct.float32)

    # Reshape gate [1, C] to [1, 1, C]
    gate_sigmoid_bf16_3d = ct.reshape(gate_sigmoid_bf16, (1, 1, BLOCK_C))
    gate_sig_f = ct.astype(gate_sigmoid_bf16_3d, ct.float32)

    mul0 = ct.astype(x_f * gate_sig_f, ct.bfloat16)
    mul1 = ct.astype(ct.astype(mul0, ct.float32) * 2.0, ct.bfloat16)
    mul2 = ct.astype(ct.astype(mul1, ct.float32) * 0.2, ct.bfloat16)
    add_v = ct.astype(ct.astype(mul2, ct.float32) + residual_f, ct.bfloat16)
    ct.store(add_out_ptr, index=(n, hw, c_block), tile=add_v)

    add_f = ct.astype(add_v, ct.float32)
    silu = add_f / (ct.exp(-add_f) + 1.0)
    silu_bf16 = ct.astype(silu, ct.bfloat16)
    out = ct.astype(silu_bf16, ct.float32) * FINAL_SCALE_
    ct.store(silu_out_ptr, index=(n, hw, c_block), tile=ct.astype(out, ct.bfloat16))


@oracle_impl(hardware="B200", point="53dca1d5", BLOCK_HW=8, BLOCK_C=128)
@oracle_impl(hardware="B200", point="c972bcba", BLOCK_HW=8, BLOCK_C=128)
@oracle_impl(hardware="B200", point="8eccb2bf", BLOCK_HW=8, BLOCK_C=128)
@oracle_impl(hardware="B200", point="9983a35a", BLOCK_HW=8, BLOCK_C=128)
@oracle_impl(hardware="B200", point="b2d12e6c", BLOCK_HW=8, BLOCK_C=128)
@oracle_impl(hardware="B200", point="a4a4052f", BLOCK_HW=8, BLOCK_C=128)
def oracle_forward(inputs, *, BLOCK_HW: int, BLOCK_C: int):
    gate, x, residual = inputs
    n, c, h, w = (int(dim) for dim in x.shape)
    HW = h * w
    # x is NHWC-in-memory: physical order (N, H, W, C). Its stride is (HWC, 1, WC, C).
    # We view it as [N, HW, C] with strides (HWC, C, 1).
    x_view = torch.as_strided(x, (n, HW, c), (HW * c, c, 1))
    residual_view = torch.as_strided(residual, (n, HW, c), (HW * c, c, 1))

    add_out = torch.empty_strided((n, c, h, w), x.stride(),
                                  device=x.device, dtype=torch.bfloat16)
    silu_out = torch.empty_strided((n, c, h, w), x.stride(),
                                   device=x.device, dtype=torch.bfloat16)
    add_view = torch.as_strided(add_out, (n, HW, c), (HW * c, c, 1))
    silu_view = torch.as_strided(silu_out, (n, HW, c), (HW * c, c, 1))

    # Gate is [N, C, 1, 1]. View as [N, C].
    gate_2d = gate.view(n, c)

    stream = torch.cuda.current_stream()
    # grid = (N, HW, C/BLOCK_C). Choose BLOCK_C such that C % BLOCK_C == 0.
    # For C in {1536, 512}, BLOCK_C = 128 divides both.
    ct.launch(
        stream,
        (n, HW, c // BLOCK_C),
        _nfnet_gate_silu_kernel,
        (gate_2d, x_view, residual_view, add_view, silu_view, c, FINAL_SCALE, BLOCK_C),
    )
    return add_out, silu_out
