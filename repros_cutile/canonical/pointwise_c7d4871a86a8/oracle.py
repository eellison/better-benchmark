"""cuTile port of pointwise_c7d4871a86a8: NFNet gate+scale+residual+GELU+avg_pool2d.

Two cuTile pointwise kernels (gate+add, then GELU epilogue with in-kernel erf
via the Abramowitz-Stegun 7.1.26 polynomial), followed by a
torch.nn.functional.avg_pool2d for the pooled output. The Triton reference
uses a single kernel that fuses everything including the 2x2 pool, but that
stencil pattern is not portable to the current cuTile primitives.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


FINAL_SCALE = 0.8980265101338745


@ct.kernel
def _nfnet_gate_add_kernel(
    gate_ptr,       # bf16 [N*C] flat
    x_ptr,          # bf16 [N*H*W*C] in (n, h, w, c) order
    scalar_ptr,     # bf16/f32 [1]
    residual_ptr,   # bf16 [N*H*W*C] in (n, h, w, c) order
    add_ptr,        # bf16 [N*H*W*C] output for add_value
    N_HW_C: ct.Constant[int],
    C: ct.Constant[int],
    HW: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int64)
    mask = offsets < N_HW_C

    c_idx = offsets % C
    row_flat = offsets // C
    n = row_flat // HW
    gate_idx = n * C + c_idx

    x = ct.astype(ct.gather(x_ptr, offsets, mask=mask), ct.float32)
    residual = ct.astype(ct.gather(residual_ptr, offsets, mask=mask), ct.float32)
    gate = ct.astype(ct.gather(gate_ptr, gate_idx, mask=mask), ct.float32)
    scalar_val = ct.load(scalar_ptr, index=(0,), shape=(1,))
    scalar_bf16 = ct.astype(scalar_val, ct.bfloat16)
    scalar_f = ct.astype(scalar_bf16, ct.float32)
    scalar_broadcast = ct.full((BLOCK,), 0.0, dtype=ct.float32) + scalar_f

    gate_bf16 = ct.astype(1.0 / (1.0 + ct.exp(-gate)), ct.bfloat16)
    mul0 = ct.astype(x * ct.astype(gate_bf16, ct.float32), ct.bfloat16)
    mul1 = ct.astype(ct.astype(mul0, ct.float32) * 2.0, ct.bfloat16)
    mul2 = ct.astype(ct.astype(mul1, ct.float32) * scalar_broadcast, ct.bfloat16)
    mul3 = ct.astype(ct.astype(mul2, ct.float32) * 0.2, ct.bfloat16)
    add_value = ct.astype(ct.astype(mul3, ct.float32) + residual, ct.bfloat16)

    ct.scatter(add_ptr, offsets, add_value, mask=mask)


@ct.kernel
def _nfnet_gelu_epilogue_kernel(
    add_ptr,        # bf16 [N*H*W*C] in (n, h, w, c) order
    out_ptr,        # bf16 [N*H*W*C]
    N_HW_C: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int64)
    mask = offsets < N_HW_C

    add_v = ct.astype(ct.gather(add_ptr, offsets, mask=mask), ct.float32)
    erf_arg = add_v * 0.7071067811865476

    # In-kernel erf via Abramowitz-Stegun 7.1.26.
    zero_f = ct.zeros((BLOCK,), dtype=ct.float32)
    sign = ct.where(erf_arg >= zero_f,
                    ct.full((BLOCK,), 1.0, dtype=ct.float32),
                    ct.full((BLOCK,), -1.0, dtype=ct.float32))
    abs_arg = ct.where(erf_arg >= zero_f, erf_arg, -erf_arg)
    t = 1.0 / (1.0 + 0.3275911 * abs_arg)
    poly = (((((1.061405429 * t) - 1.453152027) * t + 1.421413741) * t
             - 0.284496736) * t + 0.254829592) * t
    erf_v = sign * (1.0 - poly * ct.exp(-abs_arg * abs_arg))

    gelu = (add_v * 0.5) * (erf_v + 1.0)
    gelu_bf16 = ct.astype(gelu, ct.bfloat16)
    scaled = ct.astype(ct.astype(gelu_bf16, ct.float32) * 1.7015043497085571, ct.bfloat16)
    scaled2 = ct.astype(ct.astype(scaled, ct.float32) * FINAL_SCALE, ct.bfloat16)
    ct.scatter(out_ptr, offsets, scaled2, mask=mask)


@oracle_impl(hardware="B200", point="27a04b02", BLOCK=1024)
@oracle_impl(hardware="B200", point="a4fc89e1", BLOCK=1024)
@oracle_impl(hardware="B200", point="945b3208", BLOCK=1024)
@oracle_impl(hardware="B200", point="1d754e93", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    gate, x, scalar, residual = inputs
    n, c, h, w = (int(dim) for dim in x.shape)

    activation_stride = tuple(int(s) for s in x.stride())
    activation = torch.empty_strided(
        (n, c, h, w),
        activation_stride,
        device=x.device,
        dtype=torch.bfloat16,
    )

    total = n * c * h * w
    # channels-last (n,c,h,w) with stride (C*H*W,1,W*C,C) => memory order (n,h,w,c)
    x_perm = x.permute(0, 2, 3, 1).contiguous().view(-1)
    residual_perm = residual.permute(0, 2, 3, 1).contiguous().view(-1)
    scalar_1d = scalar.reshape(1)
    gate_flat = gate.view(n, c).contiguous().view(-1)

    add_out = torch.empty(total, device=x.device, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(total, BLOCK), 1, 1),
        _nfnet_gate_add_kernel,
        (gate_flat, x_perm, scalar_1d, residual_perm, add_out,
         total, c, h * w, BLOCK),
    )

    out_flat = torch.empty(total, device=x.device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        (ct.cdiv(total, BLOCK), 1, 1),
        _nfnet_gelu_epilogue_kernel,
        (add_out, out_flat, total, BLOCK),
    )

    # Write out_flat back into channels-last activation
    activation.copy_(out_flat.view(n, h, w, c).permute(0, 3, 1, 2))

    pool = torch.nn.functional.avg_pool2d(activation, 2, stride=2)
    pool = pool.contiguous(memory_format=torch.channels_last)
    return activation, pool
