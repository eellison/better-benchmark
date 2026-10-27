"""cuTile port of pointwise_54f7ee896ad5: NFNet singleton-gate residual.

For each element: out = ((x * sigmoid(gate[b,c])) * 2 * 0.2 rounded to bf16) + residual.
Per-channel gate. Input/output share channels-last strides — permute(0,2,3,1) is
already NHWC-contiguous, so we operate on the flat NHWC buffer directly. This
mirrors Triton's USE_FLAT path (one kernel over N*H*W*C elements).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _nfnet_gate_flat_kernel(
    gate_ptr,       # bf16 [B, C] — laid out (B*C) with C-major
    x_ptr,          # bf16 [N] — NHWC flat, N = B*H*W*C
    residual_ptr,   # bf16 [N]
    out_ptr,        # bf16 [N]
    C: ct.Constant[int],
    HW: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    # Element positions inside the tile.
    offsets = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    c_off = offsets % C          # column (channel) per element
    n_off = offsets // (C * HW)  # batch per element

    x = ct.astype(ct.load(x_ptr, index=(pid,), shape=(BLOCK,)), ct.float32)
    residual = ct.astype(ct.load(residual_ptr, index=(pid,), shape=(BLOCK,)), ct.float32)

    # Gate: gather from [B, C] as (n_off * C + c_off). Use a full load of the
    # gate table then index — cuTile's load_advanced_indexing supports this.
    gate_idx = n_off * C + c_off
    gate = ct.astype(ct.gather(gate_ptr, gate_idx), ct.float32)
    gate_sig = 1.0 / (1.0 + ct.exp(-gate))
    gate_bf16 = ct.astype(gate_sig, ct.bfloat16)
    gate_f = ct.astype(gate_bf16, ct.float32)

    mul0 = ct.astype(x * gate_f, ct.bfloat16)
    mul1 = ct.astype(ct.astype(mul0, ct.float32) * 2.0, ct.bfloat16)
    mul2 = ct.astype(ct.astype(mul1, ct.float32) * 0.2, ct.bfloat16)
    out = ct.astype(ct.astype(mul2, ct.float32) + residual, ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=out)


def _largest_pow2_divisor(n, cap=1024):
    b = cap
    while b > 1 and (n % b) != 0:
        b //= 2
    return b


def _launch(inputs):
    gate, x, residual = inputs
    n, c, h, w = (int(dim) for dim in x.shape)
    hw = h * w
    total = n * c * hw

    # Channels-last: permute(0,2,3,1) is metadata-only NHWC (H, W, C innermost).
    x_nhwc = x.permute(0, 2, 3, 1).reshape(total)
    resid_nhwc = residual.permute(0, 2, 3, 1).reshape(total)
    gate_flat = gate.reshape(n * c)  # [B*C] with C innermost

    # Output in channels-last, same strides as x.
    out = torch.empty_strided(
        (n, c, h, w),
        tuple(int(s) for s in x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    out_nhwc = out.permute(0, 2, 3, 1).reshape(total)

    BLOCK = _largest_pow2_divisor(total, cap=1024)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (total // BLOCK, 1, 1),
        _nfnet_gate_flat_kernel,
        (gate_flat, x_nhwc, resid_nhwc, out_nhwc, c, hw, BLOCK),
    )
    return out


@oracle_impl(hardware="B200", point="53dca1d5", USE_FLAT=True, BLOCK=1024, BLOCK_HW=8, BLOCK_C=128)
@oracle_impl(hardware="B200", point="9983a35a", USE_FLAT=False, BLOCK=1024, BLOCK_HW=8, BLOCK_C=128)
def oracle_forward(inputs, *, USE_FLAT: bool, BLOCK: int, BLOCK_HW: int, BLOCK_C: int):
    return _launch(inputs)
