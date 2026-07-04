"""cuTile port of pointwise_182f6f9450b9: EfficientNet squeeze-excitation
sigmoid(gate) * activation over channels-last strided activations.

Non-power-of-2 channel counts are handled by viewing activation as an NHWC
tensor (128 * H * W, C) where the outer dim is a nice power-friendly length
and channels get padded within the kernel via ZERO padding + scatter store.

Simplest workable approach: gate is [N, C] contiguous. Activation storage is
NHWC in memory, i.e. logical [N, H, W, C] contiguous. Since the operation is
`act * sigmoid(gate).broadcast`, treat as flat [N*H*W, C] with per-row scalar
gate that only depends on batch. We spawn one program per (batch, hw_block)
and pad the channel dim within the tile.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _se_sigmoid_mul_kernel(
    gate_arr,   # bf16 [N, C] (gate.view(N, C))
    act_arr,    # bf16 [N, H*W, C]  (channels-last-viewed activation)
    out_arr,    # bf16 [N, H*W, C]
    C: ct.Constant[int],
    HW: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    n = ct.bid(0)
    hw_block = ct.bid(1)

    # Load gate row (padded with zero on the tail)
    gate = ct.load(
        gate_arr, index=(n, 0), shape=(1, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )  # bf16 [1, BLOCK_C]
    gate_f = ct.astype(gate, ct.float32)
    scale = 1.0 / (1.0 + ct.exp(-gate_f))  # sigmoid(gate)

    # Load activation tile [1, BLOCK_HW, BLOCK_C], padded with zero
    act = ct.load(
        act_arr, index=(n, hw_block, 0), shape=(1, BLOCK_HW, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    act_f = ct.astype(act, ct.float32)
    scale_3d = ct.reshape(scale, (1, 1, BLOCK_C))
    out_f = act_f * scale_3d
    out = ct.astype(out_f, ct.bfloat16)

    # Guard OOB in both HW and C using masked-scatter is expensive; simpler:
    # only store when both (hw < HW) and (c < C). We use scatter with mask.
    hw_idx = ct.arange(BLOCK_HW, dtype=ct.int32) + hw_block * BLOCK_HW
    c_idx = ct.arange(BLOCK_C, dtype=ct.int32)
    hw_mask = hw_idx < HW
    c_mask = c_idx < C
    hw_mask_3d = ct.reshape(hw_mask, (1, BLOCK_HW, 1))
    c_mask_3d = ct.reshape(c_mask, (1, 1, BLOCK_C))
    valid = hw_mask_3d & c_mask_3d
    n_idx = ct.full(shape=(1, BLOCK_HW, BLOCK_C), fill_value=n, dtype=ct.int32)
    hw_bc = ct.broadcast_to(ct.reshape(hw_idx, (1, BLOCK_HW, 1)), (1, BLOCK_HW, BLOCK_C))
    c_bc = ct.broadcast_to(ct.reshape(c_idx, (1, 1, BLOCK_C)), (1, BLOCK_HW, BLOCK_C))
    ct.scatter(out_arr, (n_idx, hw_bc, c_bc), out, mask=valid)


def _run(gate, activation, *, BLOCK_HW: int, BLOCK_C: int):
    batch, channels, height, width = activation.shape
    hw = height * width
    output = torch.empty_strided(
        tuple(activation.shape),
        tuple(activation.stride()),
        device=activation.device,
        dtype=torch.bfloat16,
    )
    # Activation storage layout is NHWC (channels-last), so we can view it as
    # (N, H*W, C) with the last dim contiguous.
    act_nhwc = activation.permute(0, 2, 3, 1).reshape(batch, hw, channels)
    out_nhwc = output.permute(0, 2, 3, 1).reshape(batch, hw, channels)
    # gate is [N, C, 1, 1] contiguous. View as [N, C].
    gate_2d = gate.view(batch, channels)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (batch, ct.cdiv(hw, BLOCK_HW), 1),
        _se_sigmoid_mul_kernel,
        (gate_2d, act_nhwc, out_nhwc, channels, hw, BLOCK_HW, BLOCK_C),
    )
    return output


@oracle_impl(hardware="B200", point="5e1cd0cb", BLOCK_HW=64, BLOCK_C=2048)
@oracle_impl(hardware="B200", point="94a8a021", BLOCK_HW=64, BLOCK_C=1024)
@oracle_impl(hardware="B200", point="20c37c70", BLOCK_HW=32, BLOCK_C=1024)
@oracle_impl(hardware="B200", point="2ea19024", BLOCK_HW=32, BLOCK_C=512)
@oracle_impl(hardware="B200", point="7ccc5d66", BLOCK_HW=32, BLOCK_C=256)
@oracle_impl(hardware="B200", point="bd79fc38", BLOCK_HW=16, BLOCK_C=256)
@oracle_impl(hardware="B200", point="6ff05967", BLOCK_HW=16, BLOCK_C=256)
@oracle_impl(hardware="B200", point="a542f22c", BLOCK_HW=16, BLOCK_C=256)
@oracle_impl(hardware="B200", point="c5606485", BLOCK_HW=16, BLOCK_C=128)
@oracle_impl(hardware="B200", point="222a5534", BLOCK_HW=16, BLOCK_C=32)
def oracle_forward(inputs, *, BLOCK_HW: int, BLOCK_C: int):
    gate, activation = inputs
    return _run(gate, activation, BLOCK_HW=BLOCK_HW, BLOCK_C=BLOCK_C)
