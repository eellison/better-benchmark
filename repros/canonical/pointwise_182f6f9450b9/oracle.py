"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 squeeze-excitation sigmoid/multiply scope with channels-last storage-linear Triton kernels that preserve Inductor's fused `tl.sigmoid(gate) * activation` arithmetic and the captured dense output stride, whereas Inductor lowers most shapes through a generic flat expanded `[B,C,H,W]` broadcast schedule; Inductor cannot do this today because its pointwise scheduler/codegen does not specialize fused broadcast producers for this dense channel-gate layout across the shape family; the fix is SCHEDULER_FUSION: teach fused pointwise scheduling to emit layout-specialized broadcast indexing for dense squeeze-excitation gates while retaining the generic flat path where it is already at floor."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 8, "BLOCK_C": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_HW": 8, "BLOCK_C": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_HW": 8, "BLOCK_C": 256}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_HW": 16, "BLOCK_C": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_HW": 16, "BLOCK_C": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_HW": 32, "BLOCK_C": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_HW": 32, "BLOCK_C": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_HW": 64, "BLOCK_C": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_HW": 64, "BLOCK_C": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_HW": 64, "BLOCK_C": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_HW": 64, "BLOCK_C": 256}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_HW": 64, "BLOCK_C": 256}, num_warps=16, num_stages=4),
    ],
    key=["CHANNELS", "SPATIAL"],
)
@triton.jit
def _broadcast_sigmoid_mul_kernel(
    gate_ptr,
    act_ptr,
    out_ptr,
    CHANNELS: tl.constexpr,
    SPATIAL: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    batch = tl.program_id(0)
    hw_block = tl.program_id(1)
    c_block = tl.program_id(2)

    hw_offsets = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)
    c_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < CHANNELS
    hw_mask = hw_offsets < SPATIAL

    gate = tl.load(
        gate_ptr + batch * CHANNELS + c_offsets,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)
    scale = tl.sigmoid(gate)

    offsets = (
        batch * CHANNELS * SPATIAL
        + hw_offsets[:, None] * CHANNELS
        + c_offsets[None, :]
    )
    mask = hw_mask[:, None] & c_mask[None, :]
    act = tl.load(act_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = (act * scale[None, :]).to(tl.bfloat16)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _flat_sigmoid_mul_kernel(
    gate_ptr,
    act_ptr,
    out_ptr,
    CHANNELS: tl.constexpr,
    SPATIAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    batch = offsets // (CHANNELS * SPATIAL)
    channel = offsets % CHANNELS
    gate = tl.load(
        gate_ptr + batch * CHANNELS + channel,
        eviction_policy="evict_last",
    ).to(tl.float32)
    scale = tl.sigmoid(gate)
    act = tl.load(act_ptr + offsets).to(tl.float32)
    tl.store(out_ptr + offsets, act * scale)


@oracle_impl(hardware="B200", point="5e1cd0cb", USE_FLAT=True, BLOCK=2048, num_warps=4)
@oracle_impl(hardware="B200", point="94a8a021")
@oracle_impl(hardware="B200", point="20c37c70")
@oracle_impl(hardware="B200", point="2ea19024")
@oracle_impl(hardware="B200", point="7ccc5d66")
@oracle_impl(hardware="B200", point="bd79fc38")
@oracle_impl(hardware="B200", point="6ff05967")
@oracle_impl(hardware="B200", point="a542f22c")
@oracle_impl(hardware="B200", point="c5606485")
@oracle_impl(hardware="B200", point="222a5534")
def oracle_forward(inputs, *, USE_FLAT: bool = False, BLOCK: int = 0, num_warps: int = 0):
    gate, activation = inputs
    batch, channels, height, width = activation.shape
    output = torch.empty_strided(
        tuple(activation.shape),
        tuple(activation.stride()),
        device=activation.device,
        dtype=torch.bfloat16,
    )
    spatial = height * width
    if USE_FLAT:
        n_elements = activation.numel()
        grid_flat = (triton.cdiv(n_elements, BLOCK),)
        _flat_sigmoid_mul_kernel[grid_flat](
            gate,
            activation,
            output,
            CHANNELS=channels,
            SPATIAL=spatial,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return output

    grid = lambda meta: (
        batch,
        triton.cdiv(spatial, meta["BLOCK_HW"]),
        triton.cdiv(channels, meta["BLOCK_C"]),
    )
    _broadcast_sigmoid_mul_kernel[grid](
        gate,
        activation,
        output,
        CHANNELS=channels,
        SPATIAL=spatial,
    )
    return output
