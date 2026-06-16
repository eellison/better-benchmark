"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 Visformer QKV layout assembly by reading the three BMM outputs and writing the final contiguous `[B, 3*H*D, side, side]` image tensor directly, whereas Inductor lowers the view/permute/cat/view/permute/clone/view chain as generic layout materialization around the same data movement; Inductor cannot do this today because its scheduler does not canonicalize this fixed QKV concat plus per-operand transpose into one direct final-layout store template for the captured bf16 shapes; the fix is SCHEDULER_FUSION: teach layout-copy scheduling to fuse the cat and transpose-only producers into one affine multi-source materialization."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.autotune(
    configs=[
        triton.Config({"YBLOCK": 8, "XBLOCK": 128}, num_warps=4, num_stages=3),
        triton.Config({"YBLOCK": 16, "XBLOCK": 64}, num_warps=4, num_stages=3),
        triton.Config({"YBLOCK": 16, "XBLOCK": 128}, num_warps=4, num_stages=3),
        triton.Config({"YBLOCK": 32, "XBLOCK": 32}, num_warps=4, num_stages=3),
        triton.Config({"YBLOCK": 32, "XBLOCK": 64}, num_warps=8, num_stages=3),
        triton.Config({"YBLOCK": 64, "XBLOCK": 32}, num_warps=8, num_stages=3),
        triton.Config({"YBLOCK": 64, "XBLOCK": 64}, num_warps=8, num_stages=3),
        triton.Config({"YBLOCK": 256, "XBLOCK": 16}, num_warps=8, num_stages=3),
        triton.Config({"YBLOCK": 16, "XBLOCK": 256}, num_warps=8, num_stages=3),
        triton.Config({"YBLOCK": 1024, "XBLOCK": 1}, num_warps=8, num_stages=3),
    ],
    key=["B", "HEADS", "D", "P"],
)
@triton.jit
def _qkv_image_layout_kernel(
    value_ptr,
    key_ptr,
    query_ptr,
    out_ptr,
    B: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,
    YNUMEL: tl.constexpr,
    YBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    y = tl.program_id(1) * YBLOCK + tl.arange(0, YBLOCK)[:, None]
    x = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[None, :]

    d = y % D
    head = (y // D) % HEADS
    batch_qkv = y // (HEADS * D)
    batch = batch_qkv % B
    qkv = batch_qkv // B
    channel = y % (HEADS * D)

    qv_offsets = d + D * x + (P * D) * head + (HEADS * P * D) * batch
    key_offsets = x + P * channel + (HEADS * P * D) * (batch_qkv - B)
    out_offsets = x + P * channel + (HEADS * D * P) * qkv + (3 * HEADS * D * P) * batch

    mask = (y < YNUMEL) & (x < P)
    value_data = tl.load(
        value_ptr + qv_offsets,
        mask=mask & (qkv == 0),
        eviction_policy="evict_last",
        other=0.0,
    ).to(tl.float32)
    key_data = tl.load(key_ptr + key_offsets, mask=mask & (qkv == 1), other=0.0).to(tl.float32)
    query_data = tl.load(
        query_ptr + qv_offsets,
        mask=mask & (qkv == 2),
        eviction_policy="evict_last",
        other=0.0,
    ).to(tl.float32)
    data = tl.where(qkv == 0, value_data, tl.where(qkv == 1, key_data, query_data))
    tl.store(out_ptr + out_offsets, data, mask=mask)


# 0fc3d1e9: (T([768,196,64], bf16), T([768,64,196], bf16), T([768,196,64], bf16), S([128,6,196,64]), S([128,6,64,196]), S([128,6,196,64]), S([3,128,6,196,64]), S([128,1152,14,14]))
@oracle_impl(hardware="B200", point="0fc3d1e9")
# 668a8297: (T([768,49,128], bf16), T([768,128,49], bf16), T([768,49,128], bf16), S([128,6,49,128]), S([128,6,128,49]), S([128,6,49,128]), S([3,128,6,49,128]), S([128,2304,7,7]))
@oracle_impl(hardware="B200", point="668a8297")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4 = inputs
    del _shape_param_1, _shape_param_2, _shape_param_3

    output_shape = tuple(int(dim) for dim in _shape_param_4)
    output = torch.empty(output_shape, device=arg0_1.device, dtype=arg0_1.dtype)

    batch = int(_shape_param_0[0])
    heads = int(_shape_param_0[1])
    positions = int(_shape_param_0[2])
    channels = int(_shape_param_0[3])
    grid = lambda meta: (
        triton.cdiv(positions, meta["XBLOCK"]),
        triton.cdiv(3 * batch * heads * channels, meta["YBLOCK"]),
    )
    _qkv_image_layout_kernel[grid](
        arg2_1,
        arg1_1,
        arg0_1,
        output,
        B=batch,
        HEADS=heads,
        D=channels,
        P=positions,
        YNUMEL=3 * batch * heads * channels,
    )
    return output
