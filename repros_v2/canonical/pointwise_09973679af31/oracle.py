"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 `aten.glu.default(dim=1)` scope with one output-space Triton pointwise kernel that directly reads the two contiguous channel halves, evaluates the fp32 sigmoid/multiply GLU expression, and stores the final contiguous bf16 `[B, C/2, T]` result for every Demucs shape point, whereas Inductor lowers the isolated GLU op through its generic pointwise scheduler; Inductor cannot do this today because it has no B200-tuned GLU pointwise template that specializes the split-channel contiguous layout and required bf16 output rounding as one fixed idiom; the fix is NEW_PATTERN: add a guarded GLU pointwise lowering or equivalent autotuned specialization for contiguous split-channel inputs."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _glu_dim1_kernel(
    input_ptr,
    output_ptr,
    INNER: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < INNER

    input_base = batch * (INNER * 2)
    output_base = batch * INNER
    value = tl.load(input_ptr + input_base + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(input_ptr + input_base + INNER + offsets, mask=mask, other=0.0).to(tl.float32)
    sigmoid = tl.sigmoid(gate)
    out = value * sigmoid
    tl.store(output_ptr + output_base + offsets, out.to(tl.bfloat16), mask=mask)


# Demucs GLU points: T([B, 2C, T], bf16) -> T([B, C, T], bf16)
@oracle_impl(hardware="B200", point="816cc555", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="f2ecc36d", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="76f8dd2d", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="505c06f5", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="5dddb421", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="9e433626", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="4fbcf6cb", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="12f43f3f", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="55e4f843", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="e664b13e", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="39daa5a7", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="f646cd89", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="57917947", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="513d0721", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="bbd70f4d", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="499092b7", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="df113b2d", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="dd6fed99", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="aeb7570e", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="02536193", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="aa5ecfa3", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="2e94cc97", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="80d7519d", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="51987672", BLOCK_SIZE=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK_SIZE: int, num_warps: int):
    (x,) = inputs
    batch = int(x.shape[0])
    half_channels = int(x.shape[1]) // 2
    width = int(x.shape[2])
    out = torch.empty_strided(
        (batch, half_channels, width),
        (half_channels * width, width, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    inner = half_channels * width
    _glu_dim1_kernel[(batch, triton.cdiv(inner, BLOCK_SIZE))](
        x,
        out,
        INNER=inner,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=4,
    )
    return out
