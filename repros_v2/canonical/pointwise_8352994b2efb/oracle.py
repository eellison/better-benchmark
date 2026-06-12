"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the complete Longformer bf16 bias add, head/batch layout rewrite, overlapping three-window `as_strided` stencil, transpose, and final contiguous `[288,64,512]` clone into one Triton materialization kernel; Inductor currently treats the bf16 pointwise producer and overlapping layout clone as generic separate scheduling work around a pure affine output map; the fix is SCHEDULER_FUSION: sink bf16 affine producers through fixed-overlap layout clone codegen and write the requested transposed backing storage directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _longformer_transposed_stencil_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    dim = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    chunk = tl.program_id(2)

    head_batch = chunk // 3
    window = chunk - head_batch * 3
    head = head_batch - (head_batch // 12) * 12

    source_token = window * 256 + token
    source_feature = head * 64 + dim
    load_offsets = source_token[:, None] * (8 * 768) + head_batch * 64 + dim[None, :]

    values = tl.load(x_ptr + load_offsets)
    bias = tl.load(bias_ptr + source_feature)
    out = (values + bias[None, :]).to(tl.bfloat16, fp_downcast_rounding="rtne")

    store_offsets = chunk * 64 * 512 + dim[:, None] * 512 + token[None, :]
    tl.store(out_ptr + store_offsets, tl.trans(out))


@oracle_impl(hardware="B200", point="5fa3702b", BLOCK_T=64, BLOCK_D=64, num_warps=4)
def oracle_forward(inputs, *, BLOCK_T, BLOCK_D, num_warps):
    x, bias, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6 = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5

    out = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_6),
        (64 * 512, 512, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    _longformer_transposed_stencil_kernel[
        (triton.cdiv(512, BLOCK_T), triton.cdiv(64, BLOCK_D), 288)
    ](
        x,
        bias,
        out,
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
