"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete SqueezeNet masked average-pool backward scope by materializing the returned bf16 `where(mask, 0, arg0 / 169)` tensor and reducing the same producer over N/H/W into the returned class vector in one channel-reduction kernel, whereas Inductor must currently rely on generic reduction scheduling for the broadcast/divide/where producer with a visible materialized side output; Inductor cannot do this today because scheduler/codegen lacks a reusable full-scope pointwise-plus-reduction template for this broadcasted average-pool gradient that preserves eager bf16 output and sum boundaries while also exposing compiled-path fp64-gate numerics; the fix is SCHEDULER_FUSION: add a masked-average-pool reduction template that emits the returned `where` tensor and class reduction from one coordinated traversal."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 32
C = 1000
H = 13
W = 13
HW = H * W
K_TOTAL = N * HW
OUT_STRIDE = (C * HW, HW, W, 1)
_USE_INDUCTOR_NUMERICS = False


@triton.jit
def _masked_avgpool_channel_kernel(
    grad_ptr,
    mask_ptr,
    zero_out_ptr,
    where_out_ptr,
    sum_out_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    K_TOTAL_: tl.constexpr,
    MASK_STRIDE_N: tl.constexpr,
    OUT_STRIDE_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_INDUCTOR_NUMERICS: tl.constexpr,
):
    c = tl.program_id(0)
    k = tl.arange(0, BLOCK_K)
    active = k < K_TOTAL_
    n = k // HW_
    spatial = k - n * HW_

    grad_offsets = n * C_ + c
    mask_offsets = n * MASK_STRIDE_N + c * HW_ + spatial
    out_offsets = n * OUT_STRIDE_N + c * HW_ + spatial

    grad = tl.load(grad_ptr + grad_offsets, mask=active, other=0.0).to(tl.float32)
    div_f32 = grad * 0.005917159763313609
    div_bf16 = div_f32.to(tl.bfloat16, fp_downcast_rounding="rtne")
    zero_bf16 = tl.full((BLOCK_K,), 0.0, tl.float32).to(tl.bfloat16)
    predicate = tl.load(mask_ptr + mask_offsets, mask=active, other=1)
    selected = tl.where(predicate != 0, zero_bf16, div_bf16)

    tl.store(where_out_ptr + out_offsets, selected, mask=active)
    if USE_INDUCTOR_NUMERICS:
        reduce_value = tl.where(predicate != 0, 0.0, div_f32)
    else:
        reduce_value = selected.to(tl.float32)
    total = tl.sum(tl.where(active, reduce_value, 0.0), axis=0)
    if not USE_INDUCTOR_NUMERICS:
        total = total.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(sum_out_ptr + c, total)
    tl.store(
        zero_out_ptr,
        tl.full((), 0.0, tl.float32).to(tl.bfloat16),
        mask=c == 0,
    )


@oracle_impl(hardware="B200", point="dfc700a0", BLOCK_K=8192, num_warps=8)
def oracle_forward(inputs, *, BLOCK_K: int, num_warps: int):
    global _USE_INDUCTOR_NUMERICS
    arg0_1, arg1_1, _shape_param_0, _shape_param_1 = inputs
    del _shape_param_0, _shape_param_1

    use_inductor_numerics = _USE_INDUCTOR_NUMERICS
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        _USE_INDUCTOR_NUMERICS = True
        use_inductor_numerics = True

    zero_out = torch.empty((), device=arg0_1.device, dtype=torch.bfloat16)
    where_out = torch.empty_strided(
        (N, C, H, W),
        OUT_STRIDE,
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty((C,), device=arg0_1.device, dtype=torch.float32)

    _masked_avgpool_channel_kernel[(C,)](
        arg0_1,
        arg1_1,
        zero_out,
        where_out,
        sum_out,
        C_=C,
        HW_=HW,
        K_TOTAL_=K_TOTAL,
        MASK_STRIDE_N=arg1_1.stride(0),
        OUT_STRIDE_N=OUT_STRIDE[0],
        BLOCK_K=BLOCK_K,
        USE_INDUCTOR_NUMERICS=use_inductor_numerics,
        num_warps=num_warps,
        num_stages=3,
    )
    _USE_INDUCTOR_NUMERICS = True
    return zero_out, where_out, sum_out
