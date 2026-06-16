"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete bf16 GPT-J residual LayerNorm scope in one fixed-hidden Triton row kernel, including both `[128,4096]` views, the two bf16 residual-add materialization boundaries before fp32 normalization, Welford `var_mean(correction=0, keepdim=True)` over hidden size 4096, eps=1e-5 `libdevice.rsqrt`, bf16 affine weight/bias epilogue, final bf16 cast, and `[128,4096]` view output, whereas Inductor reaches this normalization through the generic fused norm lowering and currently sinks the residual add chain through the reduction without a dedicated residual-LayerNorm cast-boundary specialization; Inductor cannot do this today because its norm lowering does not have a guarded fixed-width GPT-J residual LayerNorm path that both removes generic norm overhead and preserves the captured bf16 add, Welford accumulation, eps, affine-cast, and output-stride semantics; the fix is ALGEBRAIC_ELIMINATION: add a static-hidden residual LayerNorm lowering that specializes the norm algebra while preserving the captured dtype boundaries."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


ROWS = 128
HIDDEN = 4096
EPS = 1.0e-5


@triton.jit
def _bf16_round_f32(x):
    return x.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)


@triton.jit
def _gptj_bf16_layernorm_kernel(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)[None, :]
    offsets = row * 4096 + cols

    a = tl.load(x0_ptr + offsets, eviction_policy="evict_last").to(tl.float32)
    b = tl.load(x1_ptr + offsets, eviction_policy="evict_last").to(tl.float32)
    c = tl.load(x2_ptr + offsets, eviction_policy="evict_last").to(tl.float32)
    x = _bf16_round_f32(a + b)
    x = _bf16_round_f32(x + c)

    mean_acc = tl.zeros([1, BLOCK], tl.float32)
    m2_acc = tl.zeros([1, BLOCK], tl.float32)
    weight_acc = tl.zeros([1, BLOCK], tl.float32)
    mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
        x,
        mean_acc,
        m2_acc,
        weight_acc,
        True,
    )
    mean, m2, _weight = triton_helpers.welford(
        mean_next,
        m2_next,
        weight_next,
        1,
    )

    invstd = libdevice.rsqrt((m2 / 4096.0) + 1.0e-5)
    weight = tl.load(weight_ptr + cols, eviction_policy="evict_last").to(tl.float32)
    bias = tl.load(bias_ptr + cols, eviction_policy="evict_last").to(tl.float32)
    y = ((x - mean[:, None]) * invstd[:, None]) * weight + bias
    tl.store(out_ptr + offsets, y)


@oracle_impl(
    hardware="B200",
    point="d9611874",
    BLOCK=4096,
    num_warps=8,
    num_stages=3,
)
def oracle_forward(
    inputs,
    *,
    BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape0, _shape1, _shape2 = inputs
    out = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _gptj_bf16_layernorm_kernel[(ROWS,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        out,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
