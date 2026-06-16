"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete NFNet weight-standardization scope in one point-specialized Triton row kernel, matching the population Welford `var_mean` over each `[1,C,K]` logical channel row, `rsqrt(var + 1e-5)`, the generated multiply order `((x - mean) * invstd) * (gain * 0.02551551815399144)`, the final bf16 cast for the standardized weight, and the f32 invstd and keepdim-mean side outputs, whereas Inductor lowers the captured view/var_mean/rsqrt/scale/cast graph through its generic normalization reduction schedule; Inductor cannot do this today because the scheduler lacks a fixed-inner-size weight-standardization fusion template that retains the reduction tile for the broadcast epilogue while directly emitting multiple side-output layouts and the required bf16 cast boundary; the fix is SCHEDULER_FUSION: add a guarded NFNet weight-standardization norm template that preserves Welford accumulation, fuses the dependent row epilogue, and stores the side outputs directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _f32_sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _weight_standardization_kernel(
    weight_ptr,
    gain_ptr,
    invstd_ptr,
    standardized_ptr,
    mean_ptr,
    inner_size: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    channel = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)[None, :]
    mask = offsets < inner_size
    row_offsets = channel * inner_size + offsets

    x = tl.load(weight_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    mean_acc = tl.zeros([1, BLOCK_K], tl.float32)
    m2_acc = tl.zeros([1, BLOCK_K], tl.float32)
    weight_acc = tl.zeros([1, BLOCK_K], tl.float32)
    mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
        x,
        mean_acc,
        m2_acc,
        weight_acc,
        True,
    )
    mean_acc = tl.where(mask, mean_next, mean_acc)
    m2_acc = tl.where(mask, m2_next, m2_acc)
    weight_acc = tl.where(mask, weight_next, weight_acc)
    mean, m2, _weight = triton_helpers.welford(mean_acc, m2_acc, weight_acc, 1)
    invstd = libdevice.rsqrt((m2 / inner_size) + 1.0e-5)

    one = tl.arange(0, 1)
    tl.store(invstd_ptr + channel + one, invstd, mask=one == 0)
    tl.store(mean_ptr + channel + one, mean, mask=one == 0)

    gain_scaled = _f32_mul(
        tl.load(gain_ptr + channel).to(tl.float32),
        0.02551551815399144,
    )
    centered = _f32_sub(x, mean[:, None])
    normalized = _f32_mul(centered, invstd[:, None])
    out = _f32_mul(normalized, gain_scaled)
    tl.store(standardized_ptr + row_offsets, out, mask=mask)


# edad2384: (T([3072,1536,1,1], f32), T([3072,1,1,1], f32), ...)
# 37a1f8e3: (T([1536,768,1,1], f32), T([1536,1,1,1], f32), ...)
# a943dc7a: (T([768,1536,1,1], f32), T([768,1,1,1], f32), ...)
# 988d995a: (T([1536,1536,1,1], f32), T([1536,1,1,1], f32), ...)
# ec12628c: (T([768,512,1,1], f32), T([768,1,1,1], f32), ...)
# ce4c9a46: (T([1536,512,1,1], f32), T([1536,1,1,1], f32), ...)
# b5637f4d: (T([512,256,1,1], f32), T([512,1,1,1], f32), ...)
# b9d624b6: (T([256,512,1,1], f32), T([256,1,1,1], f32), ...)
# 3749732f: (T([256,256,1,1], f32), T([256,1,1,1], f32), ...)
# 3b7e26b8: (T([256,128,1,1], f32), T([256,1,1,1], f32), ...)
# 1d6ba0cb: (T([128,128,1,1], f32), T([128,1,1,1], f32), ...)
# d031b94a: (T([2304,1536,1,1], f32), T([2304,1,1,1], f32), ...)
# 2d1e1921: (T([1536,384,1,1], f32), T([1536,1,1,1], f32), ...)
# 924b986a: (T([384,1536,1,1], f32), T([384,1,1,1], f32), ...)
# 7b2c0035: (T([384,512,1,1], f32), T([384,1,1,1], f32), ...)
# e32f54cf: (T([512,128,1,1], f32), T([512,1,1,1], f32), ...)
# a20fad4b: (T([128,512,1,1], f32), T([128,1,1,1], f32), ...)
# 3b29d126: (T([128,256,1,1], f32), T([128,1,1,1], f32), ...)
# e2b671a1: (T([256,64,1,1], f32), T([256,1,1,1], f32), ...)
# e1e52b0b: (T([64,128,1,1], f32), T([64,1,1,1], f32), ...)
@oracle_impl(hardware="B200", point="edad2384", BLOCK_K=2048, num_warps=4)
@oracle_impl(hardware="B200", point="37a1f8e3", BLOCK_K=1024, num_warps=4)
@oracle_impl(hardware="B200", point="a943dc7a", BLOCK_K=2048, num_warps=4)
@oracle_impl(hardware="B200", point="988d995a", BLOCK_K=2048, num_warps=4)
@oracle_impl(hardware="B200", point="ec12628c", BLOCK_K=512, num_warps=4)
@oracle_impl(hardware="B200", point="ce4c9a46", BLOCK_K=512, num_warps=4)
@oracle_impl(hardware="B200", point="b5637f4d", BLOCK_K=256, num_warps=1)
@oracle_impl(hardware="B200", point="b9d624b6", BLOCK_K=512, num_warps=4)
@oracle_impl(hardware="B200", point="3749732f", BLOCK_K=256, num_warps=1)
@oracle_impl(hardware="B200", point="3b7e26b8", BLOCK_K=128, num_warps=1)
@oracle_impl(hardware="B200", point="1d6ba0cb", BLOCK_K=128, num_warps=1)
@oracle_impl(hardware="B200", point="d031b94a", BLOCK_K=2048, num_warps=4)
@oracle_impl(hardware="B200", point="2d1e1921", BLOCK_K=512, num_warps=4)
@oracle_impl(hardware="B200", point="924b986a", BLOCK_K=2048, num_warps=4)
@oracle_impl(hardware="B200", point="7b2c0035", BLOCK_K=512, num_warps=4)
@oracle_impl(hardware="B200", point="e32f54cf", BLOCK_K=128, num_warps=1)
@oracle_impl(hardware="B200", point="a20fad4b", BLOCK_K=512, num_warps=4)
@oracle_impl(hardware="B200", point="3b29d126", BLOCK_K=256, num_warps=1)
@oracle_impl(hardware="B200", point="e2b671a1", BLOCK_K=64, num_warps=1)
@oracle_impl(hardware="B200", point="e1e52b0b", BLOCK_K=128, num_warps=1)
def oracle_forward(inputs, *, BLOCK_K: int, num_warps: int):
    weight, gain, _view_shape, out_shape = inputs
    channels = int(weight.shape[0])
    inner_size = int(weight.shape[1])
    out_shape = tuple(int(dim) for dim in out_shape)
    out_stride = (inner_size, 1, 1, 1)

    invstd = torch.empty_strided(
        (channels,),
        (1,),
        device=weight.device,
        dtype=torch.float32,
    )
    standardized = torch.empty_strided(
        out_shape,
        out_stride,
        device=weight.device,
        dtype=torch.bfloat16,
    )
    mean = torch.empty_strided(
        (1, channels, 1),
        (channels, 1, 1),
        device=weight.device,
        dtype=torch.float32,
    )

    _weight_standardization_kernel[(channels,)](
        weight,
        gain,
        invstd,
        standardized,
        mean,
        inner_size=inner_size,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=3,
    )
    return invstd, standardized, mean
