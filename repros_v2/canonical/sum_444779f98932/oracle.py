"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete Longformer sliding-window attention-backward band reconstruction, saved-mask/dropout selects, natural-exp probability reconstruction, exact eager-layout f32 row product sum, fma epilogue, structured slice/select-scatter bases, and cropped `[288,512,512]` output with specialized Triton layout kernels around the numerically faithful row reduction, whereas Inductor lowers the same fixed-width diagonal band as separate generic pointwise, reduction, permute, clone, scatter, and crop materializations; Inductor cannot do this today because scheduler/codegen does not recognize the Longformer diagonal-band backward scatter-reduce producer/consumer and cannot preserve the required row-sum order while sinking the downstream scatter layout; the fix is SCATTER_REDUCE: add a guarded Longformer sliding-window attention-backward lowering that fuses band assembly, row reduction/fma, and output-layout materialization while preserving bf16 casts and slice/scatter contracts."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _round_bf16_to_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _div_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _product_kernel(
    query_mask_ptr,
    scalar_ptr,
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    keep_ptr,
    logits_ptr,
    row_shift_ptr,
    row_denom_ptr,
    out_ptr,
    XBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    x1 = ((xindex // 513) % 1024)
    x3 = xindex // 6303744
    x0 = xindex % 513
    x4 = xindex // 525312
    x5 = xindex // 513
    x2 = ((xindex // 525312) % 12)
    logits_off = x0 + 513 * x1 + 525312 * x2 + 6303744 * x3
    x6 = x0 + 513 * x2 + 6156 * x1 + 6303744 * x3
    tmp0 = tl.load(query_mask_ptr + x1 + 1024 * x3, eviction_policy="evict_last").to(tl.int1)
    tmp1 = tl.load(scalar_ptr + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp23 = tl.load(keep_ptr + x0 + 513 * x2 + 6272 * x1 + 6422528 * x3).to(tl.int1)
    tmp30 = tl.load(logits_ptr + logits_off).to(tl.float32)
    tmp32 = tl.load(row_shift_ptr + x2 + 12 * x1 + 12288 * x3, eviction_policy="evict_last")
    tmp35 = tl.load(row_denom_ptr + x2 + 12 * x1 + 12288 * x3, eviction_policy="evict_last")
    tmp5 = x0 < 770
    tmp6 = x0 + 770 * (x1 % 256)
    tmp8 = tmp6 < 196864
    tmp9 = tmp8 & tmp5
    tmp10 = tmp6 % 769
    tmp12 = tmp10 < 768
    tmp13 = tmp12 & tmp9
    tmp14 = tl.load(
        arg0_ptr + 768 * (tmp6 // 769) + 196608 * (x1 // 256) + 786432 * x4 + tmp10,
        tmp13,
        other=0.0,
    ).to(tl.float32)
    tmp15 = tl.load(arg1_ptr + x0 + 770 * (x1 % 256) + 196864 * (x1 // 256) + 787456 * x4, tmp9, other=0.0).to(tl.float32)
    tmp16 = tl.where(tmp12, tmp14, tmp15)
    tmp18 = tl.where(tmp9, tmp16, tl.full([XBLOCK], 0.0, tl.float32))
    tmp19 = tl.load(arg2_ptr + x0 + 770 * x5, tmp5, other=0.0).to(tl.float32)
    tmp20 = tl.where(tmp8, tmp18, tmp19)
    tmp22 = tl.where(tmp5, tmp20, tl.full([XBLOCK], 0.0, tl.float32))
    tmp26 = _round_bf16_to_f32(tmp23.to(tl.float32) * 1.1111111111111112)
    tmp28 = _round_bf16_to_f32(tmp22 * tmp26)
    tmp29 = tl.where(tmp0, tmp2, tmp28)
    tmp33 = tmp30 - tmp32
    tmp34 = libdevice.exp(tmp33)
    tmp36 = _div_rn_f32(tmp34, tmp35)
    tmp37 = tmp29 * tmp36
    tl.store(out_ptr + x6, tmp37)


@triton.jit
def _sum_fma_kernel(
    product_ptr,
    logits_ptr,
    row_shift_ptr,
    row_denom_ptr,
    sum_ptr,
    out_ptr,
    XBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    cbase = tl.arange(0, CBLOCK)[None, :]
    x0 = xindex % 12
    x1 = (xindex // 12) % 1024
    x2 = xindex // 12288
    x5 = xindex // 12
    row_sum = tl.load(sum_ptr + xindex)
    prod2 = tl.load(product_ptr + cbase + 513 * x0 + 6156 * x1 + 6303744 * x2, eviction_policy="evict_first")
    logits = tl.load(logits_ptr + cbase + 513 * x1 + 525312 * x0 + 6303744 * x2, eviction_policy="evict_first").to(tl.float32)
    shift = tl.load(row_shift_ptr + xindex, eviction_policy="evict_last")
    denom = tl.load(row_denom_ptr + xindex, eviction_policy="evict_last")
    probs = _div_rn_f32(libdevice.exp(logits - shift), denom)
    out = libdevice.fma(-probs, row_sum, prod2)
    tl.store(out_ptr + cbase + 513 * x0 + 6208 * x5, out.to(tl.float32))
    prod2_tail = tl.load(product_ptr + 512 + 513 * x0 + 6156 * x1 + 6303744 * x2, eviction_policy="evict_first")
    logits_tail = tl.load(logits_ptr + 512 + 513 * x1 + 525312 * x0 + 6303744 * x2, eviction_policy="evict_first").to(tl.float32)
    probs_tail = _div_rn_f32(libdevice.exp(logits_tail - shift), denom)
    out_tail = libdevice.fma(-probs_tail, row_sum, prod2_tail)
    tl.store(out_ptr + 512 + 513 * x0 + 6208 * x5, out_tail.to(tl.float32))


@triton.jit
def _add_band_kernel(
    out_ptr,
    arg9_ptr,
    band_ptr,
    mask10_ptr,
    scalar_ptr,
    arg12_ptr,
    arg13_ptr,
    mask14_ptr,
    XBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    x2 = xindex // 49248
    x0 = xindex % 513
    x1 = (xindex // 513) % 96
    x3 = xindex % 49248
    tmp12 = tl.load(band_ptr + x0 + 513 * (x1 % 12) + 6208 * x2 + 6356992 * (x1 // 12)).to(tl.float32)
    tmp26 = tl.load(arg13_ptr + x0 + 513 * (x1 % 12) + 6156 * x2 + 6303744 * (x1 // 12)).to(tl.float32)
    tmp2 = x2 >= 768
    tmp5 = x0 >= 256
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(arg9_ptr + (-2368768) + x0 + 257 * (x1 % 12) + 3084 * x2 + 789504 * (x1 // 12), tmp6, other=0.0).to(tl.float32)
    tmp8 = tl.load(band_ptr + x0 + 513 * (x1 % 12) + 6208 * x2 + 6356992 * (x1 // 12), tmp2, other=0.0).to(tl.float32)
    tmp9 = tl.where(tmp5, tmp7, tmp8)
    tmp11 = tl.where(tmp2, tmp9, tl.full([XBLOCK], 0.0, tl.float32))
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = tl.load(mask10_ptr + (-2368768) + x0 + 257 * (x1 % 12) + 3084 * x2 + 789504 * (x1 // 12), tmp6, other=0.0).to(tl.int1)
    tmp15 = tl.load(scalar_ptr + 0).to(tl.float32)
    tmp17 = tl.where(tmp6, tl.broadcast_to(tmp15, [XBLOCK]), 0.0)
    tmp18 = tl.load(band_ptr + x0 + 513 * (x1 % 12) + 6208 * x2 + 6356992 * (x1 // 12), tmp6, other=0.0).to(tl.float32)
    tmp19 = tl.where(tmp14, tmp17, tmp18)
    tmp21 = tl.where(tmp6, tmp19, tl.full([XBLOCK], 0.0, tl.float32))
    tmp22 = tl.load(arg12_ptr + (-4727808) + x0 + 513 * (x1 % 12) + 6156 * x2 + 1575936 * (x1 // 12), tmp2, other=0.0).to(tl.float32)
    tmp23 = tl.where(tmp5, tmp21, tmp22)
    tmp25 = tl.where(tmp2, tmp23, tl.full([XBLOCK], 0.0, tl.float32))
    tmp27 = tl.where(tmp2, tmp25, tmp26)
    tmp28 = _round_bf16_to_f32(tmp13 + tmp27)
    tmp30 = x2 < 256
    tmp33 = x0 < 257
    tmp34 = tmp33 & tmp30
    tmp35 = tl.load(arg9_ptr + x0 + 257 * (x1 % 12) + 3084 * x2 + 789504 * (x1 // 12), tmp34, other=0.0).to(tl.float32)
    tmp36 = tl.where(tmp33, tmp35, tmp28)
    tmp38 = tl.where(tmp30, tmp36, tl.full([XBLOCK], 0.0, tl.float32))
    tmp39 = tl.where(tmp30, tmp38, tmp28)
    tmp40 = tl.load(mask14_ptr + x0 + 257 * (x1 % 12) + 3084 * x2 + 789504 * (x1 // 12), tmp34, other=0.0).to(tl.int1)
    tmp43 = tl.where(tmp34, tl.broadcast_to(tmp15, [XBLOCK]), 0.0)
    tmp44 = tl.where(tmp40, tmp43, tmp28)
    tmp46 = tl.where(tmp34, tmp44, tl.full([XBLOCK], 0.0, tl.float32))
    tmp47 = tl.load(arg12_ptr + x0 + 513 * (x1 % 12) + 6156 * x2 + 1575936 * (x1 // 12), tmp30, other=0.0).to(tl.float32)
    tmp48 = tl.where(tmp33, tmp46, tmp47)
    tmp50 = tl.where(tmp30, tmp48, tl.full([XBLOCK], 0.0, tl.float32))
    tmp51 = tl.where(tmp30, tmp50, tmp26)
    tmp52 = tmp39 + tmp51
    tl.store(out_ptr + x3 + 49280 * x2, tmp52)


@triton.jit
def _clone3_kernel(band_add_ptr, arg22_ptr, out_ptr, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    x0 = xindex % 513
    x1 = (xindex // 513) % 96
    x2 = xindex // 49248
    x4 = xindex % 49248
    tmp2 = x0 < 257
    band = tl.load(band_add_ptr + 37847296 + x4 + 49280 * x2, tmp2, other=0.0).to(tl.float32)
    base = tl.load(arg22_ptr + x0 + 513 * x2 + 131328 * x1, ~tmp2, other=0.0).to(tl.float32)
    tl.store(out_ptr + x0 + 513 * x2 + 131328 * x1, tl.where(tmp2, band, base))

@triton.jit
def _clone5_kernel(band_add_ptr, arg20_ptr, out_ptr, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    x0 = xindex % 513
    x2 = (xindex // 131328) % 3
    x3 = xindex // 393984
    x5 = xindex // 513
    x1 = (xindex // 513) % 256
    x6 = (xindex // 513) % 768
    x7 = xindex
    tmp2 = x0 < 257
    band = tl.load(band_add_ptr + 256 + x0 + 513 * x3 + 49280 * x6, tmp2, other=0.0).to(tl.float32)
    base = tl.load(arg20_ptr + x7, ~tmp2, other=0.0).to(tl.float32)
    tl.store(out_ptr + x7, tl.where(tmp2, band, base))


@triton.jit
def _add_final_kernel(
    out_ptr,
    arg15_ptr,
    band_add_ptr,
    arg20_ptr,
    arg18_ptr,
    arg16_ptr,
    arg17_ptr,
    buf6_ptr,
    buf8_ptr,
    XBLOCK: tl.constexpr,
):
    inner = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    group = tl.program_id(1)
    xout = inner + 262144 * group
    xindex = inner + 262656 * group
    x1 = (xindex // 513) % 512
    x0 = xindex % 513
    x2 = (xindex // 262656) % 3
    x3 = xindex // 787968
    x4 = xindex % 262656
    x5 = xindex // 262656
    x7 = xindex
    tmp39 = tl.load(arg18_ptr + x7).to(tl.float32)
    tmp54 = tl.load(arg17_ptr + x4 + 262656 * x3, eviction_policy="evict_last").to(tl.float32)
    tmp5 = (x1 >= 255) & (x1 < 511)
    tmp8 = x0 >= 257
    tmp9 = tmp8 & tmp5
    tmp10 = 1 + x2
    tmp12 = tmp10 == 0
    tmp13 = -255 + x1
    tmp15 = tmp13 >= 1
    tmp16 = tmp15 & tmp9
    tmp17 = -257 + x0
    tmp22 = (tmp17 >= 1) & (tmp17 < 256)
    tmp23 = tmp22 & tmp16
    tmp24 = tl.load(arg15_ptr + (-65538) + x0 + 255 * x1 + 65025 * x3, tmp23, eviction_policy="evict_last", other=0.0).to(tl.float32)
    tmp25 = tl.load(band_add_ptr + (-12566657) + x0 + 513 * x3 + 49280 * x1, tmp16, eviction_policy="evict_last", other=0.0).to(tl.float32)
    tmp26 = tl.where(tmp22, tmp24, tmp25)
    tmp28 = tl.where(tmp16, tmp26, tl.full([XBLOCK], 0.0, tl.float32))
    tmp29 = tl.load(band_add_ptr + (-12566657) + x0 + 513 * x3 + 49280 * x1, tmp9, eviction_policy="evict_last", other=0.0).to(tl.float32)
    tmp30 = tl.where(tmp15, tmp28, tmp29)
    tmp31 = tl.load(band_add_ptr + 49023 + x0 + 513 * x3 + 49280 * x1 + 12615680 * x2, tmp9, other=0.0).to(tl.float32)
    tmp32 = tl.where(tmp12, tmp30, tmp31)
    tmp34 = tl.where(tmp9, tmp32, tl.full([XBLOCK], 0.0, tl.float32))
    tmp35 = tl.load(arg20_ptr + (-130815) + x4 + 131328 * x5, tmp5, other=0.0).to(tl.float32)
    tmp36 = tl.where(tmp8, tmp34, tmp35)
    tmp38 = tl.where(tmp5, tmp36, tl.full([XBLOCK], 0.0, tl.float32))
    tmp40 = tl.where(tmp5, tmp38, tmp39)
    tmp43 = x2 == 0
    tmp44 = x1 < 255
    tmp47 = x0 >= 258
    tmp48 = tmp47 & tmp44
    tmp49 = tl.load(band_add_ptr + 49023 + x0 + 513 * x3 + 49280 * x1, tmp48, eviction_policy="evict_last", other=0.0).to(tl.float32)
    tmp50 = tl.load(arg16_ptr + x4 + 130815 * x3, tmp44, eviction_policy="evict_last", other=0.0).to(tl.float32)
    tmp51 = tl.where(tmp47, tmp49, tmp50)
    tmp53 = tl.where(tmp44, tmp51, tl.full([XBLOCK], 0.0, tl.float32))
    tmp55 = tl.where(tmp44, tmp53, tmp54)
    tmp56 = tl.where(tmp43, tmp55, tmp39)
    tmp57 = _round_bf16_to_f32(tmp56 + tmp40)
    tmp59 = x2 == 2
    tmp61 = x1 >= 256
    tmp62 = tl.load(buf6_ptr + (-131328) + x4 + 131328 * x3, tmp61, eviction_policy="evict_last", other=0.0).to(tl.float32)
    tmp63 = tl.where(tmp61, tmp62, tmp54)
    tmp64 = tl.where(tmp59, tmp63, tmp39)
    tmp65 = _round_bf16_to_f32(tmp57 + tmp64)
    tmp66 = x1 < 256
    tmp67 = tl.load(buf8_ptr + x4 + 131328 * x5, tmp66, other=0.0).to(tl.float32)
    tmp68 = tl.where(tmp66, tmp67, tmp39)
    tmp69 = tmp65 + tmp68
    tl.store(out_ptr + xout, tmp69)

@oracle_impl(hardware="B200", point="47e7063f", X=1024, RX=1, C=512, num_warps=8)
def oracle_forward(inputs, *, X: int, RX: int, C: int, num_warps: int):
    (
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        arg7,
        arg8,
        arg9,
        arg10,
        arg11,
        arg12,
        arg13,
        arg14,
        arg15,
        arg16,
        arg17,
        arg18,
        arg19,
        arg20,
        arg21,
        arg22,
        *_,
    ) = inputs
    buf0 = torch.empty_strided((8, 1024, 12, 513), (6303744, 6156, 513, 1), device=arg0.device, dtype=torch.float32)
    _product_kernel[(triton.cdiv(50429952, X),)](
        arg4,
        arg5,
        arg0,
        arg1,
        arg2,
        arg3,
        arg6,
        arg7,
        arg8,
        buf0,
        XBLOCK=X,
        num_warps=num_warps,
        num_stages=3,
    )
    row_sum = torch.sum(buf0, dim=-1, keepdim=True)
    buf2 = torch.empty_strided((8, 12, 1024, 513), (6356992, 513, 6208, 1), device=arg0.device, dtype=torch.bfloat16)
    _sum_fma_kernel[(triton.cdiv(98304, RX),)](
        buf0,
        arg6,
        arg7,
        arg8,
        row_sum,
        buf2,
        XBLOCK=RX,
        CBLOCK=C,
        num_warps=num_warps,
        num_stages=3,
    )
    buf4 = torch.empty_strided((96, 4, 256, 513), (513, 12615680, 49280, 1), device=arg0.device, dtype=torch.bfloat16)
    _add_band_kernel[(triton.cdiv(50429952, X),)](
        buf4,
        arg9,
        buf2,
        arg10,
        arg11,
        arg12,
        arg13,
        arg14,
        XBLOCK=X,
        num_warps=num_warps,
        num_stages=3,
    )
    buf6 = torch.empty_strided((96, 256, 513), (131328, 513, 1), device=arg0.device, dtype=torch.bfloat16)
    _clone3_kernel[(triton.cdiv(12607488, X),)](
        buf4,
        arg22,
        buf6,
        XBLOCK=X,
        num_warps=num_warps,
        num_stages=3,
    )
    buf8 = torch.empty_strided((96, 3, 256, 513), (393984, 131328, 513, 1), device=arg0.device, dtype=torch.bfloat16)
    _clone5_kernel[(triton.cdiv(37822464, X),)](
        buf4,
        arg20,
        buf8,
        XBLOCK=X,
        num_warps=num_warps,
        num_stages=3,
    )
    out = torch.empty_strided((96, 3, 512, 512), (786432, 262144, 512, 1), device=arg0.device, dtype=torch.bfloat16)
    _add_final_kernel[(triton.cdiv(262144, X), 288)](
        out,
        arg15,
        buf4,
        arg20,
        arg18,
        arg16,
        arg17,
        buf6,
        buf8,
        XBLOCK=X,
        num_warps=num_warps,
        num_stages=3,
    )
    return out.view(288, 512, 512)
