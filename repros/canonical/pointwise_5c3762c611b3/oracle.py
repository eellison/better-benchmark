"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MobileBERT bf16 residual column-affine pointwise scope in one storage-linear Triton kernel, including the metadata-only `[32768,128] -> [256,128,128]` input views, both hidden-dimension broadcasts, the captured bf16 rounding boundary after each visible `aten.mul`/`aten.add`, the final bf16 store, and both returned aliases of the fresh contiguous output buffer; Inductor already lowers this local graph to a single fused pointwise kernel, but the oracle records the full two-output aliasing envelope explicitly so the floor is not a subset of the captured repro; the fix is NEW_PATTERN: keep this as a full-scope pointwise floor unless broader pointwise scheduling or alias-output lowering changes both implementations."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 32768
HIDDEN = 128
N_ELEMENTS = ROWS * HIDDEN
VIEW_SHAPE = (256, 128, 128)
VIEW_STRIDE = (128 * 128, 128, 1)
FLAT_SHAPE = (ROWS, HIDDEN)
FLAT_STRIDE = (HIDDEN, 1)


@triton.jit
def _round_to_bf16_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=4),
    ],
    key=[],
)
@triton.jit
def _residual_column_affine_kernel(
    arg0_ptr,
    arg1_ptr,
    scale0_ptr,
    bias0_ptr,
    scale1_ptr,
    bias1_ptr,
    out_ptr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    cols = offsets & (HIDDEN_SIZE - 1)

    x0 = tl.load(arg0_ptr + offsets).to(tl.float32)
    x1 = tl.load(arg1_ptr + offsets).to(tl.float32)
    scale0 = tl.load(scale0_ptr + cols, eviction_policy="evict_last").to(tl.float32)
    bias0 = tl.load(bias0_ptr + cols, eviction_policy="evict_last").to(tl.float32)
    scale1 = tl.load(scale1_ptr + cols, eviction_policy="evict_last").to(tl.float32)
    bias1 = tl.load(bias1_ptr + cols, eviction_policy="evict_last").to(tl.float32)

    mul = _round_to_bf16_f32(x1 * scale0)
    add = _round_to_bf16_f32(mul + bias0)
    add_1 = _round_to_bf16_f32(x0 + add)
    mul_1 = _round_to_bf16_f32(add_1 * scale1)
    add_2 = mul_1 + bias1
    tl.store(out_ptr + offsets, add_2)


# b07f4689: (T([32768,128], bf16), T([32768,128], bf16), 4x T([128], bf16), S([256,128,128]), S([256,128,128]), S([32768,128]))
@oracle_impl(hardware="B200", point="b07f4689")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, *_shape_params = inputs
    del _shape_params

    out_flat = torch.empty_strided(
        FLAT_SHAPE,
        FLAT_STRIDE,
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    grid = lambda meta: (triton.cdiv(N_ELEMENTS, meta["BLOCK"]),)
    _residual_column_affine_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        out_flat,
        HIDDEN_SIZE=HIDDEN,
    )
    return out_flat.as_strided(VIEW_SHAPE, VIEW_STRIDE), out_flat
