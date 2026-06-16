"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 attention-probability backward row update in one Triton row-reduction kernel, including the metadata-only view, bf16-to-f32 input promotion, fp32 product and last-dimension sum, exact fma.rn.f32 epilogue, final bf16 cast, and returned contiguous `[512, K, K]` view, whereas Inductor lowers the decomposed view/cast/mul/sum/neg/fma/cast/view graph through its generic reduction scheduler; Inductor cannot do this today as a reusable B200-tuned full-scope row template across the captured K=128 and K=512 points because the scheduler does not specialize the reduction and dependent fma epilogue together with the view-only shape changes; the fix is SCHEDULER_FUSION: add a guarded row-reduction epilogue template that sinks the product, row sum, fma, bf16 store, and view-only reshapes into one scheduled unit."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _mul_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fma_rn_f32(a, b, c):
    return tl.inline_asm_elementwise(
        "fma.rn.f32 $0, $1, $2, $3;",
        constraints="=f,f,f,f",
        args=[a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _row_sum_fma_kernel(
    x_ptr,
    grad_ptr,
    out_ptr,
    N_ROWS: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_K)[None, :]
    mask = (rows < N_ROWS) & (cols < K)
    offsets = rows * K + cols

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    product = _mul_rn_f32(x, grad)
    row_sum = tl.sum(tl.where(mask, product, 0.0), axis=1)[:, None].to(tl.float32)
    out = _fma_rn_f32(-grad, row_sum, product)

    tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=mask)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


# (T([512,128,128], bf16), T([16,32,128,128], bf16), S([16,32,128,128]), S([512,128,128]))
@oracle_impl(hardware="B200", point="0c938b20", BLOCK_M=4, BLOCK_K=128, num_warps=4)
# (T([512,512,512], bf16), T([8,64,512,512], bf16), S([8,64,512,512]), S([512,512,512]))
@oracle_impl(hardware="B200", point="b31e9601", BLOCK_M=2, BLOCK_K=512, num_warps=8)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_K: int, num_warps: int):
    arg0_1, arg1_1, _shape_param_0, shape_param_1 = inputs
    out_shape = _shape_tuple(shape_param_1)
    k = int(arg1_1.shape[-1])
    rows = arg1_1.numel() // k

    out = torch.empty_strided(
        out_shape,
        tuple(arg0_1.stride()),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _row_sum_fma_kernel[(triton.cdiv(rows, BLOCK_M),)](
        arg0_1,
        arg1_1,
        out,
        N_ROWS=rows,
        K=k,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=1,
    )
    return out
