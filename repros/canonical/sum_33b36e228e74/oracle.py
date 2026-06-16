"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 T5/MT5 attention softmax-backward row update in one Triton kernel, including the shape-param view, bool dropout-mask cast and scale with the captured bf16 rounding boundary, saved-logit libdevice.exp normalization, fp32 row sum, prims.fma-equivalent epilogue, explicit bf16 cast, and both returned view aliases from the same storage, whereas Inductor lowers the decomposed view/convert/mul/sub/exp/div/sum/neg/fma/cast/view chain through generic reduction and layout scheduling; Inductor cannot do this today because its scheduler/codegen does not recognize this saved-softmax backward epilogue with observable bf16 rounding and sibling metadata-only view returns as one full-scope row-reduction template; the fix is SCHEDULER_FUSION: add a guarded attention-backward row template that fuses the mask scaling, saved-softmax reconstruction, row reduction, fma epilogue, bf16 store, and alias-return handling."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _f32_fma(a, b, c):
    return tl.inline_asm_elementwise(
        "fma.rn.f32 $0, $1, $2, $3;",
        constraints="=f,f,f,f",
        args=[a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _softmax_backward_bf16_kernel(
    grad_ptr,
    keep_ptr,
    logits_ptr,
    row_max_ptr,
    row_sum_ptr,
    out_ptr,
    N_ROWS: tl.constexpr,
    N_COLS: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    cols = tl.arange(0, BLOCK_N)
    offsets = rows[:, None] * N_COLS + cols[None, :]

    grad_bf16 = tl.load(grad_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
    keep = tl.load(keep_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
    logits = tl.load(logits_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
    row_max = tl.load(row_max_ptr + rows).to(tl.float32)
    denom = tl.load(row_sum_ptr + rows).to(tl.float32)

    scale = tl.full((ROWS_PER_PROGRAM, BLOCK_N), 1.1111111111111112, tl.float32)
    scaled_keep = (keep * scale).to(tl.bfloat16).to(tl.float32)
    dropped_grad = (grad_bf16 * scaled_keep).to(tl.bfloat16).to(tl.float32)

    probs = libdevice.exp(logits - row_max[:, None]) / denom[:, None]
    product = dropped_grad * probs
    row_dot = tl.sum(product, axis=1)[:, None]
    out = _f32_fma(-probs, row_dot, product).to(tl.bfloat16)

    tl.store(out_ptr + offsets, out)


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    running = 1
    strides = []
    for dim in reversed(shape):
        strides.append(running)
        running *= dim
    return tuple(reversed(strides))


@oracle_impl(hardware="B200", point="e0876406", ROWS_PER_PROGRAM=2, BLOCK_N=1024, num_warps=4)
@oracle_impl(hardware="B200", point="59cff286", ROWS_PER_PROGRAM=16, BLOCK_N=128, num_warps=4)
def oracle_forward(inputs, *, ROWS_PER_PROGRAM: int, BLOCK_N: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    del _shape_param_0, _shape_param_1

    out_shape = tuple(int(dim) for dim in _shape_param_2)
    flat_shape = tuple(int(dim) for dim in _shape_param_3)
    n_cols = int(out_shape[-1])
    n_rows = arg2_1.numel() // n_cols

    out = torch.empty_strided(
        out_shape,
        _contiguous_strides(out_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _softmax_backward_bf16_kernel[(triton.cdiv(n_rows, ROWS_PER_PROGRAM),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        out,
        N_ROWS=n_rows,
        N_COLS=n_cols,
        ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out, out.view(flat_shape)
