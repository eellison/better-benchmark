"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle keeps the selected-add row tile live through population var_mean, rsqrt normalization, affine scale/bias, and the returned 3D output plus 2D view alias in one Triton row kernel, whereas Inductor currently emits one generic fused Welford reduction kernel but reloads and recomputes `arg1_1 + select(arg0_1, 3, 0)` for the normalization epilogue; Inductor cannot do this today because the generic reduction scheduler does not retain a full fixed-width reduction tile's producer values across a multi-output Welford reduction for downstream row-wise consumers and alias-only view returns; the fix is SCHEDULER_FUSION: teach the row-reduction schedule/template to preserve fixed hidden-size producer tiles across var_mean and directly feed the affine epilogue and alias-view outputs when register pressure is acceptable."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _select_add_layernorm_kernel(
    view_real_ptr,
    add_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = tl.arange(0, BLOCK_N)[None, :]
    mask = (rows < ROWS) & (cols < HIDDEN)

    flat_offsets = rows * HIDDEN + cols
    add_values = tl.load(add_ptr + flat_offsets, mask=mask, other=0.0)
    selected = tl.load(view_real_ptr + flat_offsets * 2, mask=mask, other=0.0)
    values = tl.where(mask, add_values + selected, 0.0)

    mean_vec = values
    m2_vec = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    weight_vec = tl.where(mask, 1.0, 0.0)
    mean, m2, _weight = triton_helpers.welford(mean_vec, m2_vec, weight_vec, 1)
    invstd = libdevice.rsqrt(m2 / HIDDEN + 1.0e-12)[:, None]

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    out = ((values - mean[:, None]) * invstd) * weight + bias
    tl.store(out_ptr + flat_offsets, out, mask=mask)


@oracle_impl(
    hardware="B200",
    point="98ade792",
    BLOCK_M=2,
    BLOCK_N=1024,
    num_warps=4,
    num_stages=3,
)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N, num_warps, num_stages):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0 = inputs
    rows = int(arg1_1.numel() // arg1_1.shape[-1])
    hidden = int(arg1_1.shape[-1])

    out = torch.empty_strided(
        tuple(int(dim) for dim in arg1_1.shape),
        tuple(int(stride) for stride in arg1_1.stride()),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    _select_add_layernorm_kernel[(triton.cdiv(rows, BLOCK_M),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        out,
        ROWS=rows,
        HIDDEN=hidden,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, out.view(tuple(int(dim) for dim in shape0))
