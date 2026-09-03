"""cuTile port of sum_59cd03f5c32f: selected-slice L2 normalization.

Ports the Triton `_selected_slice_l2_norm_kernel` — for each row of the last
timestep slice `arg0[:, -1, :]`, compute L2 norm via fp32 accumulation,
round through bf16 denominator, clamp to 1e-12, and divide.

cuTile's default rounding is round-to-nearest, matching Triton's `rtne`.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _selected_slice_l2_norm_kernel(
    x_ptr,          # bf16 [M, N] (single timestep slice)
    out_ptr,        # bf16 [M, N]
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    values = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_N))
    values_f32 = ct.astype(values, ct.float32)
    sum_sq = ct.sum(values_f32 * values_f32)
    denom_bf16 = ct.astype(ct.sqrt(sum_sq), ct.bfloat16)
    denom = ct.astype(denom_bf16, ct.float32)
    # Clamp to 1e-12 minimum
    denom_clamped_bf16 = ct.astype(
        ct.astype(ct.where(denom < 1.0e-12, 1.0e-12, denom), ct.bfloat16), ct.float32,
    )
    out = ct.astype(values_f32 / denom_clamped_bf16, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=out)


@oracle_impl(hardware="B200", point="6cbb208b", BLOCK_N=256)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, _shape_param_0 = inputs
    out_shape = tuple(int(dim) for dim in _shape_param_0)
    rows = int(out_shape[0])
    cols = int(out_shape[1])

    # Slice the selected timestep: arg0[:, -1, :] -> bf16[64, 256]
    selected = arg0_1[:, -1, :].contiguous()
    out = torch.empty_strided(out_shape, (cols, 1), device=arg0_1.device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _selected_slice_l2_norm_kernel,
        (selected, out, BLOCK_N),
    )
    return out
