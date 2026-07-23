"""cuTile port of mean_f08c75b5f21f: MT5 bf16 residual-add RMSNorm.

SCHEDULER_FUSION: for each row of the bf16 [rows, HIDDEN=512] input, add the
matching row of the residual (rounded to bf16), compute fp32 mean-square
rsqrt with eps=1e-6, normalize/round to bf16, apply the shared weight vector,
and write into the contiguous base buffer. Returns the base viewed as
[32, 128, 512].
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _residual_rmsnorm_bf16_kernel(
    flat_ptr,      # bf16 [rows, HIDDEN]
    residual_ptr,  # bf16 [rows, HIDDEN]
    weight_ptr,    # bf16 [HIDDEN]
    out_ptr,       # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))

    flat_f = ct.astype(flat, ct.float32)
    resid_f = ct.astype(residual, ct.float32)
    added_bf16 = ct.astype(resid_f + flat_f, ct.bfloat16)

    x = ct.astype(added_bf16, ct.float32)
    sum_sq = ct.sum(x * x)
    inv_rms = ct.rsqrt(sum_sq * (1.0 / HIDDEN) + 1.0e-6)
    norm_bf16 = ct.astype(x * inv_rms, ct.bfloat16)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    weight_f = ct.astype(weight, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    out = ct.astype(weight_2d * ct.astype(norm_bf16, ct.float32), ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=out)


@oracle_impl(
    hardware="B200",
    point="841ed042",
    BLOCK_H=512,
    BLOCK_M=1,
)
def oracle_forward(inputs, *, BLOCK_H: int, BLOCK_M: int):
    arg0_1, arg1_1, arg2_1, base_shape_arg, out_shape_arg = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    base_shape = tuple(int(dim) for dim in base_shape_arg)
    out_shape = tuple(int(dim) for dim in out_shape_arg)

    # arg1_1 is shape [32, 128, 512] contiguous — reshape to [rows, hidden].
    residual_2d = arg1_1.view(rows, hidden)

    out_base = torch.empty_strided(
        (rows, hidden),
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _residual_rmsnorm_bf16_kernel,
        (arg0_1, residual_2d, arg2_1, out_base, hidden, BLOCK_H),
    )
    return out_base.view(out_shape)
