"""cuTile port of mean_426e2409482e: MT5 bf16 residual RMSNorm with 17 view aliases.

Row kernel: bf16 residual add (rtne), fp32 mean-square over HIDDEN=512, rsqrt
with eps=1e-6, bf16 normalize, bf16 weight multiply, output bf16 base tensor.
Then produce 16 view aliases from the base output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _residual_rmsnorm_kernel(
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
    add = resid_f + flat_f
    add_bf16 = ct.astype(add, ct.bfloat16)

    x = ct.astype(add_bf16, ct.float32)
    sum_sq = ct.sum(x * x)
    inv_rms = ct.rsqrt(sum_sq * (1.0 / HIDDEN) + 1.0e-6)
    norm_bf16 = ct.astype(x * inv_rms, ct.bfloat16)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    out = ct.astype(
        ct.astype(norm_bf16, ct.float32) * ct.astype(weight_2d, ct.float32),
        ct.bfloat16,
    )
    ct.store(out_ptr, index=(row, 0), tile=out)


@oracle_impl(hardware="B200", point="841ed042", BLOCK_H=512)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, *shape_args = inputs
    # arg0_1 bf16[4096, 512]; arg1_1 bf16[32, 128, 512]; arg2_1 bf16[512]
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    flat = arg0_1
    residual = arg1_1.view(rows, hidden)

    out = torch.empty_strided(
        (rows, hidden), (hidden, 1),
        device=arg0_1.device, dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1),
        _residual_rmsnorm_kernel,
        (flat, residual, arg2_1, out, hidden, BLOCK_H),
    )
    # Match the repro: (mul_1, view_1 ... view_16)
    # mul_1 has shape [batch, seq, hidden] contiguous; base view is our
    # [rows, hidden] out — the repro's Repro.forward multiplies weight * norm
    # over [32,128,512] which is elementwise; then reshapes to 16 [4096,512]
    # views. Since bf16 tensor storage is 1-1 with our out, we return
    # out.view(batch, seq, hidden) as mul_1 and 16 views of the same storage.
    batch = int(arg1_1.shape[0])
    seq = int(arg1_1.shape[1])
    mul_1 = out.view(batch, seq, hidden)
    view_shapes = [tuple(int(dim) for dim in s) for s in shape_args[1:]]
    views = tuple(out.view(shape) for shape in view_shapes)
    return (mul_1,) + views
