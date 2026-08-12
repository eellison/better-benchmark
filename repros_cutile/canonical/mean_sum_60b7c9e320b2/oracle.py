"""cuTile port of mean_sum_60b7c9e320b2: RMSNormBackward-static.

Reference computes:
  rsqrt = 1 / sqrt(mean(x**2, dim=-1) + 1e-6)
  y = bf16(x * rsqrt * weight)
  returns (rsqrt, bf16(sum(y)))

Row-wise RMSNorm over 1152000 rows of 512, then global sum of the affine
output reduced back to bf16. Uses BLOCK_M=8 rows per program (matching
Triton) so a single kernel produces both rsqrt per row and a partial sum per
row-block. Then two reduction passes fold to a scalar.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HIDDEN = 512


@ct.kernel
def _rmsnorm_row_sum_kernel(
    x_ptr,          # bf16 [ROWS, HIDDEN]
    weight_ptr,     # f32 [HIDDEN]
    rsqrt_ptr,      # f32 [ROWS] (viewed as (ROWS,))
    partial_ptr,    # f32 [N_PARTIALS] (one partial per row-block)
    HIDDEN_: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
):
    row_block = ct.bid(0)
    x = ct.load(x_ptr, index=(row_block, 0), shape=(BLOCK_M, HIDDEN_))
    x_f = ct.astype(x, ct.float32)
    sum_sq = ct.sum(x_f * x_f, axis=1)                                # (BLOCK_M,)
    inv_rms = ct.rsqrt(sum_sq * (1.0 / HIDDEN_) + 1.0e-6)             # (BLOCK_M,)
    ct.store(rsqrt_ptr, index=(row_block,), tile=inv_rms)

    weight = ct.load(weight_ptr, index=(0,), shape=(HIDDEN_,))
    weight_f = ct.reshape(ct.astype(weight, ct.float32), (1, HIDDEN_))
    inv_rms_2d = ct.reshape(inv_rms, (BLOCK_M, 1))
    affine = x_f * inv_rms_2d * weight_f
    affine_bf16 = ct.astype(affine, ct.bfloat16)
    affine_f32 = ct.astype(affine_bf16, ct.float32)
    block_sum = ct.sum(affine_f32)  # scalar
    ct.store(partial_ptr, index=(row_block,), tile=ct.reshape(block_sum, (1,)))


@ct.kernel
def _sum_block_kernel(
    partials_ptr,
    out_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    v = ct.load(partials_ptr, index=(pid,), shape=(BLOCK,),
                padding_mode=ct.PaddingMode.ZERO)
    total = ct.sum(v)
    ct.store(out_ptr, index=(pid,), tile=ct.reshape(total, (1,)))


@ct.kernel
def _final_bf16_round_kernel(
    partial_ptr,
    out_ptr,
    BLOCK: ct.Constant[int],
):
    v = ct.load(partial_ptr, index=(0,), shape=(BLOCK,),
                padding_mode=ct.PaddingMode.ZERO)
    total = ct.sum(v)
    total_bf16 = ct.astype(ct.reshape(total, (1,)), ct.bfloat16)
    ct.store(out_ptr, index=(0,), tile=total_bf16)


@oracle_impl(hardware="B200", point="e5ae55b5", BLOCK_M=8, BLOCK_H=512, REDUCE_BLOCK=1024)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_H: int, REDUCE_BLOCK: int):
    x, weight = inputs
    rows = int(x.shape[0])
    hidden = int(x.shape[1])
    assert hidden == HIDDEN, f"unexpected HIDDEN={hidden}"

    rsqrt = torch.empty_strided(
        (rows, 1), (1, 1), device=x.device, dtype=torch.float32
    )
    rsqrt_flat = rsqrt.view(rows)
    n_partials = ct.cdiv(rows, BLOCK_M)
    partials = torch.empty((n_partials,), device=x.device, dtype=torch.float32)
    out = torch.empty((), device=x.device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_partials, 1, 1), _rmsnorm_row_sum_kernel,
        (x, weight, rsqrt_flat, partials, hidden, BLOCK_M),
    )

    n_partials2 = ct.cdiv(n_partials, REDUCE_BLOCK)
    partials2 = torch.empty((n_partials2,), device=x.device, dtype=torch.float32)
    ct.launch(stream, (n_partials2, 1, 1), _sum_block_kernel,
              (partials, partials2, REDUCE_BLOCK))

    def _next_p2(v):
        return 1 << (v - 1).bit_length()

    BLOCK_FINAL = max(_next_p2(n_partials2), 1)
    ct.launch(stream, (1, 1, 1), _final_bf16_round_kernel,
              (partials2, out.view(1), BLOCK_FINAL))
    return rsqrt, out
