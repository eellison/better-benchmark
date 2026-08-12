"""cuTile port of sum_1ba47c0de2b0: GoogleFnet masked add + column sum.

Matches Triton's two-kernel structure: (1) materialize the fnet masked value
and produce per-row-block partial column sums; (2) finalize the column sum
over the partial buffer. Both reductions live INSIDE cuTile kernels — no
torch.sum in oracle_forward.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 16384
HIDDEN = 768
HIDDEN_PAD = 1024
ROWS_PER_GROUP = 64
BLOCK_C = 64
FINAL_BLOCK_C = 64
GROUPS = ROWS // ROWS_PER_GROUP
SCALE = 1.1111111640930176


@ct.kernel
def _store_and_partial_sum_kernel(
    real_ptr,       # f32 [ROWS, HIDDEN]  (sliced real component)
    residual_ptr,   # f32 [ROWS, HIDDEN]
    keep_ptr,       # b8  [ROWS, HIDDEN]
    out_ptr,        # f32 [ROWS, HIDDEN]
    partials_ptr,   # f32 [GROUPS, HIDDEN]
    ROWS_PER_GROUP_C: ct.Constant[int],
    BLOCK_C_C: ct.Constant[int],
    SCALE_C: ct.Constant[float],
):
    col_block = ct.bid(0)
    group = ct.bid(1)

    real = ct.load(real_ptr, index=(group, col_block),
                   shape=(ROWS_PER_GROUP_C, BLOCK_C_C))
    residual = ct.load(residual_ptr, index=(group, col_block),
                       shape=(ROWS_PER_GROUP_C, BLOCK_C_C))
    keep = ct.load(keep_ptr, index=(group, col_block),
                   shape=(ROWS_PER_GROUP_C, BLOCK_C_C))
    keep_f = ct.astype(keep, ct.float32) * SCALE_C
    value = (residual + real) * keep_f

    # Full-row write (in-bounds because ROWS_PER_GROUP divides ROWS and
    # BLOCK_C divides HIDDEN).
    ct.store(out_ptr, index=(group, col_block), tile=value)

    # Partial sum over rows within this group (axis=0), producing (BLOCK_C,)
    partial = ct.sum(value, axis=0)
    partial_2d = ct.reshape(partial, (1, BLOCK_C_C))
    ct.store(partials_ptr, index=(group, col_block), tile=partial_2d)


@ct.kernel
def _final_sum_kernel(
    partials_ptr,   # f32 [GROUPS, HIDDEN]
    sum_out_ptr,    # f32 [HIDDEN]
    GROUPS_C: ct.Constant[int],
    GROUP_BLOCK_C: ct.Constant[int],
    FINAL_BLOCK_C_C: ct.Constant[int],
):
    col_block = ct.bid(0)
    values = ct.load(
        partials_ptr, index=(0, col_block),
        shape=(GROUP_BLOCK_C, FINAL_BLOCK_C_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    group_idx = ct.arange(GROUP_BLOCK_C, dtype=ct.int32)
    group_valid = ct.reshape(group_idx < GROUPS_C, (GROUP_BLOCK_C, 1))
    zero = ct.zeros((GROUP_BLOCK_C, FINAL_BLOCK_C_C), dtype=ct.float32)
    masked = ct.where(group_valid, values, zero)
    total = ct.sum(masked, axis=0)  # (FINAL_BLOCK_C,)
    ct.store(sum_out_ptr, index=(col_block,), tile=total)


@oracle_impl(hardware="B200", point="bdff4e84")
def oracle_forward(inputs):
    real_pair, residual, keep_mask, _shape0, _shape1 = inputs
    device = real_pair.device

    # real_pair: (32, 512, 768, 2). Slice the real component (index 0).
    # Triton reads with stride 2 to skip the imag component; cuTile has no
    # equivalent 4D->3D strided-view load, so we materialize the [ROWS, HIDDEN]
    # real tensor before the kernel (this is a metadata-only slice + copy).
    real_3d = real_pair[..., 0].contiguous().view(ROWS, HIDDEN)
    residual_2d = residual.view(ROWS, HIDDEN)
    keep_2d = keep_mask.view(ROWS, HIDDEN)

    out = torch.empty_strided((ROWS, HIDDEN), (HIDDEN, 1),
                              device=device, dtype=torch.float32)
    partials = torch.empty_strided((GROUPS, HIDDEN), (HIDDEN, 1),
                                    device=device, dtype=torch.float32)
    sum_out = torch.empty_strided((HIDDEN,), (1,),
                                  device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (HIDDEN // BLOCK_C, GROUPS, 1),
        _store_and_partial_sum_kernel,
        (real_3d, residual_2d, keep_2d, out, partials,
         ROWS_PER_GROUP, BLOCK_C, SCALE),
    )

    # GROUP_BLOCK must be a power of 2 >= GROUPS (256).
    group_block = 1 << (GROUPS - 1).bit_length()
    ct.launch(
        stream,
        (HIDDEN // FINAL_BLOCK_C, 1, 1),
        _final_sum_kernel,
        (partials, sum_out, GROUPS, group_block, FINAL_BLOCK_C),
    )

    return out, out.permute(1, 0), sum_out
