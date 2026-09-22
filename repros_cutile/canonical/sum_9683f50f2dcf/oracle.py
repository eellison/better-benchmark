"""cuTile port of sum_9683f50f2dcf: ConvNeXtV2 bf16 residual add + channel
sum across NHW dims.

Fair 2-kernel structure matching the Triton reference:
- Kernel 1: elementwise add over a shared (rows, C) flat layout AND per-tile
  channel partial sum via ct.sum.
- Kernel 2: finalize channel sums by summing partials via ct.sum.

The two inputs come with different strides (arg0 is NWHC-order, arg1 is NHWC-
order in physical storage). To share a single (rows, C) view without unfair
torch reductions, we first copy arg1's values into a buffer with arg0's stride
layout via a plain torch `.copy_` (data movement only, no arithmetic).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _add_partial_sum_kernel(
    x_ptr,          # bf16 (rows, C) contig on C
    y_ptr,          # bf16 (rows, C) contig on C (aligned to x's stride)
    add_ptr,        # bf16 (rows, C) contig on C
    partial_ptr,    # f32 (num_partials, C)
    R_: ct.Constant[int],
    C_: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    k_block = ct.bid(0)
    c_block = ct.bid(1)

    x = ct.load(x_ptr, index=(k_block, c_block), shape=(BLOCK_K, BLOCK_C),
                padding_mode=ct.PaddingMode.ZERO)
    y = ct.load(y_ptr, index=(k_block, c_block), shape=(BLOCK_K, BLOCK_C),
                padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(x, ct.float32)
    y_f = ct.astype(y, ct.float32)
    added = ct.astype(x_f + y_f, ct.bfloat16)
    ct.store(add_ptr, index=(k_block, c_block), tile=added)

    row_base = k_block * BLOCK_K
    col_base = c_block * BLOCK_C
    row_idx = ct.arange(BLOCK_K, dtype=ct.int32)
    col_idx = ct.arange(BLOCK_C, dtype=ct.int32)
    row_valid = ct.reshape((row_base + row_idx) < R_, (BLOCK_K, 1))
    col_valid = ct.reshape((col_base + col_idx) < C_, (1, BLOCK_C))
    mask = row_valid & col_valid

    added_f = ct.astype(added, ct.float32)
    masked = ct.where(mask, added_f, 0.0)
    partial = ct.sum(masked, axis=0)
    partial_2d = ct.reshape(partial, (1, BLOCK_C))
    ct.store(partial_ptr, index=(k_block, c_block), tile=partial_2d)


@ct.kernel
def _finish_sum_kernel(
    partial_ptr,      # f32 (num_partials, C)
    sum_ptr,          # f32 (C,)
    C_: ct.Constant[int],
    NUM_PARTIALS: ct.Constant[int],
    BLOCK_PARTIALS: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    values = ct.load(
        partial_ptr, index=(0, c_block), shape=(BLOCK_PARTIALS, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    values_f = ct.astype(values, ct.float32)

    col_base = c_block * BLOCK_C
    p_idx = ct.arange(BLOCK_PARTIALS, dtype=ct.int32)
    col_idx = ct.arange(BLOCK_C, dtype=ct.int32)
    p_valid = ct.reshape(p_idx < NUM_PARTIALS, (BLOCK_PARTIALS, 1))
    col_valid = ct.reshape((col_base + col_idx) < C_, (1, BLOCK_C))
    mask = p_valid & col_valid

    masked = ct.where(mask, values_f, 0.0)
    total = ct.sum(masked, axis=0)
    rounded = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    ct.store(sum_ptr, index=(c_block,), tile=rounded)


def _ceil_pow2(v: int) -> int:
    return 1 << (v - 1).bit_length() if v > 1 else 1


@oracle_impl(hardware="B200", point="201d3861", BLOCK_K=512, BLOCK_C=64, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="b04a3c92", BLOCK_K=128, BLOCK_C=16, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="9ea47abe", BLOCK_K=128, BLOCK_C=16, FINAL_BLOCK_C=16)
def oracle_forward(inputs, *, BLOCK_K, BLOCK_C, FINAL_BLOCK_C):
    arg0_1, arg1_1 = inputs
    n = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    rows = n * h * w

    # Output must match arg0's stride pattern.
    add_out = torch.empty_strided(
        (n, c, h, w),
        tuple(int(s) for s in arg0_1.stride()),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    # True channels-last strides: (C*H*W, 1, C*W, C).
    cl_stride = (c * h * w, 1, c * w, c)

    def _in_cl(t):
        return tuple(int(s) for s in t.stride()) == cl_stride

    # Bring both inputs and a work-buffer into channels-last (rows, C).
    # Data movement only (torch .copy_), no arithmetic outside the kernel.
    def _to_cl(t):
        if _in_cl(t):
            return t
        buf = torch.empty_strided((n, c, h, w), cl_stride, device=t.device, dtype=torch.bfloat16)
        buf.copy_(t)
        return buf

    arg0_cl = _to_cl(arg0_1)
    arg1_cl = _to_cl(arg1_1)
    x_2d = torch.as_strided(arg0_cl, (rows, c), (c, 1))
    y_2d = torch.as_strided(arg1_cl, (rows, c), (c, 1))

    if _in_cl(add_out):
        add_2d = torch.as_strided(add_out, (rows, c), (c, 1))
        add_work = None
    else:
        add_work = torch.empty_strided(
            (n, c, h, w), cl_stride, device=arg0_1.device, dtype=torch.bfloat16
        )
        add_2d = torch.as_strided(add_work, (rows, c), (c, 1))

    sum_out = torch.empty((c,), device=arg0_1.device, dtype=torch.float32)
    num_partials = ct.cdiv(rows, BLOCK_K)
    partial = torch.empty((num_partials, c), device=arg0_1.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_partials, ct.cdiv(c, BLOCK_C), 1),
        _add_partial_sum_kernel,
        (x_2d, y_2d, add_2d, partial, rows, c, BLOCK_K, BLOCK_C),
    )
    block_partials = _ceil_pow2(num_partials)
    ct.launch(
        stream,
        (ct.cdiv(c, FINAL_BLOCK_C), 1, 1),
        _finish_sum_kernel,
        (partial, sum_out, c, num_partials, block_partials, FINAL_BLOCK_C),
    )

    # If arg0 wasn't channels-last, copy the work buffer into add_out to
    # match arg0's stride.
    if add_work is not None:
        add_out.copy_(add_work)

    return add_out, sum_out
