"""cuTile port of sum_4a4493837e6e: SiLU-backward materialize + channel sum.

The input is channels-last bf16 (strides (C*H*W, 1, C*W, C)); so linearly the
memory is `[N*H*W, C]`. Fair 2-kernel structure mirroring the Triton reference:
- Kernel 1: materialize SiLU-backward, store bf16, produce per-tile channel
  partial sums with ct.sum.
- Kernel 2: finalize by ct.sum across partials.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _materialize_partial_kernel(
    grad_ptr,       # bf16 (rows, C)
    x_ptr,          # bf16 (rows, C)
    out_ptr,        # bf16 (rows, C)
    partial_ptr,    # f32 (num_groups, C) OR alias of sum_out when direct
    sum_ptr,        # f32 (C,)
    R_: ct.Constant[int],
    C_: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    STORE_DIRECT_SUM: ct.Constant[bool],
):
    row_group = ct.bid(0)
    c_block = ct.bid(1)

    grad = ct.load(grad_ptr, index=(row_group, c_block), shape=(BLOCK_R, BLOCK_C),
                   padding_mode=ct.PaddingMode.ZERO)
    x = ct.load(x_ptr, index=(row_group, c_block), shape=(BLOCK_R, BLOCK_C),
                padding_mode=ct.PaddingMode.ZERO)
    grad_f = ct.astype(grad, ct.float32)
    x_f = ct.astype(x, ct.float32)
    sig = 1.0 / (1.0 + ct.exp(-x_f))
    value = grad_f * sig * (x_f * (1.0 - sig) + 1.0)
    value_bf16 = ct.astype(value, ct.bfloat16)

    ct.store(out_ptr, index=(row_group, c_block), tile=value_bf16)

    # Row/col mask for the sum (OOB elements contribute 0).
    row_base = row_group * BLOCK_R
    col_base = c_block * BLOCK_C
    row_idx = ct.arange(BLOCK_R, dtype=ct.int32)
    col_idx = ct.arange(BLOCK_C, dtype=ct.int32)
    row_valid = ct.reshape((row_base + row_idx) < R_, (BLOCK_R, 1))
    col_valid = ct.reshape((col_base + col_idx) < C_, (1, BLOCK_C))
    mask = row_valid & col_valid

    value_bf16_f = ct.astype(value_bf16, ct.float32)
    masked = ct.where(mask, value_bf16_f, 0.0)
    partial = ct.sum(masked, axis=0)  # (BLOCK_C,)

    if STORE_DIRECT_SUM:
        rounded = ct.astype(ct.astype(partial, ct.bfloat16), ct.float32)
        ct.store(sum_ptr, index=(c_block,), tile=rounded)
    else:
        partial_2d = ct.reshape(partial, (1, BLOCK_C))
        ct.store(partial_ptr, index=(row_group, c_block), tile=partial_2d)


@ct.kernel
def _final_sum_kernel(
    partial_ptr,      # f32 (num_groups, C)
    sum_ptr,          # f32 (C,)
    C_: ct.Constant[int],
    NUM_GROUPS: ct.Constant[int],
    BLOCK_GROUPS: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    values = ct.load(
        partial_ptr, index=(0, c_block), shape=(BLOCK_GROUPS, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    values_f = ct.astype(values, ct.float32)

    col_base = c_block * BLOCK_C
    group_idx = ct.arange(BLOCK_GROUPS, dtype=ct.int32)
    col_idx = ct.arange(BLOCK_C, dtype=ct.int32)
    group_valid = ct.reshape(group_idx < NUM_GROUPS, (BLOCK_GROUPS, 1))
    col_valid = ct.reshape((col_base + col_idx) < C_, (1, BLOCK_C))
    mask = group_valid & col_valid

    masked = ct.where(mask, values_f, 0.0)
    total = ct.sum(masked, axis=0)
    rounded = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    ct.store(sum_ptr, index=(c_block,), tile=rounded)


def _ceil_pow2(v: int) -> int:
    return 1 << (v - 1).bit_length() if v > 1 else 1


@oracle_impl(hardware="B200", point="c5cf0dd3", BLOCK_R=64, BLOCK_C=16, FINAL_BLOCK_C=4)
@oracle_impl(hardware="B200", point="f40368ca", BLOCK_R=64, BLOCK_C=32, FINAL_BLOCK_C=4)
@oracle_impl(hardware="B200", point="934b0f73", BLOCK_R=64, BLOCK_C=32, FINAL_BLOCK_C=4)
@oracle_impl(hardware="B200", point="96449bbf", BLOCK_R=64, BLOCK_C=32, FINAL_BLOCK_C=8)
@oracle_impl(hardware="B200", point="d34f6e69", BLOCK_R=64, BLOCK_C=32, FINAL_BLOCK_C=8)
@oracle_impl(hardware="B200", point="13a2d815", BLOCK_R=64, BLOCK_C=16, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="f961de61", BLOCK_R=64, BLOCK_C=32, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="7c24d0c8", BLOCK_R=64, BLOCK_C=32, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="5c7adf25", BLOCK_R=64, BLOCK_C=32, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="03f3f1e3", BLOCK_R=128, BLOCK_C=16, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="9926e6d2", BLOCK_R=128, BLOCK_C=16, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="5045fc42", BLOCK_R=128, BLOCK_C=16, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="6ced833e", BLOCK_R=128, BLOCK_C=16, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="cadd9933", BLOCK_R=128, BLOCK_C=16, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="00335e2b", BLOCK_R=128, BLOCK_C=16, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="78ac4aa7", BLOCK_R=128, BLOCK_C=16, FINAL_BLOCK_C=16)
def oracle_forward(inputs, *, BLOCK_R: int, BLOCK_C: int, FINAL_BLOCK_C: int):
    grad, x = inputs
    n, c, h, w = (int(grad.shape[i]) for i in range(4))
    rows = n * h * w

    # Materialize view in channels-last physical layout.
    out = torch.empty_strided(
        tuple(grad.shape), tuple(grad.stride()), device=grad.device, dtype=torch.bfloat16
    )
    grad_2d = torch.as_strided(grad, (rows, c), (c, 1))
    x_2d = torch.as_strided(x, (rows, c), (c, 1))
    out_2d = torch.as_strided(out, (rows, c), (c, 1))

    sum_out = torch.empty((c,), device=grad.device, dtype=torch.float32)
    num_groups = ct.cdiv(rows, BLOCK_R)
    direct_sum = num_groups == 1
    if direct_sum:
        partial = sum_out
    else:
        partial = torch.empty((num_groups, c), device=grad.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_groups, ct.cdiv(c, BLOCK_C), 1),
        _materialize_partial_kernel,
        (grad_2d, x_2d, out_2d, partial, sum_out, rows, c, BLOCK_R, BLOCK_C, direct_sum),
    )

    if not direct_sum:
        block_groups = _ceil_pow2(num_groups)
        ct.launch(
            stream,
            (ct.cdiv(c, FINAL_BLOCK_C), 1, 1),
            _final_sum_kernel,
            (partial, sum_out, c, num_groups, block_groups, FINAL_BLOCK_C),
        )

    return out, sum_out
