"""cuTile port of sum_7cfdb80c54c5: bf16 `where(le, scalar, source)` producer
plus channel reduction.

Fair 2-kernel structure matching the Triton reference:
- Kernel 1: elementwise `where(mask <= 0, scalar, source)` and per-tile
  channel partial sums via ct.sum.
- Kernel 2 (skipped when num_groups==1): finalize partials via ct.sum.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _where_partial_sum_kernel(
    mask_ptr,       # bf16 (rows, C)
    scalar_ptr,     # bf16 (1,)
    source_ptr,     # bf16 (rows, C)
    out_ptr,        # bf16 (rows, C)
    partial_ptr,    # f32  (num_groups, C) OR alias to sum_out when direct
    sum_ptr,        # f32  (C,)
    R_: ct.Constant[int],
    C_: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    STORE_DIRECT_SUM: ct.Constant[bool],
):
    row_group = ct.bid(0)
    c_block = ct.bid(1)

    mask_v = ct.load(mask_ptr, index=(row_group, c_block),
                     shape=(BLOCK_R, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
    source = ct.load(source_ptr, index=(row_group, c_block),
                     shape=(BLOCK_R, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
    scalar_v = ct.load(scalar_ptr, index=(0,), shape=(1,))

    mask_f = ct.astype(mask_v, ct.float32)
    take_scalar = mask_f <= 0.0

    # Broadcast scalar to (BLOCK_R, BLOCK_C).
    scalar_bcast = ct.full((BLOCK_R, BLOCK_C), 0.0, dtype=ct.bfloat16) + \
                    ct.reshape(scalar_v, (1, 1))
    selected = ct.where(take_scalar, scalar_bcast, source)
    ct.store(out_ptr, index=(row_group, c_block), tile=selected)

    row_base = row_group * BLOCK_R
    col_base = c_block * BLOCK_C
    row_idx = ct.arange(BLOCK_R, dtype=ct.int32)
    col_idx = ct.arange(BLOCK_C, dtype=ct.int32)
    row_valid = ct.reshape((row_base + row_idx) < R_, (BLOCK_R, 1))
    col_valid = ct.reshape((col_base + col_idx) < C_, (1, BLOCK_C))
    mask_all = row_valid & col_valid

    selected_f = ct.astype(selected, ct.float32)
    masked = ct.where(mask_all, selected_f, 0.0)
    partial = ct.sum(masked, axis=0)

    if STORE_DIRECT_SUM:
        ct.store(sum_ptr, index=(c_block,), tile=partial)
    else:
        ct.store(partial_ptr, index=(row_group, c_block),
                 tile=ct.reshape(partial, (1, BLOCK_C)))


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
    g_idx = ct.arange(BLOCK_GROUPS, dtype=ct.int32)
    col_idx = ct.arange(BLOCK_C, dtype=ct.int32)
    g_valid = ct.reshape(g_idx < NUM_GROUPS, (BLOCK_GROUPS, 1))
    col_valid = ct.reshape((col_base + col_idx) < C_, (1, BLOCK_C))
    mask = g_valid & col_valid

    masked = ct.where(mask, values_f, 0.0)
    total = ct.sum(masked, axis=0)
    ct.store(sum_ptr, index=(c_block,), tile=total)


def _ceil_pow2(v: int) -> int:
    return 1 << (v - 1).bit_length() if v > 1 else 1


# For each shape point, pick a (BLOCK_R, BLOCK_C, FINAL_BLOCK_C) triple that
# matches Triton's launch BLOCK sizes.
@oracle_impl(hardware="B200", point="3523706e", BLOCK_R=128, BLOCK_C=4, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="14f2271d", BLOCK_R=128, BLOCK_C=32, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="ef334e73", BLOCK_R=128, BLOCK_C=32, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="c9cc6b74", BLOCK_R=64, BLOCK_C=16, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="b96a53f1", BLOCK_R=64, BLOCK_C=16, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="59d19ce6", BLOCK_R=64, BLOCK_C=16, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="eb086b37", BLOCK_R=64, BLOCK_C=16, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="d8c8090e", BLOCK_R=64, BLOCK_C=16, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="f480845f", BLOCK_R=64, BLOCK_C=16, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="f759a34a", BLOCK_R=128, BLOCK_C=16, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="e7edaf95", BLOCK_R=128, BLOCK_C=32, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="a20b8dac", BLOCK_R=64, BLOCK_C=64, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="663c0ae7", BLOCK_R=64, BLOCK_C=32, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="af287ed4", BLOCK_R=32, BLOCK_C=32, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="48c0f941", BLOCK_R=32, BLOCK_C=32, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="e2abc8eb", BLOCK_R=32, BLOCK_C=32, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="c56de08d", BLOCK_R=32, BLOCK_C=32, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="bb80badd", BLOCK_R=32, BLOCK_C=32, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="2d5b3c9a", BLOCK_R=64, BLOCK_C=16, FINAL_BLOCK_C=8)
@oracle_impl(hardware="B200", point="6b77577e", BLOCK_R=64, BLOCK_C=16, FINAL_BLOCK_C=8)
@oracle_impl(hardware="B200", point="feea43c5", BLOCK_R=64, BLOCK_C=32, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="b2b247db", BLOCK_R=64, BLOCK_C=32, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="e61b46b8", BLOCK_R=64, BLOCK_C=32, FINAL_BLOCK_C=16)
def oracle_forward(inputs, *, BLOCK_R, BLOCK_C, FINAL_BLOCK_C):
    mask_input, scalar, source = inputs
    n = int(mask_input.shape[0])
    c = int(mask_input.shape[1])
    h = int(mask_input.shape[2])
    w = int(mask_input.shape[3])
    r = n * h * w

    where_out = torch.empty_strided(
        tuple(mask_input.shape),
        tuple(mask_input.stride()),
        device=mask_input.device,
        dtype=torch.bfloat16,
    )

    # Bring all tensors to channels-last (rows, C).
    cl_stride = (c * h * w, 1, c * w, c)

    def _flat_rc(t):
        s = tuple(int(x) for x in t.stride())
        if s[1] == 1 and s[0] == c * h * w:
            return torch.as_strided(t, (r, c), (c, 1))
        buf = torch.empty_strided((n, c, h, w), cl_stride, device=t.device, dtype=t.dtype)
        buf.copy_(t)
        return torch.as_strided(buf, (r, c), (c, 1))

    mask_2d = _flat_rc(mask_input)
    source_2d = _flat_rc(source)
    # For where_out, we need it channels-last to write through; then copy back to original stride if needed.
    if tuple(where_out.stride()) == cl_stride:
        out_2d = torch.as_strided(where_out, (r, c), (c, 1))
        where_work = None
    else:
        where_work = torch.empty_strided((n, c, h, w), cl_stride,
                                         device=mask_input.device, dtype=torch.bfloat16)
        out_2d = torch.as_strided(where_work, (r, c), (c, 1))

    sum_out = torch.empty((c,), device=mask_input.device, dtype=torch.float32)
    num_groups = ct.cdiv(r, BLOCK_R)
    direct_sum = num_groups == 1
    partial = sum_out if direct_sum else torch.empty((num_groups, c),
                                                     device=mask_input.device,
                                                     dtype=torch.float32)

    stream = torch.cuda.current_stream()
    scalar_1d = scalar.view(1)
    ct.launch(
        stream,
        (num_groups, ct.cdiv(c, BLOCK_C), 1),
        _where_partial_sum_kernel,
        (mask_2d, scalar_1d, source_2d, out_2d, partial, sum_out,
         r, c, BLOCK_R, BLOCK_C, direct_sum),
    )
    if not direct_sum:
        ct.launch(
            stream,
            (ct.cdiv(c, FINAL_BLOCK_C), 1, 1),
            _final_sum_kernel,
            (partial, sum_out, c, num_groups, _ceil_pow2(num_groups), FINAL_BLOCK_C),
        )

    if where_work is not None:
        where_out.copy_(where_work)
    return where_out, sum_out
