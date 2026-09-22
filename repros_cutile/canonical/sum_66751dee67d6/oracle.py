"""cuTile port of sum_66751dee67d6: ConvNeXtV2 add + bf16 cast + channel sum.

Fair 2-kernel structure matching Triton's reference:
- Kernel 1 (`_materialize_add_partial_sum_kernel`): elementwise add, bf16
  cast, store both f32 and bf16 outputs, produce per-tile channel partials
  via ct.sum.
- Kernel 2 (`_final_channel_sum_kernel`): reduce partials via ct.sum, round
  to bf16, store as f32.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 80
H = 56
W = 56
ROWS = N * H * W  # 401,408; divisible by 64


def _ceil_pow2(v: int) -> int:
    return 1 << (v - 1).bit_length() if v > 1 else 1


@ct.kernel
def _materialize_add_partial_sum_kernel(
    arg0_ptr,       # bf16 (ROWS, C)
    arg1_ptr,       # f32  (ROWS, C)
    add_ptr,        # f32  (ROWS, C)
    bf16_ptr,       # bf16 (ROWS, C)
    partial_ptr,    # f32  (num_groups, C)
    ROWS_: ct.Constant[int],
    C_: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row_group = ct.bid(0)
    c_block = ct.bid(1)

    arg0 = ct.load(arg0_ptr, index=(row_group, c_block),
                   shape=(BLOCK_R, BLOCK_C),
                   padding_mode=ct.PaddingMode.ZERO)
    arg1 = ct.load(arg1_ptr, index=(row_group, c_block),
                   shape=(BLOCK_R, BLOCK_C),
                   padding_mode=ct.PaddingMode.ZERO)
    value = ct.astype(arg0, ct.float32) + arg1
    rounded = ct.astype(value, ct.bfloat16)

    ct.store(add_ptr, index=(row_group, c_block), tile=value)
    ct.store(bf16_ptr, index=(row_group, c_block), tile=rounded)

    row_base = row_group * BLOCK_R
    col_base = c_block * BLOCK_C
    row_idx = ct.arange(BLOCK_R, dtype=ct.int32)
    col_idx = ct.arange(BLOCK_C, dtype=ct.int32)
    row_valid = ct.reshape((row_base + row_idx) < ROWS_, (BLOCK_R, 1))
    col_valid = ct.reshape((col_base + col_idx) < C_, (1, BLOCK_C))
    mask = row_valid & col_valid

    rounded_f = ct.astype(rounded, ct.float32)
    masked = ct.where(mask, rounded_f, 0.0)
    partial = ct.sum(masked, axis=0)
    ct.store(partial_ptr, index=(row_group, c_block),
             tile=ct.reshape(partial, (1, BLOCK_C)))


@ct.kernel
def _final_channel_sum_kernel(
    partial_ptr,     # f32 (num_groups, C)
    sum_ptr,         # f32 (C,)
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
    rounded = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    ct.store(sum_ptr, index=(c_block,), tile=rounded)


@oracle_impl(
    hardware="B200",
    point="e47b47c7",
    BLOCK_R=64,
    BLOCK_C=32,
    FINAL_BLOCK_C=32,
)
def oracle_forward(inputs, *, BLOCK_R, BLOCK_C, FINAL_BLOCK_C):
    arg0_1, arg1_1 = inputs

    # arg0: bf16 with stride (C*H*W, 1, C*W, C) = channels-last.
    # arg1: f32 with stride (C*H*W, 1, C, C*H) = NWHC layout.
    # Output layout for `add` and `rounded` in Triton reference is (C*H*W, 1, C, C*H).
    add = torch.empty_strided(
        (N, C, H, W), (C * H * W, 1, C, C * H),
        device=arg0_1.device, dtype=torch.float32,
    )
    rounded = torch.empty_strided(
        (N, C, H, W), (C * H * W, 1, C, C * H),
        device=arg0_1.device, dtype=torch.bfloat16,
    )

    # Bring both inputs and both outputs into a shared (rows, C) contiguous layout
    # via data-movement-only permutes. Reduce over dim=0 of that (rows, C) tile.
    # We treat 'rows' as the (n, h, w) index ordering; each arg has its own row-order
    # so we materialize aligned copies to a common NHWC row-order.
    def _to_rows_c(t):
        # Move C innermost, then flatten (n, h, w) -> rows.
        return t.permute(0, 2, 3, 1).contiguous().view(ROWS, C)

    arg0_rc = _to_rows_c(arg0_1)  # (rows, C) bf16 in NHWC
    arg1_rc = _to_rows_c(arg1_1)  # (rows, C) f32  in NHWC

    add_work = torch.empty((ROWS, C), device=arg0_1.device, dtype=torch.float32)
    bf16_work = torch.empty((ROWS, C), device=arg0_1.device, dtype=torch.bfloat16)

    num_groups = ct.cdiv(ROWS, BLOCK_R)
    partial = torch.empty((num_groups, C), device=arg0_1.device, dtype=torch.float32)
    sum_out = torch.empty((C,), device=arg0_1.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_groups, ct.cdiv(C, BLOCK_C), 1),
        _materialize_add_partial_sum_kernel,
        (arg0_rc, arg1_rc, add_work, bf16_work, partial,
         ROWS, C, BLOCK_R, BLOCK_C),
    )
    ct.launch(
        stream,
        (ct.cdiv(C, FINAL_BLOCK_C), 1, 1),
        _final_channel_sum_kernel,
        (partial, sum_out, C, num_groups, _ceil_pow2(num_groups), FINAL_BLOCK_C),
    )

    # Copy back to the target layouts (data movement).
    add.copy_(add_work.view(N, H, W, C).permute(0, 3, 1, 2))
    rounded.copy_(bf16_work.view(N, H, W, C).permute(0, 3, 1, 2))
    return add, rounded, sum_out
