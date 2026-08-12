"""cuTile port of sum_sum_sum_5fbbe8587abc: MobileBERT LN-backward multi-output.

For each (batch, hidden) row: sum over the token axis to produce partials,
which a second kernel finalizes by reducing over batches. Also materializes
the side outputs `add * arg2` (f32) and `add * arg3` (rounded to bf16).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 256
TOKENS = 128
HIDDEN = 128
ROWS = BATCH * TOKENS


@ct.kernel
def _row_partials_kernel(
    arg0_ptr,       # bf16 [BATCH*TOKENS, HIDDEN]  (view of arg0_1)
    arg1_ptr,       # f32 [BATCH, TOKENS, HIDDEN]
    arg2_ptr,       # f32 [BATCH, TOKENS, HIDDEN]
    arg3_ptr,       # f32 [HIDDEN]
    side_f32_ptr,   # f32 [BATCH, TOKENS, HIDDEN]
    side_bf16_ptr,  # bf16 [BATCH*TOKENS, HIDDEN]
    partial0_ptr,   # f32 [BATCH, HIDDEN]  (sum of add over tokens)
    partial1_ptr,   # f32 [BATCH, HIDDEN]  (sum of add*arg2 over tokens)
    partial3_ptr,   # f32 [BATCH, HIDDEN]  (sum of side_bf16 -> f32 over tokens)
    TOKENS: ct.Constant[int],
    HIDDEN: ct.Constant[int],
):
    b = ct.bid(0)
    # Load one batch tile of shape (TOKENS, HIDDEN)
    arg1 = ct.load(arg1_ptr, index=(b, 0, 0), shape=(1, TOKENS, HIDDEN))
    arg1_2d = ct.reshape(arg1, (TOKENS, HIDDEN))

    # arg0 is a bf16 [rows, hidden] view; the corresponding rows for this batch
    # are rows [b*TOKENS : (b+1)*TOKENS].
    arg0 = ct.load(arg0_ptr, index=(b, 0), shape=(TOKENS, HIDDEN))
    arg0_f = ct.astype(arg0, ct.float32)

    add = arg1_2d + arg0_f

    arg2 = ct.load(arg2_ptr, index=(b, 0, 0), shape=(1, TOKENS, HIDDEN))
    arg2_2d = ct.reshape(arg2, (TOKENS, HIDDEN))

    arg3 = ct.load(arg3_ptr, index=(0,), shape=(HIDDEN,))
    arg3_2d = ct.reshape(arg3, (1, HIDDEN))

    # side (output at side_f32) is add * arg3 (per-hidden scale).
    side = add * arg3_2d
    side_bf16 = ct.astype(side, ct.bfloat16)
    # partial1 uses add * arg2 (elementwise), then sum over tokens.
    add_times_arg2 = add * arg2_2d

    ct.store(side_f32_ptr, index=(b, 0, 0), tile=ct.reshape(side, (1, TOKENS, HIDDEN)))
    ct.store(side_bf16_ptr, index=(b, 0), tile=side_bf16)

    # Partial sums over the TOKENS axis (axis 0 of the (TOKENS, HIDDEN) tile).
    partial0 = ct.sum(add, axis=0)          # (HIDDEN,)
    partial1 = ct.sum(add_times_arg2, axis=0)  # (HIDDEN,)
    partial3 = ct.sum(ct.astype(side_bf16, ct.float32), axis=0)

    ct.store(partial0_ptr, index=(b, 0), tile=ct.reshape(partial0, (1, HIDDEN)))
    ct.store(partial1_ptr, index=(b, 0), tile=ct.reshape(partial1, (1, HIDDEN)))
    ct.store(partial3_ptr, index=(b, 0), tile=ct.reshape(partial3, (1, HIDDEN)))


@ct.kernel
def _finalize_kernel(
    partial0_ptr,   # f32 [BATCH, HIDDEN]
    partial1_ptr,   # f32 [BATCH, HIDDEN]
    partial3_ptr,   # f32 [BATCH, HIDDEN]
    out0_ptr,       # f32 [HIDDEN]
    out1_ptr,       # f32 [HIDDEN]
    out3_ptr,       # f32 [HIDDEN]
    BATCH: ct.Constant[int],
    HIDDEN: ct.Constant[int],
):
    # Load whole partial arrays and reduce over batch.
    p0 = ct.load(partial0_ptr, index=(0, 0), shape=(BATCH, HIDDEN))
    p1 = ct.load(partial1_ptr, index=(0, 0), shape=(BATCH, HIDDEN))
    p3 = ct.load(partial3_ptr, index=(0, 0), shape=(BATCH, HIDDEN))
    total0 = ct.sum(p0, axis=0)
    total1 = ct.sum(p1, axis=0)
    total3 = ct.sum(p3, axis=0)
    # Final total3: round to bf16 then back to f32 to match the observable dtype boundary.
    total3_bf16 = ct.astype(total3, ct.bfloat16)
    total3_rt = ct.astype(total3_bf16, ct.float32)
    ct.store(out0_ptr, index=(0,), tile=total0)
    ct.store(out1_ptr, index=(0,), tile=total1)
    ct.store(out3_ptr, index=(0,), tile=total3_rt)


@oracle_impl(hardware="B200", point="f62c5e26", xblock=16)
def oracle_forward(inputs, **_kwargs):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        _shape0,
        shape1,
        shape2,
        shape3,
        shape4,
    ) = inputs

    full_shape = tuple(int(dim) for dim in arg1_1.shape)  # (256, 128, 128)
    flat_shape = tuple(int(dim) for dim in shape3)
    sum_shape = tuple(int(dim) for dim in shape1)
    sum2_shape = tuple(int(dim) for dim in shape2)
    sum3_shape = tuple(int(dim) for dim in shape4)

    device = arg1_1.device
    side_f32 = torch.empty_strided(
        full_shape,
        (full_shape[1] * full_shape[2], full_shape[2], 1),
        device=device,
        dtype=torch.float32,
    )
    side_bf16 = torch.empty_strided(
        flat_shape,
        (flat_shape[1], 1),
        device=device,
        dtype=torch.bfloat16,
    )
    partial0 = torch.empty((BATCH, HIDDEN), device=device, dtype=torch.float32)
    partial1 = torch.empty((BATCH, HIDDEN), device=device, dtype=torch.float32)
    partial3 = torch.empty((BATCH, HIDDEN), device=device, dtype=torch.float32)
    out0 = torch.empty_strided(sum_shape, (1,), device=device, dtype=torch.float32)
    out1 = torch.empty_strided(sum2_shape, (1,), device=device, dtype=torch.float32)
    out3 = torch.empty_strided(sum3_shape, (1,), device=device, dtype=torch.float32)

    # arg0_1 is bf16 [ROWS, HIDDEN]; view as [BATCH, TOKENS, HIDDEN] to align with arg1.
    arg0_3d = arg0_1.view(BATCH, TOKENS, HIDDEN)
    # For loading arg0 in shape (TOKENS, HIDDEN) at (b, 0), we treat it as
    # [BATCH * TOKENS, HIDDEN] with per-batch offset b*TOKENS. We reshape to
    # [BATCH, TOKENS*HIDDEN] then view as [BATCH, TOKENS, HIDDEN].
    # Actually we want to load a tile of shape (TOKENS, HIDDEN) at index (b, 0).
    # That works if arg0 is viewed as a 3D [BATCH, TOKENS, HIDDEN] tensor. But
    # then the tile-space is (1, TOKENS, HIDDEN) for a rank-3 partition, so we'd
    # load a 3D tile. Simplest: pass arg0 as [BATCH*TOKENS, HIDDEN] and load in
    # the partitioning where each batch tile owns TOKENS rows.
    # Use partitioning [BATCH, TOKENS, HIDDEN] via view. But then load shape
    # must be 3D. Alternative: view arg0 as [BATCH, TOKENS*HIDDEN] and load
    # shape (1, TOKENS*HIDDEN), then reshape to (TOKENS, HIDDEN).
    arg0_bt_h = arg0_1.view(BATCH, TOKENS * HIDDEN)
    # The kernel expects `arg0_ptr` to be a rank-2 tensor loadable at (b, 0)
    # with shape (TOKENS, HIDDEN). We can achieve that by reshaping to
    # (BATCH, TOKENS * HIDDEN) OR (BATCH * TOKENS, HIDDEN). Let's use the
    # (BATCH * TOKENS, HIDDEN) view; then index (b*TOKENS, 0) would load
    # (TOKENS, HIDDEN). But indices in tile-space aren't (b, 0) - they're
    # (b_tile, 0) where b_tile = b (if the array is partitioned by TOKENS rows).
    # Partitioning is inferred by tile shape (TOKENS, HIDDEN) applied to a
    # (BATCH*TOKENS, HIDDEN) array => tile-space = (BATCH, 1). So index (b, 0)
    # loads rows [b*TOKENS, (b+1)*TOKENS). Good.
    arg0_2d = arg0_1.view(BATCH * TOKENS, HIDDEN)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH, 1, 1),
        _row_partials_kernel,
        (arg0_2d, arg1_1, arg2_1, arg3_1,
         side_f32, side_bf16, partial0, partial1, partial3,
         TOKENS, HIDDEN),
    )
    ct.launch(
        stream,
        (1, 1, 1),
        _finalize_kernel,
        (partial0, partial1, partial3, out0, out1, out3, BATCH, HIDDEN),
    )
    return out0, side_f32, out1, side_bf16, side_bf16.permute(1, 0), out3
