"""cuTile port of pointwise_9b95b7531c94: VGG16 ReLU + 2x2 maxpool + view + le mask.

One cuTile kernel emits: pooled values, offsets (int8), flat values, and
the input-shaped `le0` mask. NaN handling: relu preserves NaN; maxpool
takes candidate if candidate > best or (best == best and candidate != candidate).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 64
CHANNELS = 512
H_IN = 14
W_IN = 14
H_OUT = 7
W_OUT = 7
INPUT_PLANE = H_IN * W_IN
OUTPUT_PLANE = H_OUT * W_OUT
TOTAL_OUTPUTS = BATCH * CHANNELS * OUTPUT_PLANE
BLOCK = 128


@ct.kernel
def _relu_maxpool_kernel(
    input_ptr,       # bf16 [BATCH*CHANNELS*H_IN*W_IN]
    values_ptr,      # bf16 [BATCH*CHANNELS*OUTPUT_PLANE]
    offsets_ptr,     # i8   [BATCH*CHANNELS*OUTPUT_PLANE]
    flat_values_ptr, # bf16 [BATCH*CHANNELS*OUTPUT_PLANE]
    le_mask_ptr,     # bool [BATCH*CHANNELS*H_IN*W_IN]
    TOTAL: ct.Constant[int],
    BLOCK_: ct.Constant[int],
):
    pid = ct.bid(0)
    idx = ct.arange(BLOCK_, dtype=ct.int32) + pid * BLOCK_
    active = idx < TOTAL

    out_w = idx % 7
    tmp = idx // 7
    out_h = tmp % 7
    plane = tmp // 7

    # For each output element, four input positions in a 2x2 block.
    input_base = plane * 196 + out_h * 28 + out_w * 2
    # Note: cuTile doesn't have arbitrary gather in the same form as Triton,
    # but the input_base indices are per-thread. Use scatter-load via
    # advanced indexing: build the linear indices and use ct.gather.
    idx00 = input_base
    idx01 = input_base + 1
    idx10 = input_base + 14
    idx11 = input_base + 15

    x00 = ct.gather(input_ptr, idx00)
    x01 = ct.gather(input_ptr, idx01)
    x10 = ct.gather(input_ptr, idx10)
    x11 = ct.gather(input_ptr, idx11)

    x00_f = ct.astype(x00, ct.float32)
    x01_f = ct.astype(x01, ct.float32)
    x10_f = ct.astype(x10, ct.float32)
    x11_f = ct.astype(x11, ct.float32)

    # relu preserving NaN: where (x > 0 or x != x), x, else 0
    zero_f = ct.zeros((BLOCK_,), dtype=ct.float32)
    r00 = ct.where((x00_f > zero_f) | (x00_f != x00_f), x00_f, zero_f)
    r01 = ct.where((x01_f > zero_f) | (x01_f != x01_f), x01_f, zero_f)
    r10 = ct.where((x10_f > zero_f) | (x10_f != x10_f), x10_f, zero_f)
    r11 = ct.where((x11_f > zero_f) | (x11_f != x11_f), x11_f, zero_f)

    # Track best value and offset. Start with position 0.
    best = r00
    best_offset = ct.zeros((BLOCK_,), dtype=ct.int32)

    # candidate 01, offset 1
    take = (r01 > best) | ((best == best) & (r01 != r01))
    best = ct.where(take, r01, best)
    best_offset = ct.where(take, ct.full((BLOCK_,), 1, dtype=ct.int32), best_offset)

    # candidate 10, offset 2
    take = (r10 > best) | ((best == best) & (r10 != r10))
    best = ct.where(take, r10, best)
    best_offset = ct.where(take, ct.full((BLOCK_,), 2, dtype=ct.int32), best_offset)

    # candidate 11, offset 3
    take = (r11 > best) | ((best == best) & (r11 != r11))
    best = ct.where(take, r11, best)
    best_offset = ct.where(take, ct.full((BLOCK_,), 3, dtype=ct.int32), best_offset)

    best_bf = ct.astype(best, ct.bfloat16)
    offset_i8 = ct.astype(best_offset, ct.int8)

    ct.scatter(values_ptr, idx, best_bf, mask=active)
    ct.scatter(offsets_ptr, idx, offset_i8, mask=active)
    ct.scatter(flat_values_ptr, idx, best_bf, mask=active)

    # Store le mask: for each of the 4 input positions, x <= 0
    le00 = x00_f <= 0.0
    le01 = x01_f <= 0.0
    le10 = x10_f <= 0.0
    le11 = x11_f <= 0.0
    ct.scatter(le_mask_ptr, idx00, le00, mask=active)
    ct.scatter(le_mask_ptr, idx01, le01, mask=active)
    ct.scatter(le_mask_ptr, idx10, le10, mask=active)
    ct.scatter(le_mask_ptr, idx11, le11, mask=active)


@oracle_impl(hardware="B200", point="2c0e267a")
def oracle_forward(inputs):
    arg0_1 = inputs[0]
    device = arg0_1.device

    values = torch.empty_strided(
        (BATCH, CHANNELS, H_OUT, W_OUT),
        (CHANNELS * OUTPUT_PLANE, OUTPUT_PLANE, W_OUT, 1),
        device=device, dtype=arg0_1.dtype,
    )
    offsets = torch.empty_strided(
        (BATCH, CHANNELS, H_OUT, W_OUT),
        (CHANNELS * OUTPUT_PLANE, OUTPUT_PLANE, W_OUT, 1),
        device=device, dtype=torch.int8,
    )
    flat_values = torch.empty_strided(
        (BATCH, CHANNELS * OUTPUT_PLANE),
        (CHANNELS * OUTPUT_PLANE, 1),
        device=device, dtype=arg0_1.dtype,
    )
    le_mask = torch.empty_strided(
        (BATCH, CHANNELS, H_IN, W_IN),
        (CHANNELS * INPUT_PLANE, INPUT_PLANE, W_IN, 1),
        device=device, dtype=torch.bool,
    )

    # Flatten to 1D views for scatter/gather.
    input_flat = arg0_1.contiguous().view(-1)
    values_flat = values.view(-1)
    offsets_flat = offsets.view(-1)
    flat_values_flat = flat_values.view(-1)
    le_flat = le_mask.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ct.cdiv(TOTAL_OUTPUTS, BLOCK), 1, 1), _relu_maxpool_kernel,
        (input_flat, values_flat, offsets_flat, flat_values_flat, le_flat,
         TOTAL_OUTPUTS, BLOCK),
    )
    return values, offsets, flat_values, le_mask
