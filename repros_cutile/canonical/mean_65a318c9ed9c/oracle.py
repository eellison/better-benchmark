"""cuTile port of mean_65a318c9ed9c (SCHEDULER_FUSION): MobileNetV3
BN affine + HardSwish + spatial mean.

The Triton kernel processes rows = batch * channel, and each row contains
HW spatial elements from a channels-last layout. Channels count (960/672/480)
is not a power of 2, HW is 49 or 196. We tile HW to the next power of 2 with
padding_mode.ZERO for loads and use ct.scatter for the mean writeback while
using ct.load / ct.store on the channels-last output where the tile size
matches. Since output layout matches input, we reshape (B, C, H, W) as
(B*C, HW) via a permutation and process rows in this space.

Actually simpler: for each (batch, channel), the channels-last address is:
  base = batch * (C * HW) + channel   (element offset in flat NHWC memory)
  stride = C (each next HW element is C apart)
We tile 1 (batch, channel) per program and compute the HW-length row using
ct.gather / ct.scatter with computed indices.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 512
EPS = 1.0e-5


@ct.kernel
def _bn_hardswish_mean_kernel(
    running_mean_ptr,   # bf16 [C]
    x_ptr,              # bf16 flat [BATCH * C * HW]
    running_var_ptr,    # bf16 [C]
    weight_ptr,         # bf16 [C]
    bias_ptr,           # bf16 [C]
    out_ptr,            # bf16 flat [BATCH * C * HW]
    mean_out_ptr,       # bf16 flat [BATCH * C]
    C: ct.Constant[int],
    HW: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    row = ct.bid(0)  # one (batch, channel) per program
    channel_ids = ct.arange(1, dtype=ct.int64) + (row - (row // C) * C)
    batch_ids = ct.arange(1, dtype=ct.int64) + (row // C)
    channel = row - (row // C) * C
    batch = row // C

    # Load per-channel scalars via gather with 1-element index.
    ch_1 = ct.arange(1, dtype=ct.int64) + channel
    running_mean_bf = ct.gather(running_mean_ptr, ch_1)
    running_var_bf = ct.gather(running_var_ptr, ch_1)
    weight_bf = ct.gather(weight_ptr, ch_1)
    bias_bf = ct.gather(bias_ptr, ch_1)
    running_mean = ct.astype(running_mean_bf, ct.float32)
    running_var = ct.astype(running_var_bf, ct.float32)
    weight = ct.astype(weight_bf, ct.float32)
    bias = ct.astype(bias_bf, ct.float32)

    # Now iterate BLOCK_HW elements at a time.
    hw = ct.arange(BLOCK_HW, dtype=ct.int64)
    hw_mask = hw < HW
    # Element offset in x/out (channels-last flat): batch*(C*HW) + channel + hw*C
    base = batch * (C * HW) + channel
    offsets = ct.full((BLOCK_HW,), 0, dtype=ct.int64) + base + hw * C
    zero64 = ct.zeros((BLOCK_HW,), dtype=ct.int64)
    safe_off = ct.where(hw_mask, offsets, zero64)

    x_bf = ct.gather(x_ptr, safe_off)
    x = ct.astype(x_bf, ct.float32)

    sqrt_var = ct.sqrt(running_var + EPS)
    invstd = 1.0 / sqrt_var
    centered = x - running_mean
    normalized = centered * invstd
    affine = normalized * weight + bias
    relu6 = affine + 3.0
    zero_bf_tile = ct.full((BLOCK_HW,), 0.0, dtype=ct.float32)
    six_tile = ct.full((BLOCK_HW,), 6.0, dtype=ct.float32)
    relu6 = ct.where(relu6 < 0.0, zero_bf_tile, relu6)
    relu6 = ct.where(relu6 > 6.0, six_tile, relu6)
    hardswish = (affine * relu6) * (1.0 / 6.0)
    out_bf16 = ct.astype(hardswish, ct.bfloat16)

    ct.scatter(out_ptr, safe_off, out_bf16, mask=hw_mask)

    # Compute mean over HW valid elements. Zero the OOB.
    contrib = ct.where(hw_mask, ct.astype(out_bf16, ct.float32), zero_bf_tile)
    mean_sum = ct.sum(contrib)
    mean_value = mean_sum * (1.0 / HW)
    # Store scalar mean into mean_out_ptr at index (batch * C + channel).
    # ct.store on a single scalar: use a 1-element tile.
    mean_out_1 = ct.astype(ct.full((1,), mean_value, dtype=ct.float32), ct.bfloat16)
    ct.scatter(mean_out_ptr, ch_1 + batch * C, mean_out_1)


def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="3e244c1d", C=960, HW=49)
@oracle_impl(hardware="B200", point="57e42e70", C=672, HW=49)
@oracle_impl(hardware="B200", point="c37163dc", C=672, HW=196)
@oracle_impl(hardware="B200", point="86f01d63", C=480, HW=196)
def oracle_forward(inputs, *, C: int, HW: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = inputs
    height = int(arg1_1.shape[2])
    width = int(arg1_1.shape[3])
    out = torch.empty_strided(
        (BATCH, C, height, width),
        (C * HW, 1, width * C, C),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    mean = torch.empty_strided(
        (BATCH, C, 1, 1),
        (C, 1, 1, 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )

    # Input is NHWC-strided; permute to (N, H, W, C) gives contiguous memory,
    # then flatten. Output has the same NHWC storage layout.
    x_flat = arg1_1.permute(0, 2, 3, 1).reshape(-1)
    out_flat = out.permute(0, 2, 3, 1).reshape(-1)
    mean_flat = mean.view(-1)

    BLOCK_HW = _next_power_of_2(HW)
    if BLOCK_HW > 512:
        BLOCK_HW = 512

    total_rows = BATCH * C

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (total_rows, 1, 1),
        _bn_hardswish_mean_kernel,
        (arg0_1, x_flat, arg2_1, arg3_1, arg4_1, out_flat, mean_flat, C, HW, BLOCK_HW),
    )
    return out, mean
