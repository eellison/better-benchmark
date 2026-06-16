"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Demucs channel-slice scope, returning the metadata-only bf16 `arg0_1[:,1:5,:,0:382788]` view and directly reducing those four sliced channels into the contiguous bf16 `[8,2,382788]` sum output, whereas Inductor already avoids materializing the slice view and the remaining work is the mandatory four bf16 channel reads plus bf16 sum store. Inductor cannot materially improve this local repro with scheduler fusion, scatter-reduce, split-K, algebraic elimination, or a new pattern because the direct slice reduction is the required memory-traffic floor; the fix is BANDWIDTH_BOUND: only broader reduction bandwidth or launch-overhead improvements would move both paths."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 8
CHANNELS_IN = 5
PAIR = 2
IN_WIDTH = 426888
OUT_WIDTH = 382788
CHANNEL_STRIDE = PAIR * IN_WIDTH
BATCH_STRIDE = CHANNELS_IN * CHANNEL_STRIDE
OUT_ROWS = BATCH * PAIR


@triton.jit
def _slice_sum_kernel(
    x_ptr,
    out_ptr,
    OUT_WIDTH_: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = (rows < 16) & (cols < OUT_WIDTH_)

    batch = rows // 2
    pair_index = rows - batch * 2
    base = batch * 4268880 + pair_index * 426888 + cols

    c1 = tl.load(
        x_ptr + base + 853776,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    c2 = tl.load(
        x_ptr + base + 2 * 853776,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    c3 = tl.load(
        x_ptr + base + 3 * 853776,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    c4 = tl.load(
        x_ptr + base + 4 * 853776,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    total = c1 + c2 + c3 + c4
    tl.store(
        out_ptr + rows * OUT_WIDTH_ + cols,
        total.to(tl.bfloat16),
        mask=mask,
    )


@oracle_impl(hardware="B200", point="0dd51176", ROW_BLOCK=4, BLOCK_N=512, num_warps=8)
def oracle_forward(inputs, *, ROW_BLOCK: int, BLOCK_N: int, num_warps: int):
    (arg0_1,) = inputs
    slice_out = torch.as_strided(
        arg0_1,
        (BATCH, 4, PAIR, OUT_WIDTH),
        (BATCH_STRIDE, CHANNEL_STRIDE, IN_WIDTH, 1),
        CHANNEL_STRIDE,
    )
    sum_out = torch.empty_strided(
        (BATCH, PAIR, OUT_WIDTH),
        (PAIR * OUT_WIDTH, OUT_WIDTH, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _slice_sum_kernel[(triton.cdiv(OUT_ROWS, ROW_BLOCK), triton.cdiv(OUT_WIDTH, BLOCK_N))](
        arg0_1,
        sum_out,
        OUT_WIDTH_=OUT_WIDTH,
        ROW_BLOCK=ROW_BLOCK,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return slice_out, sum_out
