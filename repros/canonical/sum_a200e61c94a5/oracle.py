"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete Demucs bf16 reshape-plus-channel-sum scope by returning the `[4,8,371372]` view as an alias of the input storage and reducing each channel through explicit split-K partials over the batch/time domain, whereas Inductor lowers the metadata view and the large dim `[0,2]` reduction through generic reduction scheduling; Inductor cannot do this today because its scheduler/codegen does not specialize this fixed Demucs layout into a compact channel-wise split reduction while preserving the returned alias view and compiled-path f32 accumulator; the fix is COOPERATIVE_SPLIT_K: add a guarded layout-aware channel reduction template for static reshape views that splits the long batch/time dimension into cooperative partials and finalizes each channel directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 4
CHANNELS = 8
TIME = 371372
ELEMS_PER_CHANNEL = BATCH * TIME
PARTIAL_SIZE = 8192
NUM_PARTIALS = triton.cdiv(ELEMS_PER_CHANNEL, PARTIAL_SIZE)
FINAL_BLOCK = 256


@triton.jit
def _partial_sum_kernel(
    x_ptr,
    partial_ptr,
    time: tl.constexpr,
    channels: tl.constexpr,
    elems_per_channel: tl.constexpr,
    partial_size: tl.constexpr,
    num_partials: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    channel = tl.program_id(0)
    partial = tl.program_id(1)
    r = tl.arange(0, BLOCK_R)
    logical = partial * partial_size + r
    active = logical < elems_per_channel
    batch = logical // time
    t = logical - batch * time
    offsets = batch * channels * time + channel * time + t

    values = tl.load(x_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    total = tl.sum(tl.where(active, values, 0.0), axis=0)
    tl.store(partial_ptr + channel * num_partials + partial, total)


@triton.jit
def _final_sum_kernel(
    partial_ptr,
    out_ptr,
    num_partials: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    channel = tl.program_id(0)
    p = tl.arange(0, BLOCK_P)
    active = p < num_partials
    values = tl.load(
        partial_ptr + channel * num_partials + p,
        mask=active,
        other=0.0,
    ).to(tl.float32)
    total = tl.sum(tl.where(active, values, 0.0), axis=0)
    tl.store(out_ptr + channel, total)


# a809b30f: torchbench demucs train, bf16 [4,4,2,371372] viewed as [4,8,371372].
@oracle_impl(hardware="B200", point="a809b30f", BLOCK_R=PARTIAL_SIZE, final_warps=4)
def oracle_forward(inputs, *, BLOCK_R: int, final_warps: int):
    x, _shape = inputs
    del _shape

    view = torch.as_strided(
        x,
        (BATCH, CHANNELS, TIME),
        (CHANNELS * TIME, TIME, 1),
    )
    partial = torch.empty_strided(
        (CHANNELS, NUM_PARTIALS),
        (NUM_PARTIALS, 1),
        device=x.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided((CHANNELS,), (1,), device=x.device, dtype=torch.float32)

    _partial_sum_kernel[(CHANNELS, NUM_PARTIALS)](
        x,
        partial,
        time=TIME,
        channels=CHANNELS,
        elems_per_channel=ELEMS_PER_CHANNEL,
        partial_size=PARTIAL_SIZE,
        num_partials=NUM_PARTIALS,
        BLOCK_R=BLOCK_R,
        num_warps=8,
        num_stages=3,
    )
    _final_sum_kernel[(CHANNELS,)](
        partial,
        out,
        num_partials=NUM_PARTIALS,
        BLOCK_P=FINAL_BLOCK,
        num_warps=final_warps,
        num_stages=3,
    )
    return view, out
