"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Demucs bf16 masked-zero producer scope, returning the bf16 scalar zero, materializing `where(arg0 <= 0, 0, arg1)`, and accumulating the compiled-path f32 channel sum from the same source-space traversal, whereas Inductor already emits a compact fused masked reduction but still has to materialize the returned bf16 where tensor as part of the full repro scope. Inductor cannot expose a separate high-ratio scheduler-fusion, scatter-reduce, cooperative split-K, algebraic-elimination, or recompute-fusion opportunity here because the remaining work is the required two-input read, bf16 where store, f32 channel accumulation, scalar-zero store, and launch overhead; the fix is BANDWIDTH_BOUND: record the full-scope floor unless broader masked-reduction/materialization codegen moves both implementations."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 4
CHANNELS = 2048
TIME = 92
REDUCTION = BATCH * TIME
BLOCK_R = 512


@triton.jit
def _where_channel_sum_kernel(
    pred_ptr,
    src_ptr,
    full_ptr,
    where_ptr,
    sum_ptr,
    channels: tl.constexpr,
    time: tl.constexpr,
    reduction: tl.constexpr,
    BLOCK_R_: tl.constexpr,
):
    channel = tl.program_id(0)
    r = tl.arange(0, BLOCK_R_)
    active = r < reduction
    batch = r // time
    t = r - batch * time
    offsets = batch * channels * time + channel * time + t

    pred = tl.load(pred_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    src = tl.load(src_ptr + offsets, mask=active, other=0.0)
    zero = tl.full((BLOCK_R_,), 0.0, tl.float32).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    values = tl.where(pred <= 0.0, zero, src)

    zero_scalar = tl.full((), 0.0, tl.float32).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    tl.store(full_ptr, zero_scalar, mask=channel == 0)
    tl.store(where_ptr + offsets, values, mask=active)
    total = tl.sum(tl.where(active, values.to(tl.float32), 0.0), axis=0)
    tl.store(sum_ptr + channel, total)


# torchbench_demucs train, bf16 [4,2048,92] masked-zero where plus channel sum.
@oracle_impl(hardware="B200", point="d91f9612", BLOCK_R=BLOCK_R, num_warps=8)
def oracle_forward(inputs, *, BLOCK_R: int, num_warps: int):
    pred, src = inputs
    device = pred.device

    full = torch.empty((), device=device, dtype=torch.bfloat16)
    where = torch.empty_strided(
        (BATCH, CHANNELS, TIME),
        (CHANNELS * TIME, TIME, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    reduced = torch.empty_strided((CHANNELS,), (1,), device=device, dtype=torch.float32)

    _where_channel_sum_kernel[(CHANNELS,)](
        pred,
        src,
        full,
        where,
        reduced,
        channels=CHANNELS,
        time=TIME,
        reduction=REDUCTION,
        BLOCK_R_=BLOCK_R,
        num_warps=num_warps,
        num_stages=3,
    )
    return full, where, reduced
