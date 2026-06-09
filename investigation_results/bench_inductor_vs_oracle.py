"""
Benchmark the Inductor-generated softmax kernel vs the oracle at matching BLOCK sizes.
This isolates the code quality difference from the block size difference.
"""
import torch
import triton
import triton.language as tl
import sys
sys.path.insert(0, '/tmp/scratch_space/better_benchmark')

from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math


# === Inductor-generated kernel (exact copy from output_code.py) ===
@triton.jit
def inductor_softmax_kernel(in_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 8192
    r0_numel = 262144
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :].to(tl.int64)
    rbase = r0_base
    x0 = xindex
    _tmp3_max = tl.full([XBLOCK], float('-inf'), tl.float32)
    _tmp3_sum = tl.full([XBLOCK], 0.0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 262144*x0), None, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])

        _tmp3_max_block_max = triton_helpers.max2(tmp2, 1)
        _tmp3_max_new = triton_helpers.maximum(_tmp3_max, _tmp3_max_block_max)
        _tmp3_sum_correction = tl.where(
            _tmp3_max_new == float("-inf"),
            1.0,
            tl_math.exp((_tmp3_max - _tmp3_max_new).to(tl.float32))
        )
        _tmp3_sum_block = tl.sum(
            tl_math.exp((tmp2 - _tmp3_max_new[:, None]).to(tl.float32)),
            1
        )
        _tmp3_sum = _tmp3_sum * _tmp3_sum_correction + _tmp3_sum_block
        _tmp3_max = _tmp3_max_new

    tmp3 = _tmp3_max[:, None]
    tmp4 = _tmp3_sum[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp5 = tl.load(in_ptr0 + (r0_1 + 262144*x0), None, eviction_policy='evict_first').to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp6 - tmp3
        tmp8 = libdevice.exp(tmp7)
        tmp9 = (tmp8 / tmp4)
        tmp10 = tmp9.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 262144*x0), tmp10, None)


# === Oracle kernel (simplified from oracle_online_softmax.py) ===
@triton.jit
def oracle_softmax_kernel(
    input_ptr,
    output_ptr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * N

    # Pass 1: Online computation of max and sum_exp
    m_i = float("-inf")  # running max (scalar)
    l_i = 0.0           # running sum of exp(x - m) (scalar)

    for block_start in tl.range(0, N, BLOCK_N):
        cols = block_start + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=float("-inf")).to(tl.float32)

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(x, axis=0))
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m_i = m_new

    # Pass 2: Normalize and write output
    for block_start in tl.range(0, N, BLOCK_N):
        cols = block_start + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=float("-inf")).to(tl.float32)
        out = tl.exp(x - m_i) / l_i
        tl.store(output_ptr + row_start + cols, out.to(tl.bfloat16), mask=mask)


def benchmark_cuda_graph(fn, warmup=5, rep=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(rep):
        start_event.record()
        graph.replay()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
    times.sort()
    return times[len(times)//2] * 1000.0  # us


def main():
    torch.manual_seed(42)
    x = torch.randn(8192, 262144, dtype=torch.bfloat16, device='cuda')
    out = torch.empty_like(x)

    print("=" * 70)
    print("Inductor vs Oracle Softmax Kernel Benchmark")
    print(f"Shape: bf16[8192, 262144]")
    print("=" * 70)

    for block_size in [2048, 4096, 8192, 16384]:
        print(f"\n--- BLOCK_SIZE = {block_size} (iterations per pass = {262144//block_size}) ---")

        # Oracle
        grid = (8192,)
        def run_oracle():
            oracle_softmax_kernel[grid](x, out, N=262144, BLOCK_N=block_size)
        oracle_us = benchmark_cuda_graph(run_oracle, warmup=5, rep=20)

        # Inductor
        grid_inductor = (8192,)  # XBLOCK=1
        def run_inductor():
            inductor_softmax_kernel[grid_inductor](x, out, 8192, 262144, XBLOCK=1, R0_BLOCK=block_size)
        inductor_us = benchmark_cuda_graph(run_inductor, warmup=5, rep=20)

        ratio = inductor_us / oracle_us
        print(f"  Oracle:   {oracle_us:.1f} us")
        print(f"  Inductor: {inductor_us:.1f} us")
        print(f"  Ratio:    {ratio:.3f}x ({(ratio-1)*100:.1f}% slower)")


if __name__ == "__main__":
    with torch.no_grad():
        main()
