import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def simple_max_kernel(in_ptr, out_ptr, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]

    acc = tl.full([XBLOCK, RBLOCK], float('-inf'), tl.float32)
    for roffset in tl.range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        val = tl.load(in_ptr + (rindex + 50272 * xindex), rmask & xmask, other=float('-inf')).to(tl.float32)
        acc = tl.maximum(acc, val)

    result = tl.max(acc, 1)[:, None]
    tl.store(out_ptr + xindex, result, xmask)


@triton.jit
def online_softmax_kernel(in_ptr, out_max, out_sum, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    """Mimics the online softmax kernel from inductor."""
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]

    acc_max = tl.full([XBLOCK, RBLOCK], float('-inf'), tl.float32)
    acc_sum = tl.zeros([XBLOCK, RBLOCK], tl.float32)

    for roffset in tl.range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        val = tl.load(in_ptr + (rindex + 50272 * xindex), rmask & xmask, other=float('-inf')).to(tl.float32)

        new_max = tl.maximum(acc_max, val)
        lhs_scale = tl.where(new_max == float('-inf'), 1.0, tl.math.exp(acc_max - new_max))
        rhs_scale = tl.where(new_max == float('-inf'), 1.0, tl.math.exp(val - new_max))
        new_sum = acc_sum * lhs_scale + rhs_scale

        acc_max = tl.where(rmask & xmask, new_max, acc_max)
        acc_sum = tl.where(rmask & xmask, new_sum, acc_sum)

    # Final reduce across RBLOCK dimension
    final_max = tl.max(acc_max, 1)[:, None]
    delta = tl.where(final_max == float('-inf'), 0.0, acc_max - final_max)
    final_sum = tl.sum(acc_sum * tl.math.exp(delta), 1)[:, None]

    tl.store(out_max + xindex, final_max, xmask)
    tl.store(out_sum + xindex, final_sum, xmask)


@triton.jit
def twopass_kernel(in_ptr, out_max, out_sum, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    """Two-pass: first max, then exp+sum. Uses fast math."""
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]

    # Pass 1: max
    acc_max = tl.full([XBLOCK, RBLOCK], float('-inf'), tl.float32)
    for roffset in tl.range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        val = tl.load(in_ptr + (rindex + 50272 * xindex), rmask & xmask, other=float('-inf')).to(tl.float32)
        acc_max = tl.maximum(acc_max, val)
    final_max = tl.max(acc_max, 1)[:, None]
    tl.store(out_max + xindex, final_max, xmask)

    # Pass 2: exp(x - max) sum
    acc_sum = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in tl.range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        val = tl.load(in_ptr + (rindex + 50272 * xindex), rmask & xmask, other=float('-inf')).to(tl.float32)
        exp_val = tl.math.exp(val - final_max)
        exp_val = tl.where(rmask & xmask, exp_val, 0.0)
        acc_sum = acc_sum + exp_val
    final_sum = tl.sum(acc_sum, 1)[:, None]
    tl.store(out_sum + xindex, final_sum, xmask)


@triton.jit
def twopass_libdevice_kernel(in_ptr, out_max, out_sum, xnumel, rnumel, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    """Two-pass: uses libdevice.exp (full precision, matching inductor)."""
    from triton.language.extra.cuda import libdevice
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]

    # Pass 1: max
    acc_max = tl.full([XBLOCK, RBLOCK], float('-inf'), tl.float32)
    for roffset in tl.range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        val = tl.load(in_ptr + (rindex + 50272 * xindex), rmask & xmask, other=float('-inf')).to(tl.float32)
        acc_max = tl.maximum(acc_max, val)
    final_max = tl.max(acc_max, 1)[:, None]
    tl.store(out_max + xindex, final_max, xmask)

    # Pass 2: exp(x - max) sum
    acc_sum = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in tl.range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        val = tl.load(in_ptr + (rindex + 50272 * xindex), rmask & xmask, other=float('-inf')).to(tl.float32)
        exp_val = libdevice.exp(val - final_max)
        exp_val = tl.where(rmask & xmask, exp_val, 0.0)
        acc_sum = acc_sum + exp_val
    final_sum = tl.sum(acc_sum, 1)[:, None]
    tl.store(out_sum + xindex, final_sum, xmask)


def bench():
    x = torch.randn(1024, 50272, dtype=torch.bfloat16, device='cuda')
    out_max = torch.empty(1024, dtype=torch.float32, device='cuda')
    out_sum = torch.empty(1024, dtype=torch.float32, device='cuda')

    data_bytes = 1024 * 50272 * 2

    print("=== Simple Max Reduction ===")
    for rblock in [512, 1024, 2048, 4096, 8192]:
        grid = (1024,)
        try:
            ms = do_bench(
                lambda rb=rblock: simple_max_kernel[grid](x, out_max, 1024, 50265, XBLOCK=1, RBLOCK=rb),
                warmup=25, rep=100
            )
            bw = data_bytes / ms / 1e6
            print(f'  RBLOCK={rblock:5d}: {ms*1000:.1f} us, BW: {bw:.0f} GB/s')
        except Exception as e:
            print(f'  RBLOCK={rblock:5d}: FAILED - {e}')

    print("\n=== Online Softmax ===")
    for rblock in [512, 1024, 2048, 4096]:
        grid = (1024,)
        try:
            ms = do_bench(
                lambda rb=rblock: online_softmax_kernel[grid](x, out_max, out_sum, 1024, 50265, XBLOCK=1, RBLOCK=rb),
                warmup=25, rep=100
            )
            bw = data_bytes / ms / 1e6
            print(f'  RBLOCK={rblock:5d}: {ms*1000:.1f} us, BW: {bw:.0f} GB/s (1 read)')
        except Exception as e:
            print(f'  RBLOCK={rblock:5d}: FAILED - {e}')

    print("\n=== Two-Pass fast math (max then exp+sum) ===")
    for rblock in [512, 1024, 2048, 4096]:
        grid = (1024,)
        try:
            ms = do_bench(
                lambda rb=rblock: twopass_kernel[grid](x, out_max, out_sum, 1024, 50265, XBLOCK=1, RBLOCK=rb),
                warmup=25, rep=100
            )
            bw = data_bytes / ms / 1e6
            print(f'  RBLOCK={rblock:5d}: {ms*1000:.1f} us, read BW: {bw:.0f} GB/s (2 reads)')
        except Exception as e:
            print(f'  RBLOCK={rblock:5d}: FAILED - {e}')

    print("\n=== Two-Pass libdevice.exp (full precision, matching inductor) ===")
    for rblock in [512, 1024, 2048, 4096]:
        grid = (1024,)
        try:
            ms = do_bench(
                lambda rb=rblock: twopass_libdevice_kernel[grid](x, out_max, out_sum, 1024, 50265, XBLOCK=1, RBLOCK=rb),
                warmup=25, rep=100
            )
            bw = data_bytes / ms / 1e6
            print(f'  RBLOCK={rblock:5d}: {ms*1000:.1f} us, read BW: {bw:.0f} GB/s (2 reads)')
        except Exception as e:
            print(f'  RBLOCK={rblock:5d}: FAILED - {e}')

    print("\n=== Reference ===")
    src = torch.empty(data_bytes // 4, dtype=torch.float32, device='cuda')
    dst = torch.empty_like(src)
    ms = do_bench(lambda: dst.copy_(src), warmup=25, rep=100)
    print(f'  memcopy ({data_bytes/1e6:.1f}MB r+w): {ms*1000:.1f} us, BW: {data_bytes*2/ms/1e6:.0f} GB/s')


if __name__ == "__main__":
    bench()
