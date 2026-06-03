"""
Oracle kernel for amax_sum_68fe981b18dd: online softmax forward (Longformer inference).

Pattern: f32[8, 1024, 12, 513] -> amax(dim=-1) -> sub -> exp -> sum(dim=-1) -> div
         -> where(mask, 0.0, softmax_result)

This is the Longformer sliding-window attention softmax forward during inference.
The reduction dimension is 513 (sliding window size), which is small enough for
a persistent kernel approach.

After the softmax, a post-softmax mask zeros out padding positions, and the result
goes through diagonal-to-dense reshaping for the BMM.

Oracle strategy:
  - Online softmax (Milakov & Gimelshein 2018): single pass computing running max
    and sum_exp, then a second pass to normalize.
  - Each program handles one row of the [M, 513] flattened input.
  - Since N=513 is small (fits in registers with BLOCK_N=512), we use a persistent
    single-pass approach: load entire row, compute max, subtract, exp, sum, divide.
  - After softmax, apply the padding mask inline.

Memory traffic (softmax + mask only):
  - Read: 8*1024*12*513 * 4 bytes (input) + 8*1024 * 1 byte (mask)
  - Write: 8*1024*12*513 * 4 bytes (output)
  - Total: ~402 MB
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

REPRO_ID = "amax_sum_68fe981b18dd"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

# Problem dimensions
# Softmax operates on [8, 1024, 12, 513], flattened to [98304, 513]
BATCH = 8
SEQ_LEN = 1024
N_HEADS = 12
WINDOW = 513
M = BATCH * SEQ_LEN * N_HEADS  # 98304


# --- Triton Kernel: Fused softmax + mask ---

@triton.jit
def online_softmax_masked_kernel(
    input_ptr,       # [M, N] attention scores
    mask_ptr,        # [BATCH, SEQ_LEN] padding mask (bool)
    output_ptr,      # [M, N] softmax output
    N: tl.constexpr,
    N_HEADS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused online softmax + padding mask.

    Each program handles one row (one head of one sequence position).
    After computing softmax, applies the row-level mask:
        if mask[batch, seq_pos] == True:
            output[row, :] = 0.0
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * N

    # Determine batch and seq position for masking
    # row_idx maps to [batch, seq_pos, head] in row-major order
    # For [8, 1024, 12, 513]: row = batch*1024*12 + seq*12 + head
    batch_idx = row_idx // (1024 * N_HEADS)
    seq_idx = (row_idx % (1024 * N_HEADS)) // N_HEADS

    # Load mask for this position
    mask_val = tl.load(mask_ptr + batch_idx * 1024 + seq_idx)

    # Load row
    cols = tl.arange(0, BLOCK_N)
    mask_cols = cols < N
    x = tl.load(input_ptr + row_start + cols, mask=mask_cols, other=float("-inf")).to(tl.float32)

    # Compute softmax
    m = tl.max(x, axis=0)
    x_shifted = x - m
    exp_x = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax_out = exp_x / sum_exp

    # Apply padding mask: if mask is True, zero out the row
    # mask_val is bool (True = pad position in this repro)
    result = tl.where(mask_val, 0.0, softmax_out)

    # Store
    tl.store(output_ptr + row_start + cols, result, mask=mask_cols)


@triton.jit
def online_softmax_kernel(
    input_ptr,       # [M, N] attention scores
    output_ptr,      # [M, N] softmax output
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Simple online softmax without mask (for benchmarking the core pattern).

    Each program handles one row.
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * N

    cols = tl.arange(0, BLOCK_N)
    mask_cols = cols < N
    x = tl.load(input_ptr + row_start + cols, mask=mask_cols, other=float("-inf")).to(tl.float32)

    # Softmax
    m = tl.max(x, axis=0)
    x_shifted = x - m
    exp_x = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax_out = exp_x / sum_exp

    tl.store(output_ptr + row_start + cols, softmax_out, mask=mask_cols)


def oracle_softmax(x: torch.Tensor) -> torch.Tensor:
    """Launch the softmax kernel on [M, N] input."""
    assert x.ndim == 2
    M_val, N_val = x.shape
    output = torch.empty_like(x)

    BLOCK_N = triton.next_power_of_2(N_val)
    grid = (M_val,)

    online_softmax_kernel[grid](
        x, output,
        N=N_val,
        BLOCK_N=BLOCK_N,
    )
    return output


def oracle_softmax_masked(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Launch the fused softmax + mask kernel.

    Args:
        x: [8, 1024, 12, 513] attention scores
        mask: [8, 1024] bool padding mask (True = zeroed)
    """
    B, S, H, W = x.shape
    x_flat = x.reshape(B * S * H, W)
    output_flat = torch.empty_like(x_flat)

    BLOCK_N = triton.next_power_of_2(W)
    grid = (B * S * H,)

    online_softmax_masked_kernel[grid](
        x_flat, mask,
        output_flat,
        N=W,
        N_HEADS=H,
        BLOCK_N=BLOCK_N,
    )
    return output_flat.reshape(B, S, H, W)


# --- Reference ---

def softmax_reference(x: torch.Tensor) -> torch.Tensor:
    """Reference softmax matching repro pattern."""
    m = x.amax(dim=-1, keepdim=True)
    e = torch.exp(x - m)
    s = e.sum(dim=-1, keepdim=True)
    return e / s


def softmax_masked_reference(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Reference softmax + mask."""
    softmax_out = softmax_reference(x)
    # mask shape [8, 1024], expand to [8, 1024, 12, 513]
    mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)  # [8, 1024, 1, 1]
    return torch.where(mask_expanded, torch.zeros_like(softmax_out), softmax_out)


# --- Load repro module ---

def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare_oracle_inputs():
    """Generate synthetic inputs matching the softmax reduction shape."""
    torch.manual_seed(42)
    # Attention scores before softmax: [8, 1024, 12, 513]
    x = torch.randn(BATCH, SEQ_LEN, N_HEADS, WINDOW, device="cuda", dtype=torch.float32)
    # Padding mask: [8, 1024] bool
    mask = torch.zeros(BATCH, SEQ_LEN, device="cuda", dtype=torch.bool)
    # Mark some positions as padding
    mask[:, 800:] = True

    return x, mask


# --- Correctness check ---

def run_check():
    """Verify oracle produces same results as reference."""
    print(f"Correctness check for {REPRO_ID}")
    print(f"Shape: f32[{BATCH}, {SEQ_LEN}, {N_HEADS}, {WINDOW}], reduction dim=-1")

    x, mask = prepare_oracle_inputs()

    with torch.no_grad():
        # Test basic softmax
        x_flat = x.reshape(-1, WINDOW)
        ref_flat = softmax_reference(x_flat)
        out_flat = oracle_softmax(x_flat)

        max_diff = (ref_flat - out_flat).abs().max().item()
        allclose_basic = torch.allclose(ref_flat, out_flat, rtol=1e-4, atol=1e-5)
        print(f"\n  Basic softmax:")
        print(f"    Max abs diff: {max_diff:.6e}")
        print(f"    allclose: {allclose_basic}")

        # Test masked softmax
        ref_masked = softmax_masked_reference(x, mask)
        out_masked = oracle_softmax_masked(x, mask)

        max_diff_m = (ref_masked - out_masked).abs().max().item()
        allclose_masked = torch.allclose(ref_masked, out_masked, rtol=1e-4, atol=1e-5)
        print(f"\n  Masked softmax:")
        print(f"    Max abs diff: {max_diff_m:.6e}")
        print(f"    allclose: {allclose_masked}")

    all_ok = allclose_basic and allclose_masked
    print(f"\nCorrectness: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# --- Benchmark ---

def run_bench(rep=100, warmup=25):
    """Benchmark oracle vs torch.compile."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {REPRO_ID} (online_softmax_cross_entropy)")
    print(f"Oracle: persistent online softmax, N={WINDOW}")
    print(f"Shape: f32[{BATCH}, {SEQ_LEN}, {N_HEADS}, {WINDOW}]")
    print(f"M={M} rows, N={WINDOW}")
    print(f"{'='*60}")

    x, mask = prepare_oracle_inputs()

    # Memory metrics for softmax only (read input + write output)
    total_bytes = 2 * BATCH * SEQ_LEN * N_HEADS * WINDOW * 4  # read + write, f32
    print(f"\nMemory traffic (softmax): {total_bytes / 1e6:.1f} MB")

    # Benchmark oracle (basic softmax)
    x_flat = x.reshape(-1, WINDOW).contiguous()
    with torch.no_grad():
        _ = oracle_softmax(x_flat)
    torch.cuda.synchronize()

    oracle_us = triton.testing.do_bench(
        lambda: oracle_softmax(x_flat),
        warmup=warmup * 20,
        rep=rep * 10,
    ) * 1000  # ms -> us

    oracle_bw = total_bytes / (oracle_us * 1e-6) / 1e12
    print(f"\nOracle (persistent softmax):")
    print(f"  Median: {oracle_us:.1f} us")
    print(f"  Effective BW: {oracle_bw:.3f} TB/s")

    # Benchmark masked version
    with torch.no_grad():
        _ = oracle_softmax_masked(x, mask)
    torch.cuda.synchronize()

    oracle_masked_us = triton.testing.do_bench(
        lambda: oracle_softmax_masked(x, mask),
        warmup=warmup * 20,
        rep=rep * 10,
    ) * 1000

    print(f"\nOracle (softmax + mask):")
    print(f"  Median: {oracle_masked_us:.1f} us")

    # Benchmark torch.compile on full repro
    repro_mod = _load_repro_module()
    model = repro_mod.Repro().cuda()
    inputs = repro_mod.make_inputs()
    inputs = tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs)

    with torch.no_grad():
        compiled = torch.compile(model)
        _ = compiled(*inputs)
        torch.cuda.synchronize()

    compile_us = triton.testing.do_bench(
        lambda: compiled(*inputs),
        warmup=warmup * 20,
        rep=rep * 10,
    ) * 1000

    print(f"\ntorch.compile (full repro):")
    print(f"  Median: {compile_us:.1f} us")

    print(f"\nSpeedup (compile / oracle): {compile_us / oracle_us:.2f}x")
    print(f"\nNote: Oracle measures the softmax kernel only.")
    print(f"Full repro includes sliding-window assembly + post-softmax reshape.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run correctness check")
    parser.add_argument("--bench", action="store_true", help="Run benchmark")
    parser.add_argument("--rep", type=int, default=100, help="Benchmark repetitions")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check:
        ok = run_check()
        if not ok:
            sys.exit(1)

    if args.bench:
        run_bench(rep=args.rep, warmup=args.warmup)


if __name__ == "__main__":
    with torch.no_grad():
        main()
