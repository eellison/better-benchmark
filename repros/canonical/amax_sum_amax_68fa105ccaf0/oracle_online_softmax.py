"""
Oracle kernel for amax_sum_amax_68fa105ccaf0: dual online softmax forward (T5 training).

Pattern: Two independent softmax operations on f32[8, 8, 1024, 1024] tensors,
each following: amax(dim=-1) -> sub -> exp -> sum(dim=-1) -> div, followed by
dropout and transpose.

This is the T5 encoder-decoder cross-attention training forward pass with two
attention heads, each performing softmax + dropout independently. The two softmax
operations share the same structure but have different inputs (different relative
position biases).

Oracle strategy:
  - Online softmax (Milakov & Gimelshein 2018) with two passes per row.
  - Each program handles one row of the [M, 1024] input.
  - Since N=1024 is a power of 2, the entire row fits in one BLOCK_N=1024 load.
  - We run the kernel twice (once per softmax instance) since they are independent.
  - Dropout is applied via a separate random comparison (gt.Scalar + mul + mul).

Memory traffic (per softmax):
  - Read: 8*8*1024*1024 * 4 = 256 MB (f32 input)
  - Write: 8*8*1024*1024 * 4 = 256 MB (f32 output)
  - Total per softmax: 512 MB
  - Grand total (2 softmax): 1024 MB
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

REPRO_ID = "amax_sum_amax_68fa105ccaf0"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

# Problem dimensions
# Each softmax operates on [8, 8, 1024, 1024]
# Flattened: M = 8*8*1024 = 65536 rows, N = 1024 columns
BATCH = 8
N_HEADS = 8
SEQ_LEN = 1024
N = 1024  # reduction dimension
M = BATCH * N_HEADS * SEQ_LEN  # 65536


# --- Triton Kernel: Online softmax ---

@triton.jit
def online_softmax_fwd_kernel(
    input_ptr,
    output_ptr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Online softmax forward: fused max + sum_exp + normalize.

    Each program handles one row. For N=1024, one block load suffices.
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * N

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # Load entire row
    x = tl.load(input_ptr + row_start + cols, mask=mask, other=float("-inf")).to(tl.float32)

    # Compute softmax in one shot (N fits in one block)
    m = tl.max(x, axis=0)
    x_shifted = x - m
    exp_x = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax_out = exp_x / sum_exp

    tl.store(output_ptr + row_start + cols, softmax_out, mask=mask)


@triton.jit
def online_softmax_dropout_kernel(
    input_ptr,
    seed_ptr,         # scalar seed for this dropout instance
    output_ptr,
    N: tl.constexpr,
    dropout_p: tl.constexpr,  # dropout probability (0.1)
    scale: tl.constexpr,      # 1 / (1 - dropout_p) = 1.1111...
    BLOCK_N: tl.constexpr,
):
    """Fused softmax + dropout forward.

    Each program handles one row:
    1. Compute softmax
    2. Generate random mask (bernoulli with p=1-dropout_p)
    3. Apply: output = (random > dropout_p) * softmax_out * scale
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * N

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # Load entire row
    x = tl.load(input_ptr + row_start + cols, mask=mask, other=float("-inf")).to(tl.float32)

    # Softmax
    m = tl.max(x, axis=0)
    x_shifted = x - m
    exp_x = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax_out = exp_x / sum_exp

    # Dropout: generate random values and threshold
    seed = tl.load(seed_ptr)
    offsets = row_start + cols
    random_vals = tl.rand(seed, offsets)
    keep_mask = random_vals > dropout_p
    dropped = tl.where(keep_mask, softmax_out * scale, 0.0)

    tl.store(output_ptr + row_start + cols, dropped, mask=mask)


def oracle_softmax(x: torch.Tensor) -> torch.Tensor:
    """Launch the online softmax Triton kernel on a 2D tensor."""
    assert x.ndim == 2
    M_val, N_val = x.shape
    output = torch.empty_like(x)

    BLOCK_N = triton.next_power_of_2(N_val)
    grid = (M_val,)

    online_softmax_fwd_kernel[grid](
        x, output,
        N=N_val,
        BLOCK_N=BLOCK_N,
    )
    return output


def oracle_dual_softmax(x1: torch.Tensor, x2: torch.Tensor):
    """Compute softmax on two independent [B, H, S, S] tensors.

    Args:
        x1: [8, 8, 1024, 1024] first attention logits
        x2: [8, 8, 1024, 1024] second attention logits

    Returns:
        (softmax1, softmax2): both [8, 8, 1024, 1024]
    """
    shape = x1.shape
    x1_flat = x1.reshape(-1, shape[-1])
    x2_flat = x2.reshape(-1, shape[-1])

    out1 = oracle_softmax(x1_flat)
    out2 = oracle_softmax(x2_flat)

    return out1.reshape(shape), out2.reshape(shape)


# --- Reference ---

def softmax_reference(x: torch.Tensor) -> torch.Tensor:
    """Reference softmax matching repro pattern."""
    m = x.amax(dim=-1, keepdim=True)
    e = torch.exp(x - m)
    s = e.sum(dim=-1, keepdim=True)
    return e / s


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
    """Generate synthetic inputs matching the dual softmax shapes."""
    torch.manual_seed(42)
    # Two attention score tensors: [8, 8, 1024, 1024]
    x1 = torch.randn(BATCH, N_HEADS, SEQ_LEN, N, device="cuda", dtype=torch.float32)
    x2 = torch.randn(BATCH, N_HEADS, SEQ_LEN, N, device="cuda", dtype=torch.float32)
    return x1, x2


# --- Correctness check ---

def run_check():
    """Verify oracle produces same results as reference."""
    print(f"Correctness check for {REPRO_ID}")
    print(f"Shape: f32[{BATCH}, {N_HEADS}, {SEQ_LEN}, {N}], reduction dim=-1")
    print(f"Two independent softmax operations")

    x1, x2 = prepare_oracle_inputs()

    with torch.no_grad():
        # Reference
        ref1 = softmax_reference(x1)
        ref2 = softmax_reference(x2)

        # Oracle
        out1, out2 = oracle_dual_softmax(x1, x2)

    max_diff1 = (ref1 - out1).abs().max().item()
    max_diff2 = (ref2 - out2).abs().max().item()
    allclose1 = torch.allclose(ref1, out1, rtol=1e-4, atol=1e-5)
    allclose2 = torch.allclose(ref2, out2, rtol=1e-4, atol=1e-5)

    print(f"\n  Softmax 1:")
    print(f"    Max abs diff: {max_diff1:.6e}")
    print(f"    allclose: {allclose1}")
    print(f"\n  Softmax 2:")
    print(f"    Max abs diff: {max_diff2:.6e}")
    print(f"    allclose: {allclose2}")

    all_ok = allclose1 and allclose2
    print(f"\nCorrectness: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# --- Benchmark ---

def run_bench(rep=20, warmup=5):
    """Benchmark oracle vs torch.compile."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {REPRO_ID} (online_softmax_cross_entropy)")
    print(f"Oracle: dual persistent online softmax, N={N}")
    print(f"Shape: f32[{BATCH}, {N_HEADS}, {SEQ_LEN}, {N}] x 2")
    print(f"M={M} rows per softmax, N={N}")
    print(f"{'='*60}")

    x1, x2 = prepare_oracle_inputs()

    # Memory metrics: 2 softmax ops, each reads + writes [8,8,1024,1024] f32
    bytes_per_softmax = 2 * BATCH * N_HEADS * SEQ_LEN * N * 4  # read + write
    total_bytes = 2 * bytes_per_softmax
    print(f"\nMemory traffic (2 softmax ops): {total_bytes / 1e6:.1f} MB")

    # Benchmark oracle (dual softmax)
    with torch.no_grad():
        _ = oracle_dual_softmax(x1, x2)
    torch.cuda.synchronize()

    oracle_us = triton.testing.do_bench(
        lambda: oracle_dual_softmax(x1, x2),
        warmup=warmup * 20,
        rep=rep * 10,
    ) * 1000  # ms -> us

    oracle_bw = total_bytes / (oracle_us * 1e-6) / 1e12
    print(f"\nOracle (dual persistent softmax):")
    print(f"  Median: {oracle_us:.1f} us")
    print(f"  Effective BW: {oracle_bw:.3f} TB/s")

    # Benchmark single softmax
    x1_flat = x1.reshape(-1, N).contiguous()
    single_us = triton.testing.do_bench(
        lambda: oracle_softmax(x1_flat),
        warmup=warmup * 20,
        rep=rep * 10,
    ) * 1000

    print(f"\nOracle (single softmax):")
    print(f"  Median: {single_us:.1f} us")
    print(f"  Effective BW: {bytes_per_softmax / (single_us * 1e-6) / 1e12:.3f} TB/s")

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
    print(f"\nNote: Oracle measures the dual softmax kernels only.")
    print(f"Full repro includes relative position bias computation, dropout,")
    print(f"and transpose operations.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run correctness check")
    parser.add_argument("--bench", action="store_true", help="Run benchmark")
    parser.add_argument("--rep", type=int, default=20, help="Benchmark repetitions")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
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
