"""
Oracle kernel for sum_sum_sum_431633879271 (ConvNextV2 GELU backward + layerscale).

Repro pattern (timm_convnextv2_nano):
    GELU backward combined with layerscale backward producing three [320]-shaped
    reduction outputs. The computation involves:
      - GELU' = 0.5*(1+erf(x/sqrt(2))) + x*gauss(x) applied to input arg89_1
      - Layerscale scaling and division
      - Multiple reductions over spatial dims [0,2,3] and intermediate reductions

    Key structure:
      sum_dim_int_list   [1,320,1,1] = sum(getitem_126, [0,2,3])
      sum_dim_int_list_1 [1,320,1,1] = sum(getitem_126 * mul_scalar * mul_tensor_2 * div_tensor, [0,2,3])
      sum_dim_int_list_2 [128,320,1,1] = sum(getitem_126*arg6_view*mul_tensor_2, [2,3])
      sum_dim_int_list_3 [128,1,1,1] = sum(-sum_dim_int_list_2 * div/arg91, [1])
      Then: sum_dim_int_list_4 [320] = sum(complex_gelu_backward_expr, [0,2,3])

    Oracle strategy:
      Phase 1: Compute two channel reductions (sum1, sum2) via a single Triton
               2-accumulator pass over getitem_126 with pointwise multipliers.
      Phase 2: Compute the per-sample-per-channel reduction (sum_dim_int_list_2)
               and derive sum_dim_int_list_3 via a separate reduction.
      Phase 3: Compute the final GELU backward reduction (sum_dim_int_list_4)
               that depends on results from phases 1-2.

    Since the final reduction (sum_dim_int_list_4) depends on intermediate
    reduction results, we cannot trivially fuse everything into one pass.
    The oracle uses two Triton kernels:
      Kernel 1: fused dual accumulator for sum1 + sum2, plus sum_dim_int_list_2
      Kernel 2: final reduction computing GELU' * combined_grad, folding the
                intermediate results as per-channel scalars.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_sum_431633879271"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Triton kernel 1: Fused channel-reduction for sum1 and sum2
# ---------------------------------------------------------------------------

@triton.jit
def _dual_channel_reduce_kernel(
    getitem_126_ptr,     # [N, C, H, W] contiguous
    mul_factor_ptr,      # [N, C, H, W] = mul_scalar * mul_tensor_2 * div_tensor
    partial_sum1_ptr,    # [C, N_TILES]
    partial_sum2_ptr,    # [C, N_TILES]
    N: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    N_TILES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Partial 2-accumulator reduction: each program handles one (channel, tile).
    sum1[c] = sum_{n,h,w} getitem_126[n,c,h,w]
    sum2[c] = sum_{n,h,w} getitem_126[n,c,h,w] * mul_factor[n,c,h,w]
    """
    c = tl.program_id(0)
    tile_id = tl.program_id(1)

    total = N * HW
    tile_size = (total + N_TILES - 1) // N_TILES
    tile_start = tile_id * tile_size
    tile_end = tl.minimum(tile_start + tile_size, total)

    acc1 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    base = c * HW

    for block_start in range(tile_start, tile_end, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < tile_end

        n = offsets // HW
        hw = offsets % HW
        idx = n * (C * HW) + base + hw

        g_val = tl.load(getitem_126_ptr + idx, mask=mask, other=0.0)
        f_val = tl.load(mul_factor_ptr + idx, mask=mask, other=0.0)

        acc1 += g_val
        acc2 += g_val * f_val

    sum1 = tl.sum(acc1, axis=0)
    sum2 = tl.sum(acc2, axis=0)

    out_idx = c * N_TILES + tile_id
    tl.store(partial_sum1_ptr + out_idx, sum1)
    tl.store(partial_sum2_ptr + out_idx, sum2)


@triton.jit
def _finalize_2acc_kernel(
    partial_sum1_ptr,
    partial_sum2_ptr,
    out_sum1_ptr,
    out_sum2_ptr,
    N_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    """Sum partial results across tiles for each channel."""
    c = tl.program_id(0)
    base = c * N_TILES

    acc1 = tl.zeros([BLOCK_TILES], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_TILES], dtype=tl.float32)

    for i in range(0, N_TILES, BLOCK_TILES):
        offsets = i + tl.arange(0, BLOCK_TILES)
        mask = offsets < N_TILES
        acc1 += tl.load(partial_sum1_ptr + base + offsets, mask=mask, other=0.0)
        acc2 += tl.load(partial_sum2_ptr + base + offsets, mask=mask, other=0.0)

    tl.store(out_sum1_ptr + c, tl.sum(acc1, axis=0))
    tl.store(out_sum2_ptr + c, tl.sum(acc2, axis=0))


# ---------------------------------------------------------------------------
# Triton kernel 2: Final GELU backward reduction
# ---------------------------------------------------------------------------

@triton.jit
def _gelu_backward_reduce_kernel(
    arg89_1_ptr,         # [N, C, H, W] - original input (for GELU')
    combined_grad_ptr,   # [N, C, H, W] - combined gradient (add_tensor_3)
    partial_out_ptr,     # [C, N_TILES]
    N: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    N_TILES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduction computing sum_{n,h,w} combined_grad * gelu_backward(arg89_1).
    GELU'(x) = 0.5*(1+erf(x/sqrt(2))) + x * exp(-x^2/2) * 1/sqrt(2*pi)
    """
    c = tl.program_id(0)
    tile_id = tl.program_id(1)

    total = N * HW
    tile_size = (total + N_TILES - 1) // N_TILES
    tile_start = tile_id * tile_size
    tile_end = tl.minimum(tile_start + tile_size, total)

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    base = c * HW

    for block_start in range(tile_start, tile_end, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < tile_end

        n = offsets // HW
        hw = offsets % HW
        idx = n * (C * HW) + base + hw

        x = tl.load(arg89_1_ptr + idx, mask=mask, other=0.0)
        grad = tl.load(combined_grad_ptr + idx, mask=mask, other=0.0)

        # GELU'(x) = 0.5*(1 + erf(x/sqrt(2))) + x * exp(-x^2/2) / sqrt(2*pi)
        erf_val = tl.math.erf(x * 0.7071067811865476)
        gauss_val = tl.exp(x * x * (-0.5)) * 0.3989422804014327
        gelu_grad = 0.5 * (erf_val + 1.0) + x * gauss_val

        acc += grad * gelu_grad

    result = tl.sum(acc, axis=0)
    out_idx = c * N_TILES + tile_id
    tl.store(partial_out_ptr + out_idx, result)


@triton.jit
def _finalize_1acc_kernel(
    partial_ptr,
    out_ptr,
    N_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    """Sum partial results for single accumulator."""
    c = tl.program_id(0)
    base = c * N_TILES

    acc = tl.zeros([BLOCK_TILES], dtype=tl.float32)
    for i in range(0, N_TILES, BLOCK_TILES):
        offsets = i + tl.arange(0, BLOCK_TILES)
        mask = offsets < N_TILES
        acc += tl.load(partial_ptr + base + offsets, mask=mask, other=0.0)

    tl.store(out_ptr + c, tl.sum(acc, axis=0))


# ---------------------------------------------------------------------------
# Oracle implementation
# ---------------------------------------------------------------------------

def oracle_fused(arg89_1, arg90_1, arg91_1, getitem_126, arg6_1, full_1,
                 _shape_param_0, _shape_param_1, _shape_param_2):
    """
    Oracle implementation of the full repro computation using Triton kernels
    for the reduction-heavy portions.

    Returns: (view_default_1, view_default_2, sum_dim_int_list_4) all shape [320]
    """
    device = arg89_1.device
    N, C, H, W = arg89_1.shape  # [128, 320, 56, 56]
    HW = H * W
    NHW = N * H * W

    # --- Pre-compute pointwise intermediates ---
    # GELU forward: mul_tensor_2 = 0.5*x * (1 + erf(x/sqrt(2)))
    mul_tensor = arg89_1 * 0.5
    mul_tensor_1 = arg89_1 * 0.7071067811865476
    erf_default = torch.erf(mul_tensor_1)
    add_tensor = erf_default + 1
    mul_tensor_2 = mul_tensor * add_tensor  # GELU(x)

    # div_tensor = arg90_1 / arg91_1  -> [128, 320, 1, 1]
    div_tensor = arg90_1 / arg91_1

    # mul_factor = mul_tensor_2 * div_tensor (for sum2 computation)
    # mul_tensor_3 = mul_tensor_2 * div_tensor * 1  -> [128, 320, 56, 56]
    # mul_tensor_4 = getitem_126 * mul_tensor_3
    # So sum2[c] = sum(getitem_126 * mul_tensor_2 * div_tensor, [0,2,3])
    mul_factor = mul_tensor_2 * div_tensor  # [128, 320, 56, 56] broadcast

    # --- Phase 1: dual reduction for sum1 and sum2 ---
    N_TILES = 64
    BLOCK_SIZE = 1024

    partial_sum1 = torch.empty(C, N_TILES, dtype=torch.float32, device=device)
    partial_sum2 = torch.empty(C, N_TILES, dtype=torch.float32, device=device)

    getitem_126_c = getitem_126.contiguous()
    mul_factor_c = mul_factor.contiguous()

    _dual_channel_reduce_kernel[(C, N_TILES)](
        getitem_126_c, mul_factor_c,
        partial_sum1, partial_sum2,
        N=N, C=C, HW=HW, N_TILES=N_TILES, BLOCK_SIZE=BLOCK_SIZE,
    )

    out_sum1 = torch.empty(C, dtype=torch.float32, device=device)
    out_sum2 = torch.empty(C, dtype=torch.float32, device=device)

    BLOCK_TILES = 64
    _finalize_2acc_kernel[(C,)](
        partial_sum1, partial_sum2,
        out_sum1, out_sum2,
        N_TILES=N_TILES, BLOCK_TILES=BLOCK_TILES,
    )

    # view_default_1 = out_sum2 (sum of getitem_126 * mul_scalar * mul_tensor_2 * div_tensor)
    view_default_1 = out_sum2
    # view_default_2 = out_sum1 (sum of getitem_126)
    view_default_2 = out_sum1

    # --- Phase 2: Intermediate computation for add_tensor_3 ---
    # mul_tensor_5 = getitem_126 * arg6_view  where arg6_view = arg6_1[1,C,1,1]
    arg6_view = arg6_1.view(1, C, 1, 1)
    mul_tensor_5 = getitem_126 * arg6_view

    # sum_dim_int_list_2 = sum(mul_tensor_5 * mul_tensor_2, [2,3]) -> [128, 320, 1, 1]
    # Note: mul_tensor_6 = mul_tensor_5 * mul_tensor_2
    mul_tensor_6 = mul_tensor_5 * mul_tensor_2
    sum_dim_int_list_2 = mul_tensor_6.sum(dim=[2, 3], keepdim=True)

    # mul_tensor_7 = mul_tensor_5 * div_tensor
    mul_tensor_7 = mul_tensor_5 * div_tensor
    # add_tensor_1 = getitem_126 + mul_tensor_7
    add_tensor_1 = getitem_126 + mul_tensor_7

    # div_tensor_1 = div_tensor / arg91_1
    div_tensor_1 = div_tensor / arg91_1
    # mul_tensor_8 = -sum_dim_int_list_2 * div_tensor_1
    neg_default = -sum_dim_int_list_2
    mul_tensor_8 = neg_default * div_tensor_1

    # sum_dim_int_list_3 = sum(mul_tensor_8, [1], keepdim=True) -> [128, 1, 1, 1]
    sum_dim_int_list_3 = mul_tensor_8.sum(dim=[1], keepdim=True)

    # expand + div_scalar
    expand_default = sum_dim_int_list_3.expand(_shape_param_2)
    div_scalar = expand_default / 320

    # div_tensor_2 = sum_dim_int_list_2 / arg91_1
    div_tensor_2 = sum_dim_int_list_2 / arg91_1
    # add_tensor_2 = div_tensor_2 + div_scalar -> [128, 320, 1, 1]
    add_tensor_2 = div_tensor_2 + div_scalar

    # where_self and clone
    div_tensor_3 = mul_tensor_2 / arg90_1
    eq_scalar = arg90_1 == 0
    where_self = torch.where(eq_scalar, full_1, div_tensor_3)
    clone_default = where_self.contiguous()

    # mul_tensor_9 = add_tensor_2 * clone_default -> [128, 320, 56, 56]
    mul_tensor_9 = add_tensor_2 * clone_default

    # add_tensor_3 = add_tensor_1 + mul_tensor_9 -> combined grad [128, 320, 56, 56]
    add_tensor_3 = add_tensor_1 + mul_tensor_9

    # --- Phase 3: Final reduction: sum(add_tensor_3 * gelu'(arg89_1), [0,2,3]) ---
    # mul_tensor_10 = add_tensor * 0.5 = (erf + 1) * 0.5
    # mul_tensor_11 = arg89_1^2
    # mul_tensor_12 = -0.5 * arg89_1^2
    # exp_default = exp(mul_tensor_12)
    # mul_tensor_13 = exp_default * 0.3989422804014327 (gaussian)
    # mul_tensor_14 = arg89_1 * mul_tensor_13
    # add_tensor_4 = mul_tensor_10 + mul_tensor_14  (GELU')
    # mul_tensor_15 = add_tensor_3 * add_tensor_4
    # sum_dim_int_list_4 = sum(mul_tensor_15, [0,2,3])

    add_tensor_3_c = add_tensor_3.contiguous()
    arg89_1_c = arg89_1.contiguous()

    partial_out = torch.empty(C, N_TILES, dtype=torch.float32, device=device)

    _gelu_backward_reduce_kernel[(C, N_TILES)](
        arg89_1_c, add_tensor_3_c,
        partial_out,
        N=N, C=C, HW=HW, N_TILES=N_TILES, BLOCK_SIZE=BLOCK_SIZE,
    )

    sum_dim_int_list_4 = torch.empty(C, dtype=torch.float32, device=device)
    _finalize_1acc_kernel[(C,)](
        partial_out, sum_dim_int_list_4,
        N_TILES=N_TILES, BLOCK_TILES=BLOCK_TILES,
    )

    return view_default_1, view_default_2, sum_dim_int_list_4


# ---------------------------------------------------------------------------
# Correctness + Benchmark
# ---------------------------------------------------------------------------

def make_inputs(device: torch.device = None) -> tuple:
    sys.path.insert(0, str(REPO_ROOT))
    module = _load_repro_module()
    inputs = module.make_inputs()
    moved = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            t = value.cuda() if device is None else value.to(device=device)
            moved.append(t)
        else:
            moved.append(value)
    return tuple(moved)


def check_correctness(device: torch.device, rtol: float = 1e-4, atol: float = 1e-3):
    """Compare oracle outputs against eager reference."""
    inputs = make_inputs(device)
    module = _load_repro_module()

    with torch.no_grad():
        oracle_out = oracle_fused(*inputs)
        ref_out = module.Repro()(*inputs)

    all_close = True
    for i, (o, r) in enumerate(zip(oracle_out, ref_out)):
        max_diff = (o.float() - r.float()).abs().max().item()
        close = torch.allclose(o.float(), r.float(), rtol=rtol, atol=atol)
        rel_err = ((o.float() - r.float()).abs() / (r.float().abs() + 1e-8)).max().item()
        print(f"  output[{i}]: shape={list(o.shape)}, max_abs_diff={max_diff:.6g}, "
              f"max_rel_err={rel_err:.6g}, allclose={close}")
        if not close:
            all_close = False

    print(f"  OVERALL: {'PASS' if all_close else 'FAIL'}")
    return all_close


def benchmark_oracle(device: torch.device, warmup: int = 50, rep: int = 200):
    """Benchmark the oracle vs torch.compile."""
    inputs = make_inputs(device)

    with torch.no_grad():
        # Warm up oracle
        _ = oracle_fused(*inputs)
        torch.cuda.synchronize()

        # Benchmark oracle
        ms_oracle = triton.testing.do_bench(
            lambda: oracle_fused(*inputs),
            warmup=warmup, rep=rep,
        )
        us_oracle = ms_oracle * 1000.0
        print(f"  oracle (full fused):       {us_oracle:.1f} us")

        # Benchmark compiled repro
        module = _load_repro_module()
        compiled = torch.compile(module.Repro())
        compiled(*inputs)
        torch.cuda.synchronize()

        ms_compiled = triton.testing.do_bench(
            lambda: compiled(*inputs),
            warmup=warmup, rep=rep,
        )
        us_compiled = ms_compiled * 1000.0
        print(f"  compiled (torch.compile):  {us_compiled:.1f} us")

        print(f"\n  Summary:")
        print(f"    Oracle:   {us_oracle:.1f} us")
        print(f"    Compiled: {us_compiled:.1f} us")
        if us_oracle > 0:
            print(f"    Speedup:  {us_compiled / us_oracle:.2f}x")

    return us_oracle


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run correctness check")
    parser.add_argument("--bench", action="store_true", help="Run benchmark")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.check:
        print(f"Correctness check ({REPRO_ID}):")
        check_correctness(device, rtol=args.rtol, atol=args.atol)

    if args.bench:
        print(f"\nBenchmark ({REPRO_ID}):")
        benchmark_oracle(device, warmup=args.warmup, rep=args.rep)

    if not args.check and not args.bench:
        print("Use --check for correctness or --bench for benchmarking")
        print("Running correctness check by default...")
        check_correctness(device, rtol=args.rtol, atol=args.atol)


if __name__ == "__main__":
    main()
