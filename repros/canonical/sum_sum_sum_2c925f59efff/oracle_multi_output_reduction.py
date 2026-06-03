"""
Oracle kernel for sum_sum_sum_2c925f59efff (PyTorch U-Net training backward).

Repro pattern (torchbench_pytorch_unet_train):
    Batch-norm backward with ReLU backward mask, producing two scalar reduction
    outputs after bilinear upsample backward (index_put + scatter).

    The computation structure:
      1. Bilinear interpolation backward via index_put (scatter-add pattern)
      2. Batch norm backward (standard):
         - sum1[c] = sum_{n,h,w} where_self[n,c,h,w]
         - sum2[c] = sum_{n,h,w} where_self[n,c,h,w] * sub_tensor_1[n,c,h,w]
         - mul_tensor_9 (pointwise from sum1,sum2) => sum3
      3. Outputs:
         - mul_tensor_13[c] = sum2[c] * squeeze_dims_1[c]  (shape [512])
         - sum_dim_int_list_2[c] = sum_{n,h,w} mul_tensor_12[n,c,h,w]  (shape [512])

    Key insight: mul_tensor_12 is a pointwise function of where_self and
    sub_tensor_1 parameterized by per-channel scalars derived from sum1 and sum2:
      mul_tensor_12 = (where_self - sub1*s2_scaled - s1_scaled) * weight

    So sum3 can be algebraically computed from sum1, sum2, and sum_sub:
      sum3[c] = weight[c] * (sum1[c] - s2_scaled[c]*sum_sub[c] - s1_scaled[c]*NHW)

    Oracle strategy:
      Phase 1: Compute upstream (index_put scatter, batchnorm forward, relu mask)
      Phase 2: Single-pass 3-accumulator Triton reduction kernel over where_self
               and sub_tensor_1 computing sum1, sum2, sum_sub simultaneously
      Phase 3: Per-channel scalar epilogue deriving both outputs algebraically
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_sum_2c925f59efff"
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
# Triton kernel: fused 3-accumulator partial reduction
# ---------------------------------------------------------------------------

@triton.jit
def _partial_reduction_3acc_kernel(
    where_self_ptr,       # [N, C, H, W] contiguous
    sub_tensor_1_ptr,     # [N, C, H, W] contiguous
    partial_sum1_ptr,     # [C, N_TILES]
    partial_sum2_ptr,     # [C, N_TILES]
    partial_sum_sub_ptr,  # [C, N_TILES]
    N: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    N_TILES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Partial 3-accumulator reduction over the (N, H, W) dimensions per channel.
    sum1[c] = sum where_self[:, c, :, :]
    sum2[c] = sum where_self[:, c, :, :] * sub_tensor_1[:, c, :, :]
    sum_sub[c] = sum sub_tensor_1[:, c, :, :]
    """
    c = tl.program_id(0)
    tile_id = tl.program_id(1)

    total = N * HW
    tile_size = (total + N_TILES - 1) // N_TILES
    tile_start = tile_id * tile_size
    tile_end = tl.minimum(tile_start + tile_size, total)

    acc1 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    acc_sub = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    base = c * HW

    for block_start in range(tile_start, tile_end, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < tile_end

        n = offsets // HW
        hw = offsets % HW
        idx = n * (C * HW) + base + hw

        w_val = tl.load(where_self_ptr + idx, mask=mask, other=0.0)
        s_val = tl.load(sub_tensor_1_ptr + idx, mask=mask, other=0.0)

        acc1 += w_val
        acc2 += w_val * s_val
        acc_sub += s_val

    sum1 = tl.sum(acc1, axis=0)
    sum2 = tl.sum(acc2, axis=0)
    sum_sub = tl.sum(acc_sub, axis=0)

    out_idx = c * N_TILES + tile_id
    tl.store(partial_sum1_ptr + out_idx, sum1)
    tl.store(partial_sum2_ptr + out_idx, sum2)
    tl.store(partial_sum_sub_ptr + out_idx, sum_sub)


@triton.jit
def _finalize_3acc_kernel(
    partial_sum1_ptr,
    partial_sum2_ptr,
    partial_sum_sub_ptr,
    out_sum1_ptr,
    out_sum2_ptr,
    out_sum_sub_ptr,
    N_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    """Sum partial results across tiles for each channel."""
    c = tl.program_id(0)
    base = c * N_TILES

    acc1 = tl.zeros([BLOCK_TILES], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_TILES], dtype=tl.float32)
    acc_sub = tl.zeros([BLOCK_TILES], dtype=tl.float32)

    for i in range(0, N_TILES, BLOCK_TILES):
        offsets = i + tl.arange(0, BLOCK_TILES)
        mask = offsets < N_TILES
        acc1 += tl.load(partial_sum1_ptr + base + offsets, mask=mask, other=0.0)
        acc2 += tl.load(partial_sum2_ptr + base + offsets, mask=mask, other=0.0)
        acc_sub += tl.load(partial_sum_sub_ptr + base + offsets, mask=mask, other=0.0)

    tl.store(out_sum1_ptr + c, tl.sum(acc1, axis=0))
    tl.store(out_sum2_ptr + c, tl.sum(acc2, axis=0))
    tl.store(out_sum_sub_ptr + c, tl.sum(acc_sub, axis=0))


# ---------------------------------------------------------------------------
# Oracle implementation
# ---------------------------------------------------------------------------

def compute_upstream(getitem_24, arg89_1, arg88_1, arg85_1, arg87_1, arg86_1,
                     arg84_1, arg81_1, arg82_1, arg83_1, arg24_1, arg25_1, full):
    """Compute the upstream portion (bilinear upsample backward + batchnorm fwd + relu mask).

    Returns where_self and sub_tensor_1 which are inputs to the fused reduction.
    Also returns per-channel constants needed for the scalar epilogue.
    """
    device = getitem_24.device

    # Bilinear upsample backward via index_put (scatter-add)
    slice_tensor = getitem_24[:, 512:1024, :, :]  # [8, 512, 80, 119]
    constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(
        slice_tensor, [0, -1, 0, 0])  # [8, 512, 80, 118]

    mul_tensor = constant_pad_nd_default * arg89_1
    neg_default = -mul_tensor
    add_tensor = constant_pad_nd_default + neg_default

    mul_tensor_1 = mul_tensor * arg88_1
    neg_default_1 = -mul_tensor_1
    add_tensor_1 = mul_tensor + neg_default_1

    mul_tensor_2 = add_tensor * arg88_1
    neg_default_2 = -mul_tensor_2
    add_tensor_2 = add_tensor + neg_default_2

    full_default = torch.zeros(8, 512, 40, 59, dtype=torch.float32, device=device)
    index_put_default = torch.ops.aten.index_put.default(
        full_default, [None, None, arg85_1, arg87_1], mul_tensor_1, True)
    index_put_default_1 = torch.ops.aten.index_put.default(
        full_default, [None, None, arg85_1, arg86_1], add_tensor_1, True)
    add_tensor_3 = index_put_default + index_put_default_1

    index_put_default_2 = torch.ops.aten.index_put.default(
        full_default, [None, None, arg84_1, arg87_1], mul_tensor_2, True)
    add_tensor_4 = add_tensor_3 + index_put_default_2

    index_put_default_3 = torch.ops.aten.index_put.default(
        full_default, [None, None, arg84_1, arg86_1], add_tensor_2, True)
    add_tensor_5 = add_tensor_4 + index_put_default_3

    # Batchnorm forward (to produce relu mask)
    sub_tensor = arg81_1 - arg82_1
    mul_tensor_3 = sub_tensor * arg83_1
    unsqueeze_default_1 = arg24_1.unsqueeze(-1).unsqueeze(-1)
    mul_tensor_4 = mul_tensor_3 * unsqueeze_default_1
    unsqueeze_default_3 = arg25_1.unsqueeze(-1).unsqueeze(-1)
    add_tensor_6 = mul_tensor_4 + unsqueeze_default_3

    # ReLU backward mask
    relu_default = torch.relu(add_tensor_6)
    le_scalar = relu_default <= 0
    where_self = torch.where(le_scalar, full, add_tensor_5)

    # sub_tensor_1 = arg81_1 - mean (per channel broadcast)
    squeeze_dims = arg82_1.squeeze(0).squeeze(-1).squeeze(-1)  # [512]
    unsqueeze_default_6 = squeeze_dims.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    sub_tensor_1 = arg81_1 - unsqueeze_default_6

    # Per-channel constants
    squeeze_dims_1 = arg83_1.squeeze(0).squeeze(-1).squeeze(-1)  # rsqrt [512]

    return where_self, sub_tensor_1, squeeze_dims_1, arg24_1


def fused_reduction_oracle(where_self, sub_tensor_1, squeeze_dims_1, weight):
    """Single-pass 3-accumulator reduction + scalar epilogue.

    Computes all reductions with one pass over the data, then derives
    both final outputs from per-channel scalar arithmetic.

    Returns: (mul_tensor_13, sum_dim_int_list_2) both shape [512]
    """
    device = where_self.device
    N, C, H, W = where_self.shape  # [8, 512, 40, 59]
    HW = H * W
    NHW = N * H * W
    scale = 5.296610169491525e-05  # 1 / NHW = 1 / (8*40*59) = 1/18880

    # Ensure contiguous
    where_self_c = where_self.contiguous()
    sub_tensor_1_c = sub_tensor_1.contiguous()

    # Phase 1: fused 3-output partial reduction
    N_TILES = 64
    BLOCK_SIZE = 1024

    partial_sum1 = torch.empty(C, N_TILES, dtype=torch.float32, device=device)
    partial_sum2 = torch.empty(C, N_TILES, dtype=torch.float32, device=device)
    partial_sum_sub = torch.empty(C, N_TILES, dtype=torch.float32, device=device)

    _partial_reduction_3acc_kernel[(C, N_TILES)](
        where_self_c, sub_tensor_1_c,
        partial_sum1, partial_sum2, partial_sum_sub,
        N=N, C=C, HW=HW, N_TILES=N_TILES, BLOCK_SIZE=BLOCK_SIZE,
    )

    # Finalize: sum partials
    out_sum1 = torch.empty(C, dtype=torch.float32, device=device)
    out_sum2 = torch.empty(C, dtype=torch.float32, device=device)
    out_sum_sub = torch.empty(C, dtype=torch.float32, device=device)

    BLOCK_TILES = 64
    _finalize_3acc_kernel[(C,)](
        partial_sum1, partial_sum2, partial_sum_sub,
        out_sum1, out_sum2, out_sum_sub,
        N_TILES=N_TILES, BLOCK_TILES=BLOCK_TILES,
    )

    # Phase 2: scalar per-channel epilogue
    # Output 1: mul_tensor_13 = sum2 * squeeze_dims_1
    mul_tensor_13 = out_sum2 * squeeze_dims_1

    # Output 2: sum_dim_int_list_2 = sum_{n,h,w} mul_tensor_12[n,c,h,w]
    # Expanding mul_tensor_12:
    #   mul_tensor_12 = (where_self - sub_1 * s2_scaled - s1_scaled) * weight_factor
    # where:
    #   s2_scaled[c] = sum2[c] * scale * rsqrt[c]^2
    #   s1_scaled[c] = sum1[c] * scale
    #   weight_factor[c] = rsqrt[c] * weight[c]
    #
    # sum3[c] = weight_factor[c] * (sum1[c] - s2_scaled[c]*sum_sub[c] - s1_scaled[c]*NHW)
    s2_scaled = out_sum2 * scale * squeeze_dims_1 * squeeze_dims_1
    s1_scaled = out_sum1 * scale
    weight_factor = squeeze_dims_1 * weight

    sum_dim_int_list_2 = weight_factor * (
        out_sum1 - s2_scaled * out_sum_sub - s1_scaled * NHW
    )

    return mul_tensor_13, sum_dim_int_list_2


def triton_oracle(getitem_24, arg89_1, arg88_1, arg85_1, arg87_1, arg86_1,
                  arg84_1, arg81_1, arg82_1, arg83_1, arg24_1, arg25_1, full):
    """Full oracle: upstream (PyTorch) + fused reduction (Triton) + scalar epilogue."""
    where_self, sub_tensor_1, squeeze_dims_1, weight = compute_upstream(
        getitem_24, arg89_1, arg88_1, arg85_1, arg87_1, arg86_1,
        arg84_1, arg81_1, arg82_1, arg83_1, arg24_1, arg25_1, full
    )
    return fused_reduction_oracle(where_self, sub_tensor_1, squeeze_dims_1, weight)


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
        oracle_out = triton_oracle(*inputs)
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
        # Precompute upstream
        where_self, sub_tensor_1, squeeze_dims_1, weight = compute_upstream(*inputs)
        torch.cuda.synchronize()

        # Benchmark just the fused reduction + epilogue
        def oracle_fn():
            return fused_reduction_oracle(where_self, sub_tensor_1, squeeze_dims_1, weight)

        ms = triton.testing.do_bench(oracle_fn, warmup=warmup, rep=rep)
        us = ms * 1000.0
        print(f"  oracle (fused 3-acc reduction+epilogue): {us:.1f} us")

        # Benchmark full oracle including upstream
        def full_oracle_fn():
            return triton_oracle(*inputs)

        ms_full = triton.testing.do_bench(full_oracle_fn, warmup=warmup, rep=rep)
        us_full = ms_full * 1000.0
        print(f"  oracle (full including upstream):        {us_full:.1f} us")

        # Benchmark compiled repro
        module = _load_repro_module()
        compiled = torch.compile(module.Repro())
        compiled(*inputs)
        torch.cuda.synchronize()

        ms_compiled = triton.testing.do_bench(lambda: compiled(*inputs), warmup=warmup, rep=rep)
        us_compiled = ms_compiled * 1000.0
        print(f"  compiled (torch.compile):                {us_compiled:.1f} us")

        print(f"\n  Summary:")
        print(f"    Oracle fused reduction+epilogue: {us:.1f} us")
        print(f"    Oracle full (inc. upstream):     {us_full:.1f} us")
        print(f"    Compiled full graph:             {us_compiled:.1f} us")

    return us


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
