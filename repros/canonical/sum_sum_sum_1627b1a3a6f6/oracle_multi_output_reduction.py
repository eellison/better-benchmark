"""
Oracle kernel for sum_sum_sum_1627b1a3a6f6 (PyTorch UNet training backward).

Repro pattern:
    Batch-norm backward + ReLU backward producing three channel-wise sum
    reductions over shape [8, 128, 320, 479]:
      - sum1[c] = sum_{n,h,w} where_self[n,c,h,w]
      - sum2[c] = sum_{n,h,w} (where_self * sub_tensor_1)[n,c,h,w]
      - sum3[c] = sum_{n,h,w} mul_tensor_9[n,c,h,w]

    Key insight: mul_tensor_9 is a pointwise function of where_self and
    sub_tensor_1 parameterized by per-channel scalars derived from sum1 and sum2.
    Expanding:
      mul_tensor_9[n,c,h,w] = weight_factor[c] * (
          where_self[n,c,h,w]
          - sub_tensor_1[n,c,h,w] * (sum2[c] * scale * rsqrt[c]^2)
          - sum1[c] * scale
      )
    So: sum3[c] = weight_factor[c] * (
          sum1[c] - sum2[c] * scale * rsqrt[c]^2 * sum_sub[c] - sum1[c] * scale * NHW
    )
    where sum_sub[c] = sum_{n,h,w} sub_tensor_1[n,c,h,w].

    Therefore all three output reductions can be derived from FOUR per-channel
    partial sums computed in a SINGLE pass over the data:
      - sum1[c] = sum where_self
      - sum2[c] = sum (where_self * sub_tensor_1)
      - sum_sub[c] = sum sub_tensor_1

    The final outputs are:
      - mul_tensor_10[c] = sum2[c] * squeeze_dims_1[c]   (shape [128])
      - sum3[c] (derived algebraically)                   (shape [128])

    The oracle fuses the entire post-scatter_add computation into:
      Phase 1: Compute where_self element-wise (batchnorm + relu backward)
      Phase 2: Single-pass 3-accumulator Triton reduction kernel
      Phase 3: Per-channel scalar epilogue (no kernel needed, trivial)

    This eliminates:
      - The materialization of the large [8, 128, 320, 479] intermediate mul_tensor_9
      - Multiple reduction kernel launches
      - Multiple passes over the same data
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)



REPRO_ID = "sum_sum_sum_1627b1a3a6f6"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"


def _partial_reduction_3acc_kernel(
    where_self_ptr,     # [N, C, H, W] contiguous -> flat [N*C*H*W]
    sub_tensor_1_ptr,   # [N, C, H, W] contiguous -> flat [N*C*H*W]
    partial_sum1_ptr,   # [C, N_TILES]
    partial_sum2_ptr,   # [C, N_TILES]
    partial_sum_sub_ptr,  # [C, N_TILES]
    N: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    N_TILES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Partial 3-accumulator reduction: each program handles one (channel, tile)."""
    c = tl.program_id(0)
    tile_id = tl.program_id(1)

    total = N * HW  # elements per channel to reduce
    tile_size = (total + N_TILES - 1) // N_TILES
    tile_start = tile_id * tile_size
    tile_end = tl.minimum(tile_start + tile_size, total)

    acc1 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    acc_sub = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Channel offset within each N-slice: element [n, c, h, w] is at
    # n * C * HW + c * HW + hw
    base = c * HW

    for block_start in range(tile_start, tile_end, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < tile_end

        # Convert flat (within-channel) offset to memory index
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

    # Store partial results
    out_idx = c * N_TILES + tile_id
    tl.store(partial_sum1_ptr + out_idx, sum1)
    tl.store(partial_sum2_ptr + out_idx, sum2)
    tl.store(partial_sum_sub_ptr + out_idx, sum_sub)


@triton.jit
def _finalize_3acc_kernel(
    partial_sum1_ptr,     # [C, N_TILES]
    partial_sum2_ptr,     # [C, N_TILES]
    partial_sum_sub_ptr,  # [C, N_TILES]
    out_sum1_ptr,         # [C]
    out_sum2_ptr,         # [C]
    out_sum_sub_ptr,      # [C]
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

def compute_upstream(getitem_12, getitem_42, arg61_1, arg57_1, arg58_1, arg59_1,
                     arg9_1, arg10_1, full, _shape_param_0, _shape_param_1, _shape_param_2):
    """Compute the upstream portion (scatter_add + batchnorm + relu backward).

    Returns where_self and sub_tensor_1 which are inputs to the fused reduction.
    Also returns per-channel constants needed for the scalar epilogue.
    """
    device = getitem_12.device

    # scatter_add for max_pool backward
    slice_tensor = getitem_12[:, :128, :, :]
    full_default = torch.zeros(1024, 153280, dtype=torch.float32, device=device)
    view_default = getitem_42.view(_shape_param_0)
    offsets_i64 = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
        arg61_1, [2, 2], [320, 479], [2, 2], [0, 0], [1, 1]
    )
    view_default_1 = offsets_i64.view(_shape_param_1)
    scatter_add_default = full_default.scatter_add(1, view_default_1, view_default)
    view_default_2 = scatter_add_default.view(_shape_param_2)

    # add_tensor = slice_tensor + view_default_2
    add_tensor = slice_tensor + view_default_2

    # Batchnorm forward intermediate
    sub_tensor = arg57_1 - arg58_1
    mul_tensor = sub_tensor * arg59_1
    weight_4d = arg9_1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    mul_tensor_1 = mul_tensor * weight_4d
    bias_4d = arg10_1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    add_tensor_1 = mul_tensor_1 + bias_4d

    # ReLU backward
    relu_default = torch.relu(add_tensor_1)
    le_scalar = relu_default <= 0
    where_self = torch.where(le_scalar, full, add_tensor)

    # sub_tensor_1 = arg57_1 - mean (per channel broadcast)
    squeeze_dims = arg58_1.squeeze(0).squeeze(-1).squeeze(-1)  # [128]
    unsqueeze_default_6 = squeeze_dims.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    sub_tensor_1 = arg57_1 - unsqueeze_default_6

    # Per-channel constants
    squeeze_dims_1 = arg59_1.squeeze(0).squeeze(-1).squeeze(-1)  # rsqrt [128]

    return where_self, sub_tensor_1, squeeze_dims_1, arg9_1


def fused_reduction_oracle(where_self, sub_tensor_1, squeeze_dims_1, arg9_1):
    """Single-pass 3-accumulator reduction + scalar epilogue.

    Computes all three reductions with one pass over the data, then derives
    both final outputs from per-channel scalar arithmetic.

    Returns: (mul_tensor_10, sum_dim_int_list_2) both shape [128]
    """
    device = where_self.device
    N, C, H, W = where_self.shape
    HW = H * W
    NHW = N * H * W

    # Ensure contiguous
    where_self_c = where_self.contiguous()
    sub_tensor_1_c = sub_tensor_1.contiguous()

    # Phase 1: fused 3-output partial reduction
    N_TILES = 128
    BLOCK_SIZE = 2048

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

    # BLOCK_TILES must be >= N_TILES (power of 2)
    BLOCK_TILES = 128
    _finalize_3acc_kernel[(C,)](
        partial_sum1, partial_sum2, partial_sum_sub,
        out_sum1, out_sum2, out_sum_sub,
        N_TILES=N_TILES, BLOCK_TILES=BLOCK_TILES,
    )

    # Phase 2: scalar per-channel epilogue (no kernel needed, trivial on [128])
    scale = 8.155010438413362e-07  # 1 / (N * H * W) = 1 / 1226240

    # Output 1: mul_tensor_10 = sum2 * squeeze_dims_1
    # From repro: mul_tensor_10 = sum_dim_int_list_1 * squeeze_dims_1
    mul_tensor_10 = out_sum2 * squeeze_dims_1

    # Output 2: sum_dim_int_list_2 = sum_{n,h,w} mul_tensor_9[n,c,h,w]
    # Expanding mul_tensor_9:
    #   mul_tensor_9 = (where_self - sub_tensor_1 * factor12 - factor9) * weight_factor
    # where:
    #   factor12[c] = out_sum2[c] * scale * squeeze_dims_1[c]^2
    #   factor9[c] = out_sum1[c] * scale
    #   weight_factor[c] = squeeze_dims_1[c] * arg9_1[c]
    #
    # sum3[c] = weight_factor[c] * (
    #     sum1[c] - factor12[c] * sum_sub[c] - factor9[c] * NHW
    # )
    factor12 = out_sum2 * scale * squeeze_dims_1 * squeeze_dims_1
    factor9 = out_sum1 * scale
    weight_factor = squeeze_dims_1 * arg9_1

    sum_dim_int_list_2 = weight_factor * (
        out_sum1 - factor12 * out_sum_sub - factor9 * NHW
    )

    return mul_tensor_10, sum_dim_int_list_2


def triton_oracle(getitem_12, getitem_42, arg61_1, arg57_1, arg58_1, arg59_1,
                  arg9_1, arg10_1, full, _shape_param_0, _shape_param_1, _shape_param_2):
    """Full oracle: upstream (PyTorch) + fused reduction (Triton) + scalar epilogue."""
    where_self, sub_tensor_1, squeeze_dims_1, weight = compute_upstream(
        getitem_12, getitem_42, arg61_1, arg57_1, arg58_1, arg59_1,
        arg9_1, arg10_1, full, _shape_param_0, _shape_param_1, _shape_param_2
    )
    return fused_reduction_oracle(where_self, sub_tensor_1, squeeze_dims_1, weight)


# ---------------------------------------------------------------------------
# Correctness + Benchmark
# ---------------------------------------------------------------------------

def make_inputs(device: torch.device = None) -> tuple:
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
    """Benchmark the fused reduction kernel."""
    inputs = make_inputs(device)

    with torch.no_grad():
        # Precompute upstream (not part of oracle timing for the reduction)
        where_self, sub_tensor_1, squeeze_dims_1, weight = compute_upstream(*inputs)
        torch.cuda.synchronize()

        # Benchmark just the fused reduction + epilogue
        def oracle_fn():
            return fused_reduction_oracle(where_self, sub_tensor_1, squeeze_dims_1, weight)

        ms = triton.testing.do_bench(oracle_fn, warmup=warmup, rep=rep)
        us = ms * 1000.0
        print(f"  oracle (fused 3-acc reduction+epilogue): {us:.1f} us")

        # Benchmark the full oracle for comparison
        def full_oracle_fn():
            return triton_oracle(*inputs)

        ms_full = triton.testing.do_bench(full_oracle_fn, warmup=warmup, rep=rep)
        us_full = ms_full * 1000.0
        print(f"  oracle (full including upstream):        {us_full:.1f} us")

        # Benchmark compiled repro
        module = _load_repro_module()
        repro_instance = module.Repro()

        compiled = torch.compile(repro_instance)
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


def oracle_forward(inputs):
    return triton_oracle(*inputs)


def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true",
                        help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true",
                        help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2,
                        help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2,
                        help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25,
                        help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200,
                        help="Repetitions for benchmark")
    parser.add_argument("--no-skip-stochastic", action="store_true",
                        help="Disable auto-detection and skipping of stochastic outputs")
    parser.add_argument("--all-shapes", action="store_true",
                        help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true",
                        help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = check_oracle(
            oracle_forward,
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
            skip_stochastic=not args.no_skip_stochastic,
        )
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
        else:
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
