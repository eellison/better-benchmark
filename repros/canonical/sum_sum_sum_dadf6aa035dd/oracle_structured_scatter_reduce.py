"""
Oracle for repro sum_sum_sum_dadf6aa035dd (structured_pool_upsample_backward_reduce, PyTorch-UNet).

Pattern:
    Bilinear upsample backward (4x index_put with accumulate=True) -> BN-affine/ReLU mask
    -> 3 channel reductions (sum, sum*centered, final BN-backward linear combination).

    Input shapes: [8, 512, 160, 239] -> slice channels 256:512 -> pad -> [8, 256, 160, 238]
    Scatter output: [8, 256, 80, 119]
    Index shapes: row_idx [160, 1], col_idx [238]

Key optimization insight:
    The original graph materializes a [8, 256, 80, 119] scattered intermediate from
    4 separate index_put operations on a [8, 256, 160, 238] source tensor. This
    intermediate is then gated by BN-affine/ReLU and reduced per channel.

    Since the final operation reduces over batch and spatial dims to produce per-channel
    vectors, we fuse:
    1. The 4 index_put scatters (bilinear upsample backward)
    2. The BN-backward reductions into algebraic combinations of 3 partial sums,
       avoiding multiple full passes over the [8, 256, 80, 119] scattered tensor.

Implementation:
    torch_fused_bilinear_scatter_reduce: Pure PyTorch implementation that computes
    the bilinear scatter and fuses the 3 BN-backward reductions algebraically.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)

# Scale factor from the repro: 1 / (N * H_out * W_out) = 1 / (8 * 80 * 119)
CHANNEL_REDUCTION_SCALE = 1.3130252100840337e-05


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle implementation ---

def structured_bilinear_index_put(
    getitem_18: torch.Tensor,
    row_weight: torch.Tensor,
    col_weight: torch.Tensor,
    row_hi: torch.Tensor,
    col_hi: torch.Tensor,
    col_lo: torch.Tensor,
    row_lo: torch.Tensor,
) -> torch.Tensor:
    """Compute the 4 bilinear index_put accumulations matching the repro exactly.

    The repro does:
      src = getitem_18[:, 256:512, :, :-1]  -> [8, 256, 160, 238]
      constant_pad_nd removes the last column -> [8, 256, 160, 238]
      Then computes bilinear weights and their complements, and scatters 4 ways:
        val * row_weight * col_weight       -> index_put at (row_hi, col_hi)
        val * row_weight * (1 - col_weight) -> index_put at (row_hi, col_lo)
        val * (1 - row_weight) * col_weight -> index_put at (row_lo, col_hi)
        val * (1 - row_weight) * (1 - col_weight) -> index_put at (row_lo, col_lo)
    """
    # Source: slice channels 256:512 then pad (remove last col)
    src = getitem_18[:, 256:512, :, :-1]  # [8, 256, 160, 238]

    # Compute the 4 weighted contributions
    val_rw = src * row_weight
    val_1mrw = src - val_rw

    val_rw_cw = val_rw * col_weight
    val_rw_1mcw = val_rw - val_rw_cw
    val_1mrw_cw = val_1mrw * col_weight
    val_1mrw_1mcw = val_1mrw - val_1mrw_cw

    # Output shape
    N, C = src.shape[0], src.shape[1]
    out_H, out_W = 80, 119
    out = torch.zeros((N, C, out_H, out_W), device=src.device, dtype=src.dtype)

    # Scatter all 4 contributions (accumulate=True for overlapping indices)
    out = torch.ops.aten.index_put.default(out, [None, None, row_hi, col_hi], val_rw_cw, True)
    out = torch.ops.aten.index_put.default(out, [None, None, row_hi, col_lo], val_rw_1mcw, True)
    out = torch.ops.aten.index_put.default(out, [None, None, row_lo, col_hi], val_1mrw_cw, True)
    out = torch.ops.aten.index_put.default(out, [None, None, row_lo, col_lo], val_1mrw_1mcw, True)

    return out


def torch_fused_bilinear_scatter_reduce(
    getitem_18: torch.Tensor,    # f32[8, 512, 160, 239]
    arg102_1: torch.Tensor,      # f32[160, 1] - row weights
    arg101_1: torch.Tensor,      # f32[238] - col weights
    arg98_1: torch.Tensor,       # i64[160, 1] - row_hi indices
    arg100_1: torch.Tensor,      # i64[238] - col_hi indices
    arg99_1: torch.Tensor,       # i64[238] - col_lo indices
    arg97_1: torch.Tensor,       # i64[160, 1] - row_lo indices
    arg94_1: torch.Tensor,       # f32[8, 256, 80, 119] - BN input (x)
    arg95_1: torch.Tensor,       # f32[1, 256, 1, 1] - BN running mean
    arg96_1: torch.Tensor,       # f32[1, 256, 1, 1] - BN inv_std
    arg29_1: torch.Tensor,       # f32[256] - BN weight (gamma)
    arg30_1: torch.Tensor,       # f32[256] - BN bias (beta)
    full: torch.Tensor,          # f32[] - scalar for where masking
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused bilinear scatter + BN backward reduction oracle.

    Computes:
      1. Bilinear upsample backward via 4 index_put accumulations -> scatter_result [8,256,80,119]
      2. BN-affine/ReLU gate: where(relu(BN(x)) <= 0, full, scatter_result) -> masked
      3. Three reductions fused into two outputs:
         - out0 = sum(masked * centered, [0,2,3]) * inv_std  (per channel)
         - out1 = sum(normalized_grad * affine_scale, [0,2,3])  (per channel)
    """
    # Phase 1: Compute scatter result (bilinear upsample backward)
    scatter_result = structured_bilinear_index_put(
        getitem_18, arg102_1, arg101_1, arg98_1, arg100_1, arg99_1, arg97_1
    )

    # Phase 2: Compute BN-affine/ReLU mask
    # bn_affine = (arg94_1 - mean) * inv_std * weight + bias
    sub_tensor = arg94_1 - arg95_1
    mul_tensor_3 = sub_tensor * arg96_1
    weight_4d = arg29_1[None, :, None, None]
    bias_4d = arg30_1[None, :, None, None]
    bn_affine = mul_tensor_3 * weight_4d + bias_4d
    relu_gate = torch.relu(bn_affine)
    le_mask = relu_gate <= 0

    # Apply mask: where gate is active, use full scalar; else use scatter_result
    masked = torch.where(le_mask, full, scatter_result)

    # Phase 3: BN backward reductions (fused algebraically)
    # centered = arg94_1 - mean (same as sub_tensor)
    centered = sub_tensor  # [8, 256, 80, 119]
    inv_std_vec = arg96_1.squeeze((0, 2, 3))  # [256]

    # Reduction 1: sum(masked, [0,2,3])
    masked_sum = masked.sum(dim=(0, 2, 3))  # [256]

    # Reduction 2: sum(masked * centered, [0,2,3])
    masked_centered_sum = (masked * centered).sum(dim=(0, 2, 3))  # [256]

    # Output 0: mul_tensor_13 = sum_dim_int_list_1 * squeeze_dims_1
    # = masked_centered_sum * inv_std_vec
    out0 = masked_centered_sum * inv_std_vec

    # Output 1: sum(normalized_grad * affine_scale, [0,2,3])
    # Algebraic fusion avoids 3rd full pass:
    # out1 = affine_scale * (masked_sum - var_term * centered_sum - mean_term * N_HW)
    mean_term = masked_sum * CHANNEL_REDUCTION_SCALE  # [256]
    var_term = masked_centered_sum * CHANNEL_REDUCTION_SCALE * inv_std_vec * inv_std_vec  # [256]
    affine_scale = inv_std_vec * arg29_1  # [256]

    centered_sum = centered.sum(dim=(0, 2, 3))  # [256]
    N_HW = 8 * 80 * 119  # total spatial elements per channel

    out1 = affine_scale * (masked_sum - var_term * centered_sum - mean_term * N_HW)

    return (out0, out1)


def oracle_forward(inputs):
    """Run the fused bilinear scatter + BN backward reduction oracle.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: list of tensors from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    return torch_fused_bilinear_scatter_reduce(*inputs)


# --- CLI entry point ---
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

    # Handle --show-hw early
    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
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
                oracle_forward, REPRO_DIR, REPRO_ID,
                warmup=args.warmup, rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
