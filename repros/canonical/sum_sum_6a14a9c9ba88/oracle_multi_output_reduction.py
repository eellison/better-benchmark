"""
Oracle kernel for sum_sum_6a14a9c9ba88 (InceptionV3 batch-norm backward).

Pattern: Two channel-wise reductions over dims [0,2,3] sharing the same
input tensor (where_self), followed by element-wise computation that
depends on the reduction results.

    sum1[c] = sum_{n,h,w} where_self[n,c,h,w]
    sum2[c] = sum_{n,h,w} where_self[n,c,h,w] * (conv[n,c,h,w] - mean[c])

Then the pointwise output uses sum1 and sum2 to compute a per-element result:
    out[n,c,h,w] = (where_self - sub*s2_scaled - s1_scaled) * weight

Oracle strategy:
    - Phase 1: Parallel tiled reduction with coalesced reads respecting
      channels-last layout (C=64 fits in registers, iterate over spatial)
    - Phase 2: Fused post-reduction pointwise with coalesced I/O
    - Single read of where_self and sub_tensor_1 feeds both accumulators

The full repro also includes a scatter_add (max pool backward) that produces
where_self. That scatter is not part of this oracle; we take where_self as
given (as Inductor would separate the scatter into its own kernel).
"""
from __future__ import annotations

import argparse
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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"
REPO_ROOT = REPRO_DIR.parents[2]



# ---------------------------------------------------------------------------
# Triton kernel: fused dual reduction
# Grid: 1D over spatial tiles. Each program processes BLOCK_SPATIAL positions
# across ALL channels (C=64, fits in registers). Accumulates partial sums
# in a [C] register vector, then atomically adds to global output.
# ---------------------------------------------------------------------------

@triton.jit
def _dual_reduce_channels_last_kernel(
    where_self_ptr,
    sub_tensor_1_ptr,
    out_sum1_ptr,  # [C] output
    out_sum2_ptr,  # [C] output
    total_spatial,  # N * H * W
    C: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
):
    """Process a block of spatial positions, reduce all C channels per position.
    Data is channels-last: element [n,c,h,w] at offset = (n*H*W + h*W + w)*C + c
    """
    pid = tl.program_id(0)
    spatial_start = pid * BLOCK_SPATIAL

    # Channel offsets: [0, 1, ..., C-1]
    c_offs = tl.arange(0, C)  # [C]

    # Accumulators for each channel
    acc1 = tl.zeros([C], dtype=tl.float32)
    acc2 = tl.zeros([C], dtype=tl.float32)

    for s in range(BLOCK_SPATIAL):
        spatial_idx = spatial_start + s
        if spatial_idx < total_spatial:
            # Base address for this spatial position (all C channels contiguous)
            base = spatial_idx * C
            addrs = base + c_offs

            ws_vals = tl.load(where_self_ptr + addrs)
            sub_vals = tl.load(sub_tensor_1_ptr + addrs)

            acc1 += ws_vals
            acc2 += ws_vals * sub_vals

    # Atomic add partial sums to output
    tl.atomic_add(out_sum1_ptr + c_offs, acc1)
    tl.atomic_add(out_sum2_ptr + c_offs, acc2)


# ---------------------------------------------------------------------------
# Triton kernel: post-reduction pointwise (channels-last I/O)
# ---------------------------------------------------------------------------

@triton.jit
def _post_reduce_pointwise_cl_kernel(
    where_self_ptr,
    sub_tensor_1_ptr,
    sum1_ptr,        # [C]
    sum2_ptr,        # [C]
    rsqrt_ptr,       # [C]
    weight_ptr,      # [C] = rsqrt * primals_18
    out_ptr,         # [N, C, H, W] channels-last output
    scale_factor,    # 3.6153917349252627e-07
    total_spatial,   # N * H * W
    C: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
):
    """Pointwise kernel with channels-last layout.
    Each program processes BLOCK_SPATIAL spatial positions x all C channels.
    out[n,c,h,w] = (ws - sub * s2_scaled - s1_scaled) * weight[c]
    """
    pid = tl.program_id(0)
    spatial_start = pid * BLOCK_SPATIAL

    c_offs = tl.arange(0, C)  # [C]

    # Load per-channel constants once
    s1 = tl.load(sum1_ptr + c_offs)
    s2 = tl.load(sum2_ptr + c_offs)
    rsqrt_val = tl.load(rsqrt_ptr + c_offs)
    w = tl.load(weight_ptr + c_offs)

    s1_scaled = s1 * scale_factor
    rsqrt_sq = rsqrt_val * rsqrt_val
    s2_scaled = s2 * scale_factor * rsqrt_sq

    for s in range(BLOCK_SPATIAL):
        spatial_idx = spatial_start + s
        if spatial_idx < total_spatial:
            base = spatial_idx * C
            addrs = base + c_offs

            ws_val = tl.load(where_self_ptr + addrs)
            sub_val = tl.load(sub_tensor_1_ptr + addrs)

            result = (ws_val - sub_val * s2_scaled - s1_scaled) * w

            tl.store(out_ptr + addrs, result)


# ---------------------------------------------------------------------------
# Oracle entry point
# ---------------------------------------------------------------------------

def oracle_fused(where_self, sub_tensor_1, rsqrt_squeezed, primals_18, scale_factor):
    """
    Compute the two reductions and post-reduction pointwise in two Triton kernels.

    Returns: (mul_tensor_9, mul_tensor_10)
        mul_tensor_9: [N, C, H, W] channels-last - post-reduction pointwise
        mul_tensor_10: [C] - sum2 * rsqrt (the second scalar output)
    """
    N, C, H, W = where_self.shape
    total_spatial = N * H * W  # number of spatial positions

    # Verify channels-last layout: stride(1) == 1, stride(3) == C
    assert where_self.stride(1) == 1, f"Expected channels-last, got stride(1)={where_self.stride(1)}"
    assert sub_tensor_1.stride(1) == 1, f"Expected channels-last, got stride(1)={sub_tensor_1.stride(1)}"

    # Make contiguous channels-last views for the kernel
    # where_self: [128, 64, 147, 147] channels-last -> underlying is (N*H*W, C) row-major
    # sub_tensor_1: same layout

    # --- Phase 1: Dual reduction ---
    sum1 = torch.zeros(C, device=where_self.device, dtype=torch.float32)
    sum2 = torch.zeros(C, device=where_self.device, dtype=torch.float32)

    BLOCK_SPATIAL = 128  # Each program processes 128 spatial positions
    num_programs = (total_spatial + BLOCK_SPATIAL - 1) // BLOCK_SPATIAL

    _dual_reduce_channels_last_kernel[(num_programs,)](
        where_self, sub_tensor_1,
        sum1, sum2,
        total_spatial,
        C=C,
        BLOCK_SPATIAL=BLOCK_SPATIAL,
    )

    # --- Phase 2: Post-reduction pointwise ---
    weight = rsqrt_squeezed * primals_18

    # Output in channels-last format
    out = torch.empty(N, C, H, W, device=where_self.device, dtype=torch.float32,
                      memory_format=torch.channels_last)

    PW_BLOCK_SPATIAL = 128
    pw_num_programs = (total_spatial + PW_BLOCK_SPATIAL - 1) // PW_BLOCK_SPATIAL

    _post_reduce_pointwise_cl_kernel[(pw_num_programs,)](
        where_self, sub_tensor_1,
        sum1, sum2,
        rsqrt_squeezed, weight,
        out,
        scale_factor,
        total_spatial,
        C=C,
        BLOCK_SPATIAL=PW_BLOCK_SPATIAL,
    )

    # mul_tensor_10 = sum2 * rsqrt_squeezed
    mul_tensor_10 = sum2 * rsqrt_squeezed

    return out, mul_tensor_10


# ---------------------------------------------------------------------------
# Reference: compute using PyTorch ops (matches repro.py exactly)
# ---------------------------------------------------------------------------

def reference_pytorch(getitem_466, getitem_7, convolution_2, getitem_5, rsqrt_2,
                      primals_18, primals_19, full_default,
                      _shape_param_0, _shape_param_1, _shape_param_2):
    """Direct PyTorch implementation matching repro.py exactly."""
    import torch._inductor.inductor_prims  # noqa: F401

    # Max pool backward via scatter_add
    full_default_1 = torch.zeros([8192, 21609], dtype=torch.float32, device=getitem_466.device)
    clone_default = getitem_466.clone(memory_format=torch.contiguous_format)
    reshape_default = clone_default.reshape(_shape_param_0)

    _low_memory_max_pool_offsets_to_indices_default = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
        getitem_7, [3, 3], [147, 147], [2, 2], [0, 0], [1, 1]
    )
    clone_default_1 = _low_memory_max_pool_offsets_to_indices_default.clone(memory_format=torch.contiguous_format)
    reshape_default_1 = clone_default_1.reshape(_shape_param_1)
    scatter_add_default = full_default_1.scatter_add(1, reshape_default_1, reshape_default)
    reshape_default_2 = scatter_add_default.reshape(_shape_param_2)
    clone_default_2 = reshape_default_2.clone(memory_format=torch.channels_last)

    # Batch norm forward
    sub_tensor = convolution_2 - getitem_5
    mul_tensor = sub_tensor * rsqrt_2
    primals_18_expanded = primals_18.unsqueeze(-1).unsqueeze(-1)
    mul_tensor_1 = mul_tensor * primals_18_expanded
    primals_19_expanded = primals_19.unsqueeze(-1).unsqueeze(-1)
    add_tensor = mul_tensor_1 + primals_19_expanded

    # ReLU backward mask
    relu_default = torch.relu(add_tensor)
    le_scalar = relu_default <= 0
    where_self = torch.where(le_scalar, full_default, clone_default_2)

    # The two reductions
    squeeze_dims = getitem_5.squeeze(0).squeeze(-1).squeeze(-1)  # [64]
    unsqueeze_default_6 = squeeze_dims.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1,64,1,1]

    sum_dim_int_list = where_self.sum([0, 2, 3])
    sub_tensor_1 = convolution_2 - unsqueeze_default_6
    mul_tensor_2 = where_self * sub_tensor_1
    sum_dim_int_list_1 = mul_tensor_2.sum([0, 2, 3])

    # Post-reduction
    scale_factor = 3.6153917349252627e-07
    mul_tensor_3 = sum_dim_int_list * scale_factor
    unsqueeze_default_9 = mul_tensor_3.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    mul_tensor_4 = sum_dim_int_list_1 * scale_factor
    squeeze_dims_1 = rsqrt_2.squeeze(0).squeeze(-1).squeeze(-1)  # [64]
    mul_tensor_5 = squeeze_dims_1 * squeeze_dims_1
    mul_tensor_6 = mul_tensor_4 * mul_tensor_5
    unsqueeze_default_12 = mul_tensor_6.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    mul_tensor_7 = squeeze_dims_1 * primals_18
    unsqueeze_default_15 = mul_tensor_7.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    mul_tensor_8 = sub_tensor_1 * unsqueeze_default_12
    sub_tensor_2 = where_self - mul_tensor_8
    sub_tensor_3 = sub_tensor_2 - unsqueeze_default_9
    mul_tensor_9 = sub_tensor_3 * unsqueeze_default_15

    mul_tensor_10 = sum_dim_int_list_1 * squeeze_dims_1

    return mul_tensor_9, mul_tensor_10


def prepare_oracle_inputs(getitem_466, getitem_7, convolution_2, getitem_5, rsqrt_2,
                          primals_18, primals_19, full_default,
                          _shape_param_0, _shape_param_1, _shape_param_2):
    """
    Run the pre-reduction ops (scatter_add, batch norm forward, relu mask)
    to produce the inputs needed by the oracle kernel.

    Returns: (where_self, sub_tensor_1, rsqrt_squeezed, primals_18, scale_factor)
    """
    import torch._inductor.inductor_prims  # noqa: F401

    # Max pool backward via scatter_add
    full_default_1 = torch.zeros([8192, 21609], dtype=torch.float32, device=getitem_466.device)
    clone_default = getitem_466.clone(memory_format=torch.contiguous_format)
    reshape_default = clone_default.reshape(_shape_param_0)

    _low_memory_max_pool_offsets_to_indices_default = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
        getitem_7, [3, 3], [147, 147], [2, 2], [0, 0], [1, 1]
    )
    clone_default_1 = _low_memory_max_pool_offsets_to_indices_default.clone(memory_format=torch.contiguous_format)
    reshape_default_1 = clone_default_1.reshape(_shape_param_1)
    scatter_add_default = full_default_1.scatter_add(1, reshape_default_1, reshape_default)
    reshape_default_2 = scatter_add_default.reshape(_shape_param_2)
    clone_default_2 = reshape_default_2.clone(memory_format=torch.channels_last)

    # Batch norm forward to get relu mask
    sub_tensor = convolution_2 - getitem_5
    mul_tensor = sub_tensor * rsqrt_2
    primals_18_expanded = primals_18.unsqueeze(-1).unsqueeze(-1)
    mul_tensor_1 = mul_tensor * primals_18_expanded
    primals_19_expanded = primals_19.unsqueeze(-1).unsqueeze(-1)
    add_tensor = mul_tensor_1 + primals_19_expanded

    # ReLU backward mask
    relu_default = torch.relu(add_tensor)
    le_scalar = relu_default <= 0
    where_self = torch.where(le_scalar, full_default, clone_default_2)

    # sub_tensor_1 = conv - mean (per-channel)
    squeeze_dims = getitem_5.squeeze(0).squeeze(-1).squeeze(-1)
    mean_expanded = squeeze_dims.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    sub_tensor_1 = convolution_2 - mean_expanded

    # rsqrt squeezed
    rsqrt_squeezed = rsqrt_2.squeeze(0).squeeze(-1).squeeze(-1)

    scale_factor = 3.6153917349252627e-07

    return where_self, sub_tensor_1, rsqrt_squeezed, primals_18, scale_factor


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_repro_module():
    """Load the repro module to get Repro class and make_inputs."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("repro_mod", REPRO_DIR / "repro.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def oracle_forward(inputs):
    return oracle_fused(*inputs)


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
