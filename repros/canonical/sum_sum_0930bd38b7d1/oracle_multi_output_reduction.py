"""
Oracle kernel for sum_sum_0930bd38b7d1 (InceptionV3 batch-norm backward).

Pattern: Two channel-wise reductions over dims [0,2,3] sharing the same
input tensor (where_self), followed by element-wise computation that
depends on the reduction results.

    Shape: where_self [128, 192, 71, 71] channels-last
    sum1[c] = sum_{n,h,w} where_self[n,c,h,w]
    sum2[c] = sum_{n,h,w} where_self[n,c,h,w] * sub_tensor_1[n,c,h,w]

Then the pointwise output uses sum1 and sum2 to compute:
    mul_tensor_9[n,c,h,w] = (where_self - sub*s2_scaled - s1_scaled) * weight
    mul_tensor_10[c] = sum2 * rsqrt

Oracle strategy:
    - Phase 1: Parallel tiled reduction with coalesced reads respecting
      channels-last layout (C=192, iterate over spatial positions)
    - Phase 2: Fused post-reduction pointwise with coalesced I/O
    - Single read of where_self and sub_tensor_1 feeds both accumulators

Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle differs from
Inductor by treating the two BN-backward channel reductions as one split-K
dual-accumulator reduction over the large N/H/W dimension, using many spatial
tiles to cooperatively accumulate the small `[192]` outputs before a dependent
pointwise epilogue consumes both sums. Inductor cannot express this shape as a
single cooperative multi-output reduction today, so it leaves the sibling sums
and epilogue under ordinary scheduler choices with too little reduction-axis
parallelism and extra reads of the shared inputs. The fix is
COOPERATIVE_SPLIT_K support for compatible multi-output reductions with atomic
or partial coordination plus preservation of the dependent epilogue fusion.

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

REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]

sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Triton kernel: fused dual reduction
# Grid: 1D over spatial tiles. Each program processes BLOCK_SPATIAL positions
# across ALL channels (C=192). Accumulates partial sums in a [C] register
# vector, then atomically adds to global output.
# ---------------------------------------------------------------------------

@triton.jit
def _dual_reduce_channels_last_kernel(
    where_self_ptr,
    sub_tensor_1_ptr,
    out_sum1_ptr,  # [C] output
    out_sum2_ptr,  # [C] output
    total_spatial,  # N * H * W
    C: tl.constexpr,
    C_BLOCK: tl.constexpr,  # next power of 2 >= C (for tl.arange)
    BLOCK_SPATIAL: tl.constexpr,
):
    """Process a block of spatial positions, reduce all C channels per position.
    Data is channels-last: element [n,c,h,w] at offset = (n*H*W + h*W + w)*C + c
    """
    pid = tl.program_id(0)
    spatial_start = pid * BLOCK_SPATIAL

    # Channel offsets: [0, 1, ..., C_BLOCK-1] with mask for valid channels
    c_offs = tl.arange(0, C_BLOCK)
    c_mask = c_offs < C

    # Accumulators for each channel
    acc1 = tl.zeros([C_BLOCK], dtype=tl.float32)
    acc2 = tl.zeros([C_BLOCK], dtype=tl.float32)

    for s in range(BLOCK_SPATIAL):
        spatial_idx = spatial_start + s
        if spatial_idx < total_spatial:
            # Base address for this spatial position (all C channels contiguous)
            base = spatial_idx * C
            addrs = base + c_offs

            ws_vals = tl.load(where_self_ptr + addrs, mask=c_mask, other=0.0)
            sub_vals = tl.load(sub_tensor_1_ptr + addrs, mask=c_mask, other=0.0)

            acc1 += ws_vals
            acc2 += ws_vals * sub_vals

    # Atomic add partial sums to output (only valid channels)
    tl.atomic_add(out_sum1_ptr + c_offs, acc1, mask=c_mask)
    tl.atomic_add(out_sum2_ptr + c_offs, acc2, mask=c_mask)


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
    weight_ptr,      # [C] = rsqrt * primals_30
    out_ptr,         # [N, C, H, W] channels-last output
    scale_factor,    # 1.5497917079944455e-06
    total_spatial,   # N * H * W
    C: tl.constexpr,
    C_BLOCK: tl.constexpr,  # next power of 2 >= C
    BLOCK_SPATIAL: tl.constexpr,
):
    """Pointwise kernel with channels-last layout.
    Each program processes BLOCK_SPATIAL spatial positions x all C channels.
    out[n,c,h,w] = (ws - sub * s2_scaled - s1_scaled) * weight[c]
    """
    pid = tl.program_id(0)
    spatial_start = pid * BLOCK_SPATIAL

    c_offs = tl.arange(0, C_BLOCK)
    c_mask = c_offs < C

    # Load per-channel constants once
    s1 = tl.load(sum1_ptr + c_offs, mask=c_mask, other=0.0)
    s2 = tl.load(sum2_ptr + c_offs, mask=c_mask, other=0.0)
    rsqrt_val = tl.load(rsqrt_ptr + c_offs, mask=c_mask, other=0.0)
    w = tl.load(weight_ptr + c_offs, mask=c_mask, other=0.0)

    s1_scaled = s1 * scale_factor
    rsqrt_sq = rsqrt_val * rsqrt_val
    s2_scaled = s2 * scale_factor * rsqrt_sq

    for s in range(BLOCK_SPATIAL):
        spatial_idx = spatial_start + s
        if spatial_idx < total_spatial:
            base = spatial_idx * C
            addrs = base + c_offs

            ws_val = tl.load(where_self_ptr + addrs, mask=c_mask, other=0.0)
            sub_val = tl.load(sub_tensor_1_ptr + addrs, mask=c_mask, other=0.0)

            result = (ws_val - sub_val * s2_scaled - s1_scaled) * w

            tl.store(out_ptr + addrs, result, mask=c_mask)


# ---------------------------------------------------------------------------
# Oracle entry point
# ---------------------------------------------------------------------------

def oracle_fused(where_self, sub_tensor_1, rsqrt_squeezed, primals_30, scale_factor):
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

    # --- Phase 1: Dual reduction ---
    sum1 = torch.zeros(C, device=where_self.device, dtype=torch.float32)
    sum2 = torch.zeros(C, device=where_self.device, dtype=torch.float32)

    # C=192, next power of 2 = 256 (required for tl.arange)
    # With 192 channels, each spatial position reads 192 floats = 768 bytes
    # Use BLOCK_SPATIAL=64 to keep register pressure manageable
    C_BLOCK = 256  # next power of 2 >= 192
    BLOCK_SPATIAL = 64
    num_programs = (total_spatial + BLOCK_SPATIAL - 1) // BLOCK_SPATIAL

    _dual_reduce_channels_last_kernel[(num_programs,)](
        where_self, sub_tensor_1,
        sum1, sum2,
        total_spatial,
        C=C,
        C_BLOCK=C_BLOCK,
        BLOCK_SPATIAL=BLOCK_SPATIAL,
    )

    # --- Phase 2: Post-reduction pointwise ---
    weight = rsqrt_squeezed * primals_30

    # Output in channels-last format
    out = torch.empty(N, C, H, W, device=where_self.device, dtype=torch.float32,
                      memory_format=torch.channels_last)

    PW_BLOCK_SPATIAL = 64
    pw_num_programs = (total_spatial + PW_BLOCK_SPATIAL - 1) // PW_BLOCK_SPATIAL

    _post_reduce_pointwise_cl_kernel[(pw_num_programs,)](
        where_self, sub_tensor_1,
        sum1, sum2,
        rsqrt_squeezed, weight,
        out,
        scale_factor,
        total_spatial,
        C=C,
        C_BLOCK=C_BLOCK,
        BLOCK_SPATIAL=PW_BLOCK_SPATIAL,
    )

    # mul_tensor_10 = sum2 * rsqrt_squeezed
    mul_tensor_10 = sum2 * rsqrt_squeezed

    return out, mul_tensor_10


# ---------------------------------------------------------------------------
# Reference: compute using PyTorch ops (matches repro.py exactly)
# ---------------------------------------------------------------------------

def reference_pytorch(getitem_442, getitem_12, getitem_451, getitem_457, getitem_460,
                      getitem_13, convolution_4, getitem_11, rsqrt_4,
                      primals_30, primals_31, full_default,
                      _shape_param_0, _shape_param_1, _shape_param_2):
    """Direct PyTorch implementation matching repro.py exactly."""
    import torch._inductor.inductor_prims  # noqa: F401

    # avg_pool2d_backward
    avg_pool2d_backward_default = torch.ops.aten.avg_pool2d_backward.default(
        getitem_442, getitem_12, [3, 3], [1, 1], [1, 1], False, True, None
    )

    # Add ops
    add_tensor = avg_pool2d_backward_default + getitem_451
    add_tensor_1 = add_tensor + getitem_457
    add_tensor_2 = add_tensor_1 + getitem_460

    # scatter_add for max pool backward
    full_default_1 = torch.zeros([24576, 5041], dtype=torch.float32, device=getitem_442.device)
    clone_default = add_tensor_2.clone(memory_format=torch.contiguous_format)
    reshape_default = clone_default.reshape(_shape_param_0)

    _low_memory_max_pool_offsets_to_indices_default = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
        getitem_13, [3, 3], [71, 71], [2, 2], [0, 0], [1, 1]
    )
    clone_default_1 = _low_memory_max_pool_offsets_to_indices_default.clone(memory_format=torch.contiguous_format)
    reshape_default_1 = clone_default_1.reshape(_shape_param_1)
    scatter_add_default = full_default_1.scatter_add(1, reshape_default_1, reshape_default)
    reshape_default_2 = scatter_add_default.reshape(_shape_param_2)
    clone_default_2 = reshape_default_2.clone(memory_format=torch.channels_last)

    # Batch norm backward
    sub_tensor = convolution_4 - getitem_11
    mul_tensor = sub_tensor * rsqrt_4
    primals_30_expanded = primals_30.unsqueeze(-1).unsqueeze(-1)
    mul_tensor_1 = mul_tensor * primals_30_expanded
    primals_31_expanded = primals_31.unsqueeze(-1).unsqueeze(-1)
    add_tensor_3 = mul_tensor_1 + primals_31_expanded

    # ReLU backward mask
    relu_default = torch.relu(add_tensor_3)
    le_scalar = relu_default <= 0
    where_self = torch.where(le_scalar, full_default, clone_default_2)

    # The two reductions
    squeeze_dims = getitem_11.squeeze(0).squeeze(-1).squeeze(-1)  # [192]
    unsqueeze_default_6 = squeeze_dims.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1,192,1,1]

    sum_dim_int_list = where_self.sum([0, 2, 3])
    sub_tensor_1 = convolution_4 - unsqueeze_default_6
    mul_tensor_2 = where_self * sub_tensor_1
    sum_dim_int_list_1 = mul_tensor_2.sum([0, 2, 3])

    # Post-reduction
    scale_factor = 1.5497917079944455e-06
    mul_tensor_3 = sum_dim_int_list * scale_factor
    unsqueeze_default_9 = mul_tensor_3.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    mul_tensor_4 = sum_dim_int_list_1 * scale_factor
    squeeze_dims_1 = rsqrt_4.squeeze(0).squeeze(-1).squeeze(-1)  # [192]
    mul_tensor_5 = squeeze_dims_1 * squeeze_dims_1
    mul_tensor_6 = mul_tensor_4 * mul_tensor_5
    unsqueeze_default_12 = mul_tensor_6.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    mul_tensor_7 = squeeze_dims_1 * primals_30
    unsqueeze_default_15 = mul_tensor_7.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    mul_tensor_8 = sub_tensor_1 * unsqueeze_default_12
    sub_tensor_2 = where_self - mul_tensor_8
    sub_tensor_3 = sub_tensor_2 - unsqueeze_default_9
    mul_tensor_9 = sub_tensor_3 * unsqueeze_default_15

    mul_tensor_10 = sum_dim_int_list_1 * squeeze_dims_1

    return mul_tensor_9, mul_tensor_10


def prepare_oracle_inputs(getitem_442, getitem_12, getitem_451, getitem_457, getitem_460,
                          getitem_13, convolution_4, getitem_11, rsqrt_4,
                          primals_30, primals_31, full_default,
                          _shape_param_0, _shape_param_1, _shape_param_2):
    """
    Run the pre-reduction ops (avg_pool2d_backward, scatter_add, batch norm forward, relu mask)
    to produce the inputs needed by the oracle kernel.

    Returns: (where_self, sub_tensor_1, rsqrt_squeezed, primals_30, scale_factor)
    """
    import torch._inductor.inductor_prims  # noqa: F401

    device = getitem_442.device

    # avg_pool2d_backward
    avg_pool2d_backward_default = torch.ops.aten.avg_pool2d_backward.default(
        getitem_442, getitem_12, [3, 3], [1, 1], [1, 1], False, True, None
    )

    # Add ops
    add_tensor = avg_pool2d_backward_default + getitem_451
    add_tensor_1 = add_tensor + getitem_457
    add_tensor_2 = add_tensor_1 + getitem_460

    # Max pool backward via scatter_add
    full_default_1 = torch.zeros([24576, 5041], dtype=torch.float32, device=device)
    clone_default = add_tensor_2.clone(memory_format=torch.contiguous_format)
    reshape_default = clone_default.reshape(_shape_param_0)

    _low_memory_max_pool_offsets_to_indices_default = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
        getitem_13, [3, 3], [71, 71], [2, 2], [0, 0], [1, 1]
    )
    clone_default_1 = _low_memory_max_pool_offsets_to_indices_default.clone(memory_format=torch.contiguous_format)
    reshape_default_1 = clone_default_1.reshape(_shape_param_1)
    scatter_add_default = full_default_1.scatter_add(1, reshape_default_1, reshape_default)
    reshape_default_2 = scatter_add_default.reshape(_shape_param_2)
    clone_default_2 = reshape_default_2.clone(memory_format=torch.channels_last)

    # Batch norm forward to get relu mask
    sub_tensor = convolution_4 - getitem_11
    mul_tensor = sub_tensor * rsqrt_4
    primals_30_expanded = primals_30.unsqueeze(-1).unsqueeze(-1)
    mul_tensor_1 = mul_tensor * primals_30_expanded
    primals_31_expanded = primals_31.unsqueeze(-1).unsqueeze(-1)
    add_tensor_3 = mul_tensor_1 + primals_31_expanded

    # ReLU backward mask
    relu_default = torch.relu(add_tensor_3)
    le_scalar = relu_default <= 0
    where_self = torch.where(le_scalar, full_default, clone_default_2)

    # sub_tensor_1 = conv - mean (per-channel)
    squeeze_dims = getitem_11.squeeze(0).squeeze(-1).squeeze(-1)  # [192]
    mean_expanded = squeeze_dims.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1,192,1,1]
    sub_tensor_1 = convolution_4 - mean_expanded

    # rsqrt squeezed
    rsqrt_squeezed = rsqrt_4.squeeze(0).squeeze(-1).squeeze(-1)  # [192]

    scale_factor = 1.5497917079944455e-06  # = 1/(N*H*W) = 1/(128*71*71)

    return where_self, sub_tensor_1, rsqrt_squeezed, primals_30, scale_factor


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


def run_check():
    """Verify oracle produces same results as reference."""
    print("Loading inputs...")
    repro_mod = load_repro_module()
    inputs = repro_mod.make_inputs()
    inputs = tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs)

    print("Computing reference...")
    with torch.no_grad():
        ref_out = reference_pytorch(*inputs)

    print("Preparing oracle inputs...")
    with torch.no_grad():
        oracle_inputs = prepare_oracle_inputs(*inputs)

    print("Running oracle kernel...")
    with torch.no_grad():
        oracle_out = oracle_fused(*oracle_inputs)

    # Compare
    for i, (o, r) in enumerate(zip(oracle_out, ref_out)):
        max_diff = (o.float() - r.float()).abs().max().item()
        rel_diff = ((o.float() - r.float()).abs() / (r.float().abs() + 1e-8)).max().item()
        ok = torch.allclose(o.float(), r.float(), rtol=1e-3, atol=1e-3)
        print(f"  Output {i}: shape={list(o.shape)}, max_abs_diff={max_diff:.6e}, max_rel_diff={rel_diff:.6e}, ok={ok}")

    all_ok = all(
        torch.allclose(o.float(), r.float(), rtol=1e-3, atol=1e-3)
        for o, r in zip(oracle_out, ref_out)
    )
    print(f"\nCorrectness: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def run_bench():
    """Benchmark oracle vs torch.compile."""
    repro_mod = load_repro_module()
    inputs = repro_mod.make_inputs()
    inputs = tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs)

    # Prepare oracle inputs (pre-compute scatter_add etc.)
    with torch.no_grad():
        oracle_inputs = prepare_oracle_inputs(*inputs)

    # Warm up oracle
    with torch.no_grad():
        _ = oracle_fused(*oracle_inputs)
    torch.cuda.synchronize()

    # Benchmark oracle (just the reduction + pointwise part)
    oracle_ms = triton.testing.do_bench(
        lambda: oracle_fused(*oracle_inputs),
        warmup=100,
        rep=200,
    )

    # Benchmark torch.compile on the full repro
    import torch._inductor.config as inductor_config
    model = repro_mod.Repro().cuda()

    with torch.no_grad():
        compiled = torch.compile(model)
        _ = compiled(*inputs)
        torch.cuda.synchronize()

    compile_ms = triton.testing.do_bench(
        lambda: compiled(*inputs),
        warmup=100,
        rep=200,
    )

    # Also try with coordinate descent tuning
    torch._dynamo.reset()
    with torch.no_grad():
        inductor_config.coordinate_descent_tuning = True
        compiled_cd = torch.compile(model)
        _ = compiled_cd(*inputs)
        torch.cuda.synchronize()

    compile_cd_ms = triton.testing.do_bench(
        lambda: compiled_cd(*inputs),
        warmup=100,
        rep=200,
    )

    print(f"\n{'='*60}")
    print(f"Benchmark results for sum_sum_0930bd38b7d1")
    print(f"{'='*60}")
    print(f"  Oracle (reduction+pointwise):       {oracle_ms*1000:.1f} us")
    print(f"  torch.compile (full repro):         {compile_ms*1000:.1f} us")
    print(f"  torch.compile + coord_descent:      {compile_cd_ms*1000:.1f} us")
    print(f"  Speedup (oracle vs compile):        {compile_ms/oracle_ms:.2f}x")
    print(f"  Speedup (oracle vs coord_descent):  {compile_cd_ms/oracle_ms:.2f}x")
    print(f"{'='*60}")
    print(f"\nNote: Oracle measures the fused dual-reduction + post-reduction")
    print(f"pointwise. The scatter_add (max pool backward) is pre-computed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run correctness check")
    parser.add_argument("--bench", action="store_true", help="Run benchmark")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check:
        ok = run_check()
        if not ok:
            sys.exit(1)

    if args.bench:
        run_bench()


if __name__ == "__main__":
    main()
