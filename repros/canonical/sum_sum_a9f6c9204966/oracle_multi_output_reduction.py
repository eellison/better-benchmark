"""
Oracle kernel for sum_sum_a9f6c9204966 (Inception V3 batch-norm backward).

Pattern: Two channel-wise reductions over dims [0,2,3] sharing the same
input tensor (where_self), followed by element-wise computation that
depends on the reduction results.

    sum1[c] = sum_{n,h,w} where_self[n,c,h,w]
    sum2[c] = sum_{n,h,w} where_self[n,c,h,w] * (conv[n,c,h,w] - mean[c])

Then the pointwise output uses sum1 and sum2:
    out[n,c,h,w] = (where_self - sub*s2_scaled - s1_scaled) * weight[c]

Final outputs: (mul_tensor_9, mul_tensor_10)
    mul_tensor_9: [128, 192, 71, 71] channels-last
    mul_tensor_10: [192] = sum2 * rsqrt

Oracle strategy:
    - Phase 1: Parallel tiled reduction with coalesced reads for channels-last
      layout (C=192, iterate over spatial). Each program processes a block of
      spatial positions and accumulates per-channel partial sums.
    - Phase 2: Fused post-reduction pointwise with coalesced channels-last I/O.
    - Single read of where_self and sub_tensor_1 feeds both accumulators.

The full repro includes scatter_add (max pool backward) + avg_pool2d_backward.
Those are pre-computed; we take where_self as given.

Shape: [128, 192, 71, 71], C=192, channels-last
N*H*W = 128*71*71 = 645,248
scale_factor = 1.5497917079944455e-06 = 1/(128*71*71)
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



def next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


# ---------------------------------------------------------------------------
# Triton kernel: fused dual reduction (channels-last layout)
# For channels-last: element [n,c,h,w] at offset = (n*H*W + h*W + w)*C + c
# ---------------------------------------------------------------------------

@triton.jit
def _dual_reduce_channels_last_kernel(
    where_self_ptr,
    sub_tensor_1_ptr,
    out_sum1_ptr,  # [C] output
    out_sum2_ptr,  # [C] output
    total_spatial,  # N * H * W
    C: tl.constexpr,
    C_BLOCK: tl.constexpr,  # next power of 2 >= C
    stride_spatial,  # stride between consecutive spatial positions (= C for CL, 1 for contiguous)
    stride_c,        # stride along channel dim (= 1 for CL, H*W for contiguous)
    BLOCK_SPATIAL: tl.constexpr,
):
    """Process a block of spatial positions, reduce all C channels per position."""
    pid = tl.program_id(0)
    spatial_start = pid * BLOCK_SPATIAL

    c_offs = tl.arange(0, C_BLOCK)
    c_mask = c_offs < C

    acc1 = tl.zeros([C_BLOCK], dtype=tl.float32)
    acc2 = tl.zeros([C_BLOCK], dtype=tl.float32)

    for s in range(BLOCK_SPATIAL):
        spatial_idx = spatial_start + s
        if spatial_idx < total_spatial:
            base = spatial_idx * stride_spatial
            addrs = base + c_offs * stride_c

            ws_vals = tl.load(where_self_ptr + addrs, mask=c_mask, other=0.0)
            sub_vals = tl.load(sub_tensor_1_ptr + addrs, mask=c_mask, other=0.0)

            acc1 += ws_vals
            acc2 += ws_vals * sub_vals

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
    weight_ptr,      # [C] = rsqrt * primals_weight
    out_ptr,
    scale_factor,
    total_spatial,
    C: tl.constexpr,
    C_BLOCK: tl.constexpr,  # next power of 2 >= C
    stride_spatial,  # stride between consecutive spatial positions
    stride_c,        # stride along channel dim
    BLOCK_SPATIAL: tl.constexpr,
):
    """Pointwise kernel supporting both contiguous and channels-last layout."""
    pid = tl.program_id(0)
    spatial_start = pid * BLOCK_SPATIAL

    c_offs = tl.arange(0, C_BLOCK)
    c_mask = c_offs < C

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
            base = spatial_idx * stride_spatial
            addrs = base + c_offs * stride_c

            ws_val = tl.load(where_self_ptr + addrs, mask=c_mask, other=0.0)
            sub_val = tl.load(sub_tensor_1_ptr + addrs, mask=c_mask, other=0.0)

            result = (ws_val - sub_val * s2_scaled - s1_scaled) * w

            tl.store(out_ptr + addrs, result, mask=c_mask)


# ---------------------------------------------------------------------------
# Oracle entry point
# ---------------------------------------------------------------------------

def oracle_fused(where_self, sub_tensor_1, rsqrt_squeezed, primals_weight, scale_factor):
    """
    Compute the two reductions and post-reduction pointwise in two Triton kernels.

    Returns: (mul_tensor_9, mul_tensor_10)
        mul_tensor_9: [N, C, H, W] channels-last
        mul_tensor_10: [C] = sum2 * rsqrt
    """
    N, C, H, W = where_self.shape
    total_spatial = N * H * W
    C_BLOCK = next_power_of_2(C)

    # Determine strides - support both contiguous and channels-last
    stride_c = where_self.stride(1)
    stride_h = where_self.stride(2)
    stride_w = where_self.stride(3)
    # Check linearizability: stride(2) == W * stride(3)
    assert stride_h == W * stride_w, f"Cannot linearize: stride_h={stride_h}, W*stride_w={W*stride_w}"
    stride_spatial = stride_w  # distance between consecutive spatial positions

    # --- Phase 1: Dual reduction ---
    sum1 = torch.zeros(C, device=where_self.device, dtype=torch.float32)
    sum2 = torch.zeros(C, device=where_self.device, dtype=torch.float32)

    BLOCK_SPATIAL = 64  # Smaller block since C=192 is larger
    num_programs = (total_spatial + BLOCK_SPATIAL - 1) // BLOCK_SPATIAL

    _dual_reduce_channels_last_kernel[(num_programs,)](
        where_self, sub_tensor_1,
        sum1, sum2,
        total_spatial,
        C=C,
        C_BLOCK=C_BLOCK,
        stride_spatial=stride_spatial,
        stride_c=stride_c,
        BLOCK_SPATIAL=BLOCK_SPATIAL,
    )

    # --- Phase 2: Post-reduction pointwise ---
    weight = rsqrt_squeezed * primals_weight
    out = torch.empty_like(where_self)

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
        stride_spatial=stride_spatial,
        stride_c=stride_c,
        BLOCK_SPATIAL=PW_BLOCK_SPATIAL,
    )

    # mul_tensor_10 = sum2 * rsqrt
    mul_tensor_10 = sum2 * rsqrt_squeezed

    return out, mul_tensor_10


# ---------------------------------------------------------------------------
# Prepare oracle inputs from full repro inputs
# ---------------------------------------------------------------------------

def prepare_oracle_inputs(*inputs):
    """
    Run the pre-reduction ops (avg_pool_bwd, scatter_add, BN fwd, relu mask)
    to produce the inputs needed by the oracle kernel.

    Returns: (where_self, sub_tensor_1, rsqrt_squeezed, primals_weight, scale_factor)
    """
    (getitem_246, arg253_1, getitem_255, getitem_261, getitem_264,
     arg254_1, arg250_1, arg251_1, arg252_1, arg11_1, arg12_1, full_1,
     _shape_param_0, _shape_param_1, _shape_param_2) = inputs

    import torch._inductor.inductor_prims  # noqa: F401

    # avg_pool2d_backward
    avg_pool2d_backward = torch.ops.aten.avg_pool2d_backward.default(
        getitem_246, arg253_1, [3, 3], [1, 1], [1, 1], False, True, None
    )

    # Additions
    add_tensor = avg_pool2d_backward + getitem_255
    add_tensor_1 = add_tensor + getitem_261
    add_tensor_2 = add_tensor_1 + getitem_264

    # scatter_add (max pool backward)
    full_default = torch.zeros([24576, 5041], dtype=torch.float32, device=getitem_246.device)
    clone_default = add_tensor_2.clone(memory_format=torch.contiguous_format)
    view_default = clone_default.reshape(_shape_param_0)
    _indices = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
        arg254_1, [3, 3], [71, 71], [2, 2], [0, 0], [1, 1]
    )
    clone_default_1 = _indices.clone(memory_format=torch.contiguous_format)
    view_default_1 = clone_default_1.reshape(_shape_param_1)
    scatter_add_default = full_default.scatter_add(1, view_default_1, view_default)
    view_default_2 = scatter_add_default.reshape(_shape_param_2)
    clone_default_2 = view_default_2.clone(memory_format=torch.channels_last)

    # BN forward (to compute relu mask)
    sub_tensor = arg250_1 - arg251_1
    mul_tensor = sub_tensor * arg252_1
    unsqueeze_1 = arg11_1.unsqueeze(-1).unsqueeze(-1)
    mul_tensor_1 = mul_tensor * unsqueeze_1
    unsqueeze_3 = arg12_1.unsqueeze(-1).unsqueeze(-1)
    add_tensor_3 = mul_tensor_1 + unsqueeze_3

    # ReLU backward mask
    relu_default = torch.relu(add_tensor_3)
    le_scalar = relu_default <= 0
    where_self = torch.where(le_scalar, full_1, clone_default_2)
    # Ensure channels-last layout for coalesced access
    where_self = where_self.to(memory_format=torch.channels_last)

    # sub_tensor_1 = conv - mean
    squeeze_dims = arg251_1.squeeze(0).squeeze(-1).squeeze(-1)  # [192]
    mean_expanded = squeeze_dims.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    sub_tensor_1 = arg250_1 - mean_expanded
    # Make channels-last to match where_self
    sub_tensor_1 = sub_tensor_1.to(memory_format=torch.channels_last)

    # rsqrt squeezed
    rsqrt_squeezed = arg252_1.squeeze(0).squeeze(-1).squeeze(-1)  # [192]

    scale_factor = 1.5497917079944455e-06  # 1/(128*71*71) = 1/645248

    return where_self, sub_tensor_1, rsqrt_squeezed, arg11_1, scale_factor


# ---------------------------------------------------------------------------
# Reference: compute using PyTorch ops (matches repro.py exactly)
# ---------------------------------------------------------------------------

def reference_pytorch(*inputs):
    """Direct PyTorch implementation matching repro.py."""
    (getitem_246, arg253_1, getitem_255, getitem_261, getitem_264,
     arg254_1, arg250_1, arg251_1, arg252_1, arg11_1, arg12_1, full_1,
     _shape_param_0, _shape_param_1, _shape_param_2) = inputs

    import torch._inductor.inductor_prims  # noqa: F401

    avg_pool2d_backward = torch.ops.aten.avg_pool2d_backward.default(
        getitem_246, arg253_1, [3, 3], [1, 1], [1, 1], False, True, None
    )
    add_tensor = avg_pool2d_backward + getitem_255
    add_tensor_1 = add_tensor + getitem_261
    add_tensor_2 = add_tensor_1 + getitem_264

    full_default = torch.zeros([24576, 5041], dtype=torch.float32, device=getitem_246.device)
    clone_default = add_tensor_2.clone(memory_format=torch.contiguous_format)
    view_default = clone_default.reshape(_shape_param_0)
    _indices = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
        arg254_1, [3, 3], [71, 71], [2, 2], [0, 0], [1, 1]
    )
    clone_default_1 = _indices.clone(memory_format=torch.contiguous_format)
    view_default_1 = clone_default_1.reshape(_shape_param_1)
    scatter_add_default = full_default.scatter_add(1, view_default_1, view_default)
    view_default_2 = scatter_add_default.reshape(_shape_param_2)
    clone_default_2 = view_default_2.clone(memory_format=torch.channels_last)

    sub_tensor = arg250_1 - arg251_1
    mul_tensor = sub_tensor * arg252_1
    unsqueeze_1 = arg11_1.unsqueeze(-1).unsqueeze(-1)
    mul_tensor_1 = mul_tensor * unsqueeze_1
    unsqueeze_3 = arg12_1.unsqueeze(-1).unsqueeze(-1)
    add_tensor_3 = mul_tensor_1 + unsqueeze_3
    relu_default = torch.relu(add_tensor_3)
    le_scalar = relu_default <= 0
    where_self = torch.where(le_scalar, full_1, clone_default_2)

    squeeze_dims = arg251_1.squeeze(0).squeeze(-1).squeeze(-1)
    unsqueeze_6 = squeeze_dims.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    sum_dim_int_list = where_self.sum([0, 2, 3])
    sub_tensor_1 = arg250_1 - unsqueeze_6
    mul_tensor_2 = where_self * sub_tensor_1
    sum_dim_int_list_1 = mul_tensor_2.sum([0, 2, 3])

    scale_factor = 1.5497917079944455e-06
    mul_tensor_3 = sum_dim_int_list * scale_factor
    unsqueeze_9 = mul_tensor_3.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    mul_tensor_4 = sum_dim_int_list_1 * scale_factor
    squeeze_dims_1 = arg252_1.squeeze(0).squeeze(-1).squeeze(-1)
    mul_tensor_5 = squeeze_dims_1 * squeeze_dims_1
    mul_tensor_6 = mul_tensor_4 * mul_tensor_5
    unsqueeze_12 = mul_tensor_6.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    mul_tensor_7 = squeeze_dims_1 * arg11_1
    unsqueeze_15 = mul_tensor_7.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    mul_tensor_8 = sub_tensor_1 * unsqueeze_12
    sub_tensor_2 = where_self - mul_tensor_8
    sub_tensor_3 = sub_tensor_2 - unsqueeze_9
    mul_tensor_9 = sub_tensor_3 * unsqueeze_15

    mul_tensor_10 = sum_dim_int_list_1 * squeeze_dims_1

    return mul_tensor_9, mul_tensor_10


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_repro_module():
    """Load the repro module."""
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
        print(f"  Output {i}: max_abs_diff={max_diff:.6e}, max_rel_diff={rel_diff:.6e}, ok={ok}")

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

    with torch.no_grad():
        oracle_inputs = prepare_oracle_inputs(*inputs)

    # Warm up oracle
    with torch.no_grad():
        _ = oracle_fused(*oracle_inputs)
    torch.cuda.synchronize()

    # Benchmark oracle
    oracle_ms = triton.testing.do_bench(
        lambda: oracle_fused(*oracle_inputs),
        warmup=100,
        rep=200,
    )

    # Benchmark torch.compile on the full repro
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

    print(f"\n{'='*60}")
    print(f"Benchmark results for sum_sum_a9f6c9204966")
    print(f"{'='*60}")
    print(f"  Oracle (reduction+pointwise):       {oracle_ms*1000:.1f} us")
    print(f"  torch.compile (full repro):         {compile_ms*1000:.1f} us")
    print(f"  Speedup (oracle vs compile):        {compile_ms/oracle_ms:.2f}x")
    print(f"{'='*60}")
    print(f"\nNote: Oracle measures the fused dual-reduction + post-reduction")
    print(f"pointwise. The scatter_add and avg_pool2d_backward are pre-computed.")


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
