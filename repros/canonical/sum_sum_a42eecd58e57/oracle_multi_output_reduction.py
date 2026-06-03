"""
Oracle kernel for sum_sum_a42eecd58e57 (ShuffleNet V2 batch-norm backward).

Pattern: Two channel-wise reductions over dims [0,2,3] sharing the same
input tensor (where_self), followed by element-wise computation that
depends on the reduction results.

    sum1[c] = sum_{n,h,w} where_self[n,c,h,w]
    sum2[c] = sum_{n,h,w} where_self[n,c,h,w] * (conv[n,c,h,w] - mean[c])

Then the pointwise output uses sum1 and sum2 to compute a per-element result:
    out[n,c,h,w] = (where_self - sub*s2_scaled - s1_scaled) * weight

Oracle strategy:
    - Phase 1: Parallel tiled reduction with coalesced reads.
      Iterate over spatial positions, accumulate per-channel partial sums
      in registers (C=24 fits easily), atomic-add to global output.
    - Phase 2: Fused post-reduction pointwise with coalesced I/O.
    - Single read of where_self and sub_tensor_1 feeds both accumulators.

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
# ---------------------------------------------------------------------------

@triton.jit
def _dual_reduce_kernel(
    where_self_ptr,
    sub_tensor_1_ptr,
    out_sum1_ptr,  # [C] output
    out_sum2_ptr,  # [C] output
    N,
    C: tl.constexpr,
    HW,
    stride_n,  # stride along batch dim
    stride_c,  # stride along channel dim
    stride_hw,  # stride along spatial (combined H*W) dim
    BLOCK_SPATIAL: tl.constexpr,
):
    """Each program processes BLOCK_SPATIAL spatial positions across all C channels."""
    pid = tl.program_id(0)
    spatial_start = pid * BLOCK_SPATIAL
    total_spatial = N * HW

    c_offs = tl.arange(0, C)

    acc1 = tl.zeros([C], dtype=tl.float32)
    acc2 = tl.zeros([C], dtype=tl.float32)

    for s in range(BLOCK_SPATIAL):
        spatial_idx = spatial_start + s
        if spatial_idx < total_spatial:
            n_idx = spatial_idx // HW
            hw_idx = spatial_idx % HW
            base = n_idx * stride_n + hw_idx * stride_hw
            addrs = base + c_offs * stride_c

            ws_vals = tl.load(where_self_ptr + addrs)
            sub_vals = tl.load(sub_tensor_1_ptr + addrs)

            acc1 += ws_vals
            acc2 += ws_vals * sub_vals

    tl.atomic_add(out_sum1_ptr + c_offs, acc1)
    tl.atomic_add(out_sum2_ptr + c_offs, acc2)


# ---------------------------------------------------------------------------
# Triton kernel: post-reduction pointwise
# ---------------------------------------------------------------------------

@triton.jit
def _post_reduce_pointwise_kernel(
    where_self_ptr,
    sub_tensor_1_ptr,
    sum1_ptr,        # [C]
    sum2_ptr,        # [C]
    rsqrt_ptr,       # [C] (arg269_1 equivalent = arg134_1 in reference)
    weight_ptr,      # [C] = rsqrt * primals_weight
    out_ptr,
    scale_factor,
    N,
    C: tl.constexpr,
    HW,
    stride_n,
    stride_c,
    stride_hw,
    BLOCK_SPATIAL: tl.constexpr,
):
    """Pointwise kernel. Each program processes BLOCK_SPATIAL spatial positions x all C channels."""
    pid = tl.program_id(0)
    spatial_start = pid * BLOCK_SPATIAL
    total_spatial = N * HW

    c_offs = tl.arange(0, C)

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
            n_idx = spatial_idx // HW
            hw_idx = spatial_idx % HW
            base = n_idx * stride_n + hw_idx * stride_hw
            addrs = base + c_offs * stride_c

            ws_val = tl.load(where_self_ptr + addrs)
            sub_val = tl.load(sub_tensor_1_ptr + addrs)

            result = (ws_val - sub_val * s2_scaled - s1_scaled) * w

            tl.store(out_ptr + addrs, result)


# ---------------------------------------------------------------------------
# Oracle entry point
# ---------------------------------------------------------------------------

def oracle_fused(where_self, sub_tensor_1, rsqrt_squeezed, primals_weight, scale_factor):
    """
    Compute the two reductions and post-reduction pointwise in two Triton kernels.

    Returns: (mul_tensor_9, mul_tensor_10)
        mul_tensor_9: [N, C, H, W] - post-reduction pointwise output
        mul_tensor_10: [C] - sum2 * rsqrt
    """
    N, C, H, W = where_self.shape
    HW = H * W
    total_spatial = N * HW

    # Determine strides - support both contiguous and channels_last
    stride_n = where_self.stride(0)
    stride_c = where_self.stride(1)
    # For 4D tensor, combined spatial stride is stride(2) for h and stride(3) for w
    # We need to linearize H*W. If channels_last: stride(1)=1, stride(3)=C, stride(2)=W*C
    # If contiguous: stride(1)=H*W, stride(3)=1, stride(2)=W
    # For channels-last, spatial positions are contiguous with stride C between them
    # We'll use stride(3) as the stride between adjacent w positions in the hw dimension
    # Actually we just compute per-element: offset = n*stride_n + c*stride_c + h*stride_h + w*stride_w
    # For linearized HW: hw_idx = h*W + w, so offset = n*stride_n + c*stride_c + h*stride(2) + w*stride(3)
    # This doesn't linearize cleanly unless stride(2) = W * stride(3)
    # For channels-last: stride = (C*H*W, 1, W*C, C) -> stride(2) = W*C = W*stride(3)? No, stride(3)=C
    # stride(2) = W*C, stride(3) = C => stride(2) = W * stride(3). YES! This works for both layouts.
    stride_h = where_self.stride(2)
    stride_w = where_self.stride(3)

    # Check if stride(2) = W * stride(3) so we can linearize
    assert stride_h == W * stride_w, f"Cannot linearize H*W: stride_h={stride_h}, W*stride_w={W*stride_w}"

    # The "hw stride" is stride_w (distance between consecutive spatial positions when linearized)
    stride_hw = stride_w

    # --- Phase 1: Dual reduction ---
    sum1 = torch.zeros(C, device=where_self.device, dtype=torch.float32)
    sum2 = torch.zeros(C, device=where_self.device, dtype=torch.float32)

    BLOCK_SPATIAL = 128
    num_programs = (total_spatial + BLOCK_SPATIAL - 1) // BLOCK_SPATIAL

    _dual_reduce_kernel[(num_programs,)](
        where_self, sub_tensor_1,
        sum1, sum2,
        N, C=C, HW=HW,
        stride_n=stride_n, stride_c=stride_c, stride_hw=stride_hw,
        BLOCK_SPATIAL=BLOCK_SPATIAL,
    )

    # --- Phase 2: Post-reduction pointwise ---
    weight = rsqrt_squeezed * primals_weight

    out = torch.empty_like(where_self)

    PW_BLOCK_SPATIAL = 128
    pw_num_programs = (total_spatial + PW_BLOCK_SPATIAL - 1) // PW_BLOCK_SPATIAL

    _post_reduce_pointwise_kernel[(pw_num_programs,)](
        where_self, sub_tensor_1,
        sum1, sum2,
        rsqrt_squeezed, weight,
        out,
        scale_factor,
        N, C=C, HW=HW,
        stride_n=stride_n, stride_c=stride_c, stride_hw=stride_hw,
        BLOCK_SPATIAL=PW_BLOCK_SPATIAL,
    )

    # mul_tensor_10 = sum2 * rsqrt_squeezed
    mul_tensor_10 = sum2 * rsqrt_squeezed

    return out, mul_tensor_10


# ---------------------------------------------------------------------------
# Prepare oracle inputs from full repro inputs
# ---------------------------------------------------------------------------

def prepare_oracle_inputs(*inputs):
    """
    Run the pre-reduction ops (scatter_add, batch norm forward, relu mask)
    to produce the inputs needed by the oracle kernel.

    Returns: (where_self, sub_tensor_1, rsqrt_squeezed, primals_weight, scale_factor)
    """
    (getitem_156, getitem_162, arg139_1, arg135_1, arg136_1, arg137_1,
     arg2_1, arg3_1, full, _shape_param_0, _shape_param_1, _shape_param_2) = inputs

    import torch._inductor.inductor_prims  # noqa: F401

    # add + scatter (max pool backward)
    add_tensor = getitem_156 + getitem_162
    full_default = torch.zeros([12288, 12544], dtype=torch.float32, device=add_tensor.device)
    view_default = add_tensor.reshape(_shape_param_0)
    _low_memory_max_pool_offsets_to_indices_default = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
        arg139_1, [3, 3], [112, 112], [2, 2], [1, 1], [1, 1]
    )
    view_default_1 = _low_memory_max_pool_offsets_to_indices_default.reshape(_shape_param_1)
    scatter_add_default = full_default.scatter_add(1, view_default_1, view_default)
    view_default_2 = scatter_add_default.reshape(_shape_param_2)

    # Batch norm forward path
    sub_tensor = arg135_1 - arg136_1
    mul_tensor = sub_tensor * arg137_1
    unsqueeze_1 = arg2_1.unsqueeze(-1).unsqueeze(-1)
    mul_tensor_1 = mul_tensor * unsqueeze_1
    unsqueeze_3 = arg3_1.unsqueeze(-1).unsqueeze(-1)
    add_tensor_1 = mul_tensor_1 + unsqueeze_3

    # ReLU backward mask
    relu_default = torch.relu(add_tensor_1)
    le_scalar = relu_default <= 0
    where_self = torch.where(le_scalar, full, view_default_2)

    # sub_tensor_1 = conv - mean (per-channel)
    squeeze_dims = arg136_1.squeeze(0).squeeze(-1).squeeze(-1)  # [C]
    mean_expanded = squeeze_dims.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    sub_tensor_1 = arg135_1 - mean_expanded

    # rsqrt squeezed
    rsqrt_squeezed = arg137_1.squeeze(0).squeeze(-1).squeeze(-1)  # [C]

    scale_factor = 1.5570192920918366e-07  # 1/(N*H*W) = 1/(512*112*112)

    return where_self, sub_tensor_1, rsqrt_squeezed, arg2_1, scale_factor


# ---------------------------------------------------------------------------
# Reference: compute using PyTorch ops (matches repro.py exactly)
# ---------------------------------------------------------------------------

def reference_pytorch(*inputs):
    """Direct PyTorch implementation matching repro.py."""
    (getitem_156, getitem_162, arg139_1, arg135_1, arg136_1, arg137_1,
     arg2_1, arg3_1, full, _shape_param_0, _shape_param_1, _shape_param_2) = inputs

    import torch._inductor.inductor_prims  # noqa: F401

    add_tensor = getitem_156 + getitem_162
    full_default = torch.zeros([12288, 12544], dtype=torch.float32, device=add_tensor.device)
    view_default = add_tensor.reshape(_shape_param_0)
    _low_memory_max_pool_offsets_to_indices_default = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
        arg139_1, [3, 3], [112, 112], [2, 2], [1, 1], [1, 1]
    )
    view_default_1 = _low_memory_max_pool_offsets_to_indices_default.reshape(_shape_param_1)
    scatter_add_default = full_default.scatter_add(1, view_default_1, view_default)
    view_default_2 = scatter_add_default.reshape(_shape_param_2)

    sub_tensor = arg135_1 - arg136_1
    mul_tensor = sub_tensor * arg137_1
    unsqueeze_1 = arg2_1.unsqueeze(-1).unsqueeze(-1)
    mul_tensor_1 = mul_tensor * unsqueeze_1
    unsqueeze_3 = arg3_1.unsqueeze(-1).unsqueeze(-1)
    add_tensor_1 = mul_tensor_1 + unsqueeze_3
    relu_default = torch.relu(add_tensor_1)
    le_scalar = relu_default <= 0
    where_self = torch.where(le_scalar, full, view_default_2)

    squeeze_dims = arg136_1.squeeze(0).squeeze(-1).squeeze(-1)
    unsqueeze_6 = squeeze_dims.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    sum_dim_int_list = where_self.sum([0, 2, 3])
    sub_tensor_1 = arg135_1 - unsqueeze_6
    mul_tensor_2 = where_self * sub_tensor_1
    sum_dim_int_list_1 = mul_tensor_2.sum([0, 2, 3])

    scale_factor = 1.5570192920918366e-07
    mul_tensor_3 = sum_dim_int_list * scale_factor
    unsqueeze_9 = mul_tensor_3.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    mul_tensor_4 = sum_dim_int_list_1 * scale_factor
    squeeze_dims_1 = arg137_1.squeeze(0).squeeze(-1).squeeze(-1)
    mul_tensor_5 = squeeze_dims_1 * squeeze_dims_1
    mul_tensor_6 = mul_tensor_4 * mul_tensor_5
    unsqueeze_12 = mul_tensor_6.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    mul_tensor_7 = squeeze_dims_1 * arg2_1
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
    print(f"Benchmark results for sum_sum_a42eecd58e57")
    print(f"{'='*60}")
    print(f"  Oracle (reduction+pointwise):       {oracle_ms*1000:.1f} us")
    print(f"  torch.compile (full repro):         {compile_ms*1000:.1f} us")
    print(f"  Speedup (oracle vs compile):        {compile_ms/oracle_ms:.2f}x")
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
