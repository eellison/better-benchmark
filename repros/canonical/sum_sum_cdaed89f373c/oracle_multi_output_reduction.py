"""
Oracle kernel for repro sum_sum_cdaed89f373c (ResNet18 training backward).

Repro pattern:
    Batch-norm backward + ReLU backward producing two channel-wise sum
    reductions over a shared intermediate ``where_self`` of shape
    [512, 64, 112, 112]:
      - sum1 = where_self.sum(dim=[0, 2, 3])                    -> [64]
      - sum2 = (where_self * sub_tensor_1).sum(dim=[0, 2, 3])   -> [64]

    These two reductions read the same large tensor and reduce N*H*W elements
    per channel.  The oracle fuses them into a single-pass Triton kernel with
    two accumulators per channel, then applies the downstream pointwise epilogue
    that produces both final outputs in one additional pass.

    The upstream computation (scatter_add for max_pool backward, batchnorm
    intermediates, relu backward) is done in PyTorch and is NOT the target of
    this oracle -- those are separate kernel launches.  The oracle targets
    specifically the multi-output reduction + epilogue fusion opportunity.

    Final outputs:
      - mul_tensor_9:  shape [512, 64, 112, 112] (batchnorm input gradient)
      - mul_tensor_10: shape [64] (weight gradient contribution)
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
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)



REPRO_ID = "sum_sum_cdaed89f373c"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"


def _partial_reduction_kernel(
    where_self_ptr,    # [N*C*HW] contiguous
    sub_tensor_1_ptr,  # [N*C*HW] contiguous
    partial_sum1_ptr,  # [C, n_tiles]
    partial_sum2_ptr,  # [C, n_tiles]
    N: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    N_TILES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Partial reduction: each program handles one (channel, tile) pair."""
    c = tl.program_id(0)
    tile_id = tl.program_id(1)

    total = N * HW  # elements per channel
    # Each tile handles a contiguous chunk of the flattened N*HW dimension
    tile_size = (total + N_TILES - 1) // N_TILES
    tile_start = tile_id * tile_size
    tile_end = tl.minimum(tile_start + tile_size, total)

    acc1 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    base = c * HW  # channel offset within each N-slice

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

    sum1 = tl.sum(acc1, axis=0)
    sum2 = tl.sum(acc2, axis=0)

    # Store partial results
    out_idx = c * N_TILES + tile_id
    tl.store(partial_sum1_ptr + out_idx, sum1)
    tl.store(partial_sum2_ptr + out_idx, sum2)


@triton.jit
def _finalize_reduction_kernel(
    partial_sum1_ptr,  # [C, N_TILES]
    partial_sum2_ptr,  # [C, N_TILES]
    out_sum1_ptr,      # [C]
    out_sum2_ptr,      # [C]
    N_TILES: tl.constexpr,
):
    """Sum partial results across tiles for each channel."""
    c = tl.program_id(0)

    offsets = tl.arange(0, N_TILES)
    base = c * N_TILES

    p1 = tl.load(partial_sum1_ptr + base + offsets)
    p2 = tl.load(partial_sum2_ptr + base + offsets)

    tl.store(out_sum1_ptr + c, tl.sum(p1, axis=0))
    tl.store(out_sum2_ptr + c, tl.sum(p2, axis=0))


@triton.jit
def _fused_epilogue_kernel(
    where_self_ptr,     # [N*C*HW]
    sub_tensor_1_ptr,   # [N*C*HW]
    factor12_ptr,       # [C]
    factor9_ptr,        # [C]
    mul_factor_ptr,     # [C]
    out_ptr,            # [N*C*HW]
    C: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Epilogue: compute mul_tensor_9 element-wise.

    mul_tensor_9[n,c,h,w] = (where_self[n,c,h,w]
                              - sub_tensor_1[n,c,h,w] * factor12[c]
                              - factor9[c]) * mul_factor[c]
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Determine channel index for each element
    # Layout: [N, C, HW] contiguous -> c = (offset // HW) % C
    c_idx = (offsets // HW) % C

    # Load per-channel values
    f12 = tl.load(factor12_ptr + c_idx)
    f9 = tl.load(factor9_ptr + c_idx)
    mf = tl.load(mul_factor_ptr + c_idx)

    # Load element-wise values
    w_val = tl.load(where_self_ptr + offsets)
    s_val = tl.load(sub_tensor_1_ptr + offsets)

    # Compute epilogue
    result = (w_val - s_val * f12 - f9) * mf

    tl.store(out_ptr + offsets, result)


def compute_upstream(where_14, getitem_54, arg47_1, arg43_1, arg44_1, arg45_1,
                     arg2_1, arg3_1, full, _shape_param_0, _shape_param_1, _shape_param_2):
    """Compute the upstream portion (scatter_add, batchnorm, relu backward).

    Returns where_self and sub_tensor_1 which are inputs to the fused reduction.
    """
    device = where_14.device
    N, C, H, W = 512, 64, 112, 112

    add_tensor = where_14 + getitem_54

    # scatter_add for max_pool backward
    full_default = torch.zeros(32768, 12544, dtype=torch.float32, device=device)
    view_default = add_tensor.view(32768, 3136)
    offsets_i64 = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
        arg47_1, [3, 3], [112, 112], [2, 2], [1, 1], [1, 1]
    )
    view_default_1 = offsets_i64.view(32768, 3136)
    scatter_add_default = full_default.scatter_add(1, view_default_1, view_default)
    view_default_2 = scatter_add_default.view(N, C, H, W)

    # Batchnorm backward intermediate
    sub_tensor = arg43_1 - arg44_1
    mul_tensor = sub_tensor * arg45_1
    weight_4d = arg2_1.view(1, C, 1, 1)
    mul_tensor_1 = mul_tensor * weight_4d
    bias_4d = arg3_1.view(1, C, 1, 1)
    add_tensor_1 = mul_tensor_1 + bias_4d

    # relu backward via where
    relu_default = torch.relu(add_tensor_1)
    le_scalar = relu_default <= 0
    where_self = torch.where(le_scalar, full, view_default_2)

    # sub_tensor_1 = arg43_1 - arg44_1 (broadcast subtraction)
    sub_tensor_1 = sub_tensor  # [N, C, H, W] - already computed above

    return where_self, sub_tensor_1


def fused_reduction_and_epilogue(where_self, sub_tensor_1, arg45_1, arg2_1):
    """The fused oracle: single-pass two-accumulator reduction + epilogue.

    This is the kernel we're optimizing - it replaces what Inductor generates
    as separate reduction kernels.
    """
    device = where_self.device
    N, C, H, W = 512, 64, 112, 112
    HW = H * W  # 12544
    total_elements = N * C * HW

    # Ensure contiguous
    where_self_c = where_self.contiguous()
    sub_tensor_1_c = sub_tensor_1.contiguous()

    # Phase 1: fused two-output partial reduction with 2D grid
    # Use enough tiles to saturate the GPU (e.g., 64 channels * 128 tiles = 8192 programs)
    N_TILES = 128  # must be power of 2 for the finalize kernel
    BLOCK_SIZE = 2048

    partial_sum1 = torch.empty(C, N_TILES, dtype=torch.float32, device=device)
    partial_sum2 = torch.empty(C, N_TILES, dtype=torch.float32, device=device)

    _partial_reduction_kernel[(C, N_TILES)](
        where_self_c, sub_tensor_1_c,
        partial_sum1, partial_sum2,
        N=N, C=C, HW=HW, N_TILES=N_TILES, BLOCK_SIZE=BLOCK_SIZE,
    )

    # Finalize: sum partials
    out_sum1 = torch.empty(C, dtype=torch.float32, device=device)
    out_sum2 = torch.empty(C, dtype=torch.float32, device=device)

    _finalize_reduction_kernel[(C,)](
        partial_sum1, partial_sum2,
        out_sum1, out_sum2,
        N_TILES=N_TILES,
    )

    # Phase 2: compute per-channel factors and epilogue
    squeeze_dims_1 = arg45_1.view(C)  # invstd [64]
    scale = 1.5570192920918366e-07    # 1 / (N * H * W) = 1 / 6422528

    # mul_tensor_10 = out_sum2 * squeeze_dims_1 (output 2)
    mul_tensor_10 = out_sum2 * squeeze_dims_1

    # Per-channel factors for epilogue
    factor12 = out_sum2 * scale * squeeze_dims_1 * squeeze_dims_1  # [C]
    factor9 = out_sum1 * scale                                      # [C]
    mul_factor = squeeze_dims_1 * arg2_1                            # [C]

    # Phase 2 epilogue kernel
    out_9 = torch.empty_like(where_self_c)

    EPILOGUE_BLOCK = 1024
    n_blocks = (total_elements + EPILOGUE_BLOCK - 1) // EPILOGUE_BLOCK
    _fused_epilogue_kernel[(n_blocks,)](
        where_self_c, sub_tensor_1_c,
        factor12, factor9, mul_factor,
        out_9,
        C=C, HW=HW, BLOCK_SIZE=EPILOGUE_BLOCK,
    )

    return out_9, mul_tensor_10


def triton_oracle(where_14, getitem_54, arg47_1, arg43_1, arg44_1, arg45_1,
                  arg2_1, arg3_1, full, _shape_param_0, _shape_param_1, _shape_param_2):
    """Full oracle: upstream (PyTorch) + fused reduction+epilogue (Triton)."""
    where_self, sub_tensor_1 = compute_upstream(
        where_14, getitem_54, arg47_1, arg43_1, arg44_1, arg45_1,
        arg2_1, arg3_1, full, _shape_param_0, _shape_param_1, _shape_param_2
    )
    return fused_reduction_and_epilogue(where_self, sub_tensor_1, arg45_1, arg2_1)


def make_inputs(device: torch.device) -> tuple:
    module = _load_repro_module()
    inputs = module.make_inputs()
    moved = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
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
        print(f"  output[{i}]: shape={list(o.shape)}, max_abs_diff={max_diff:.6g}, allclose={close}")
        if not close:
            rel_err = ((o.float() - r.float()).abs() / (r.float().abs() + 1e-8)).max().item()
            print(f"           max_rel_err={rel_err:.6g}")
            all_close = False

    print(f"  OVERALL: {'PASS' if all_close else 'FAIL'}")
    return all_close


def benchmark_oracle(device: torch.device, warmup: int = 10, rep: int = 100):
    """Benchmark the fused reduction+epilogue kernel (the oracle target).

    We benchmark just the fused_reduction_and_epilogue portion since that's
    what a better Inductor template would improve.  The upstream scatter_add etc.
    are separate kernel launches that Inductor already handles.
    """
    inputs = make_inputs(device)

    with torch.no_grad():
        # Precompute upstream (not part of oracle timing)
        where_self, sub_tensor_1 = compute_upstream(*inputs)
        arg45_1 = inputs[5]  # arg45_1
        arg2_1 = inputs[6]   # arg2_1
        torch.cuda.synchronize()

        # Benchmark just the fused reduction + epilogue
        def oracle_fn():
            return fused_reduction_and_epilogue(where_self, sub_tensor_1, arg45_1, arg2_1)

        ms = triton.testing.do_bench(oracle_fn, warmup=warmup, rep=rep)
        us = ms * 1000.0
        print(f"  oracle (fused reduction+epilogue) us={us:.3f}")

        # Benchmark the full oracle for comparison
        def full_oracle_fn():
            return triton_oracle(*inputs)

        ms_full = triton.testing.do_bench(full_oracle_fn, warmup=warmup, rep=rep)
        us_full = ms_full * 1000.0
        print(f"  oracle (full including upstream) us={us_full:.3f}")

        # Benchmark compiled repro
        module = _load_repro_module()
        repro_instance = module.Repro()

        compiled = torch.compile(repro_instance)
        compiled(*inputs)
        torch.cuda.synchronize()
        ms_compiled = triton.testing.do_bench(lambda: compiled(*inputs), warmup=warmup, rep=rep)
        us_compiled = ms_compiled * 1000.0
        print(f"  compiled (torch.compile) us={us_compiled:.3f}")

        # With coordinate descent tuning
        import torch._inductor.config as inductor_config
        inductor_config.coordinate_descent_tuning = True
        compiled_cd = torch.compile(repro_instance, fullgraph=True)
        compiled_cd(*inputs)
        torch.cuda.synchronize()
        ms_cd = triton.testing.do_bench(lambda: compiled_cd(*inputs), warmup=warmup, rep=rep)
        us_cd = ms_cd * 1000.0
        inductor_config.coordinate_descent_tuning = False
        print(f"  compiled_cd (coordinate_descent_tuning) us={us_cd:.3f}")

        print(f"\n  Summary:")
        print(f"    Oracle fused reduction+epilogue: {us:.1f} us")
        print(f"    Compiled full graph:             {us_compiled:.1f} us")
        print(f"    Compiled + CD tuning:            {us_cd:.1f} us")

    return us


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run correctness check")
    parser.add_argument("--bench", action="store_true", help="Run benchmark")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    return parser.parse_args()


@oracle_impl(hardware="H100", shapes="(T([512, 64, 56, 56], f32), T([512, 64, 56, 56], f32), T([512, 64, 56, 56], i8, gen=Index(5, 4)), T([512, 64, 112, 112], f32), T([1, 64, 1, 1], f32), T([1, 64, 1, 1], f32), T([64], f32), T([64], f32), T([], f32), S([32768, 3136]), S([32768, 3136]), S([512, 64, 112, 112]))")
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
