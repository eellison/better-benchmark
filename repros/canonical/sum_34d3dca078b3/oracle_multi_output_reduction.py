"""
Oracle kernel for sum_34d3dca078b3 (VGG16 training backward).

Pattern: Max-pool backward scatter_add -> where (relu backward mask) -> sum.
    1. scatter_add: [8192, 50176].scatter_add(1, indices[8192,12544], values[8192,12544])
    2. view: [8192, 50176] -> [128, 64, 224, 224]
    3. where: mask[128, 64, 224, 224] ? scalar_0 : scatter_result
    4. sum(dim=[0,2,3]) -> [64]

The full graph has a single output: f32[64].

Oracle strategy:
    The scatter_add is an inherently expensive operation (random writes).
    However, the subsequent where + sum can be fused: instead of materializing
    the full [128, 64, 224, 224] where result and then reading it again for
    sum, we fuse where + sum into a single Triton kernel.

    Better still, we can fuse scatter_add + where + sum by iterating over the
    scatter result in-place and combining with the mask. But scatter_add
    requires the full [8192, 50176] intermediate. The real win is avoiding
    the extra read pass.

    The oracle uses a two-phase approach:
    - Phase 1: scatter_add (unavoidable, done via PyTorch)
    - Phase 2: Fused where+sum Triton kernel with 2D grid (channel, tile)
      that reads scatter_result and mask, applies where, and accumulates
      per-channel partial sums. A finalize kernel sums partials.

    This saves one full read of [128, 64, 224, 224] float32 (~3.2 GB) by
    avoiding materializing the where output.
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


REPRO_ID = "sum_34d3dca078b3"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"



@triton.jit
def _fused_where_sum_partial_kernel(
    scatter_result_ptr,  # [N, C, H, W] contiguous float32
    mask_ptr,            # [N, C, H, W] contiguous bool
    full_val,            # scalar float32 (the fill value for where)
    partial_sums_ptr,    # [C, N_TILES] output
    N: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    N_TILES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Partial reduction: each program processes one (channel, tile) pair.

    Computes: for each element, where(mask, full_val, scatter_result),
    then sums over the tile's portion of the N*HW dimension for this channel.
    """
    c = tl.program_id(0)
    tile_id = tl.program_id(1)

    total = N * HW  # elements per channel to reduce
    tile_size = (total + N_TILES - 1) // N_TILES
    tile_start = tile_id * tile_size
    tile_end = tl.minimum(tile_start + tile_size, total)

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Channel stride offset: data is [N, C, H, W] contiguous
    # element [n, c, h, w] is at offset n*C*HW + c*HW + hw
    # For fixed c, iterating over n*HW: index = (flat // HW)*C*HW + c*HW + (flat % HW)
    c_offset = c * HW

    for block_start in range(tile_start, tile_end, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask_valid = offsets < tile_end

        # Convert flat offset (within channel) to global memory index
        n = offsets // HW
        hw = offsets % HW
        idx = n * (C * HW) + c_offset + hw

        # Load scatter result and mask
        sr_val = tl.load(scatter_result_ptr + idx, mask=mask_valid, other=0.0)
        m_val = tl.load(mask_ptr + idx, mask=mask_valid, other=0)

        # Apply where: if mask is True, use full_val; else use scatter_result
        where_val = tl.where(m_val != 0, full_val, sr_val)

        acc += where_val

    partial_sum = tl.sum(acc, axis=0)

    # Store partial result
    out_idx = c * N_TILES + tile_id
    tl.store(partial_sums_ptr + out_idx, partial_sum)


@triton.jit
def _finalize_sum_kernel(
    partial_sums_ptr,  # [C, N_TILES]
    out_ptr,           # [C]
    N_TILES: tl.constexpr,
):
    """Sum partial results across tiles for each channel."""
    c = tl.program_id(0)

    offsets = tl.arange(0, N_TILES)
    base = c * N_TILES

    partials = tl.load(partial_sums_ptr + base + offsets)
    tl.store(out_ptr + c, tl.sum(partials, axis=0))


# ---------------------------------------------------------------------------
# Oracle implementation
# ---------------------------------------------------------------------------

def oracle_fused_where_sum(scatter_result, mask, full_val, N, C, H, W):
    """Fused where + sum reduction using Triton.

    Args:
        scatter_result: [N, C, H, W] contiguous float32
        mask: [N, C, H, W] contiguous bool
        full_val: scalar float
        N, C, H, W: dimensions

    Returns:
        sum_result: [C] float32
    """
    device = scatter_result.device
    HW = H * W

    # Ensure contiguous
    scatter_result_c = scatter_result.contiguous()
    mask_c = mask.contiguous()

    # Tuning: use enough tiles to saturate the GPU
    # 64 channels * 128 tiles = 8192 programs
    N_TILES = 128
    BLOCK_SIZE = 2048

    partial_sums = torch.empty(C, N_TILES, dtype=torch.float32, device=device)

    _fused_where_sum_partial_kernel[(C, N_TILES)](
        scatter_result_c, mask_c,
        full_val,
        partial_sums,
        N=N, C=C, HW=HW, N_TILES=N_TILES, BLOCK_SIZE=BLOCK_SIZE,
    )

    # Finalize: sum partials across tiles
    out = torch.empty(C, dtype=torch.float32, device=device)

    _finalize_sum_kernel[(C,)](
        partial_sums, out,
        N_TILES=N_TILES,
    )

    return out


def oracle_full(getitem_30, arg19_1, arg46_1, full, _shape_param_0, _shape_param_1, _shape_param_2):
    """Full oracle: scatter_add (PyTorch) + fused where+sum (Triton).

    This matches the repro's forward() signature exactly.
    """
    import torch._inductor.inductor_prims  # noqa: F401

    device = getitem_30.device
    N, C, H, W = 128, 64, 224, 224

    # --- Phase 1: scatter_add (unchanged from repro) ---
    full_default = torch.zeros([8192, 50176], dtype=torch.float32, device=device)
    view_default = getitem_30.view(_shape_param_0)

    _low_memory_max_pool_offsets_to_indices_default = (
        torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
            arg19_1, [2, 2], [224, 224], [2, 2], [0, 0], [1, 1]
        )
    )
    view_default_1 = _low_memory_max_pool_offsets_to_indices_default.view(_shape_param_1)
    scatter_add_default = full_default.scatter_add(1, view_default_1, view_default)
    scatter_result = scatter_add_default.view(N, C, H, W)

    # --- Phase 2: fused where + sum (Triton oracle) ---
    full_val = full.item()
    result = oracle_fused_where_sum(scatter_result, arg46_1, full_val, N, C, H, W)

    return result


# ---------------------------------------------------------------------------
# Correctness and benchmarking
# ---------------------------------------------------------------------------

def make_inputs(device: torch.device = None):
    """Generate inputs from the repro module."""
    module = _load_repro_module()
    inputs = module.make_inputs()
    if device is not None:
        moved = []
        for value in inputs:
            if isinstance(value, torch.Tensor):
                moved.append(value.to(device=device))
            else:
                moved.append(value)
        return tuple(moved)
    return tuple(inputs)


def check_correctness(device: torch.device, rtol: float = 1e-4, atol: float = 1e-3):
    """Compare oracle output against eager reference."""
    inputs = make_inputs(device)
    module = _load_repro_module()

    with torch.no_grad():
        oracle_out = oracle_full(*inputs)
        ref_out = module.Repro()(*inputs)

    max_diff = (oracle_out.float() - ref_out.float()).abs().max().item()
    close = torch.allclose(oracle_out.float(), ref_out.float(), rtol=rtol, atol=atol)
    print(f"  output: shape={list(oracle_out.shape)}, max_abs_diff={max_diff:.6g}, allclose={close}")

    if not close:
        rel_err = ((oracle_out.float() - ref_out.float()).abs() / (ref_out.float().abs() + 1e-8)).max().item()
        print(f"         max_rel_err={rel_err:.6g}")

    print(f"  OVERALL: {'PASS' if close else 'FAIL'}")
    return close


def benchmark_oracle(device: torch.device, warmup: int = 25, rep: int = 100):
    """Benchmark oracle vs torch.compile + coordinate descent."""
    import torch._inductor.inductor_prims  # noqa: F401
    import torch._inductor.config as inductor_config

    inputs = make_inputs(device)
    module = _load_repro_module()

    with torch.no_grad():
        # --- Benchmark oracle (full pipeline) ---
        # Warmup
        for _ in range(3):
            oracle_full(*inputs)
        torch.cuda.synchronize()

        oracle_ms = triton.testing.do_bench(
            lambda: oracle_full(*inputs),
            warmup=warmup,
            rep=rep,
        )
        oracle_us = oracle_ms * 1000.0

        # --- Benchmark: fused where+sum only (the Triton part) ---
        # Pre-compute scatter_add
        full_default = torch.zeros([8192, 50176], dtype=torch.float32, device=device)
        view_default = inputs[0].view(inputs[4])  # getitem_30.view(_shape_param_0)
        offsets_i64 = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
            inputs[1], [2, 2], [224, 224], [2, 2], [0, 0], [1, 1]
        )
        view_default_1 = offsets_i64.view(inputs[5])
        scatter_add_default = full_default.scatter_add(1, view_default_1, view_default)
        scatter_result = scatter_add_default.view(128, 64, 224, 224)
        mask = inputs[2]
        full_val = inputs[3].item()
        torch.cuda.synchronize()

        fused_ms = triton.testing.do_bench(
            lambda: oracle_fused_where_sum(scatter_result, mask, full_val, 128, 64, 224, 224),
            warmup=warmup,
            rep=rep,
        )
        fused_us = fused_ms * 1000.0

        # --- Benchmark torch.compile ---
        model = module.Repro()
        torch._dynamo.reset()
        compiled = torch.compile(model)
        compiled(*inputs)
        torch.cuda.synchronize()

        compile_ms = triton.testing.do_bench(
            lambda: compiled(*inputs),
            warmup=warmup,
            rep=rep,
        )
        compile_us = compile_ms * 1000.0

        # --- Benchmark torch.compile + coordinate descent ---
        torch._dynamo.reset()
        inductor_config.coordinate_descent_tuning = True
        compiled_cd = torch.compile(model)
        compiled_cd(*inputs)
        torch.cuda.synchronize()

        compile_cd_ms = triton.testing.do_bench(
            lambda: compiled_cd(*inputs),
            warmup=warmup,
            rep=rep,
        )
        compile_cd_us = compile_cd_ms * 1000.0
        inductor_config.coordinate_descent_tuning = False

        # --- Report ---
        print(f"\n{'='*60}")
        print(f"Benchmark results for {REPRO_ID}")
        print(f"{'='*60}")
        print(f"  Oracle (full pipeline):               {oracle_us:.1f} us")
        print(f"  Oracle (fused where+sum only):        {fused_us:.1f} us")
        print(f"  torch.compile (full repro):           {compile_us:.1f} us")
        print(f"  torch.compile + coord_descent:        {compile_cd_us:.1f} us")
        print(f"  Speedup (oracle vs compile):          {compile_us/oracle_us:.2f}x")
        print(f"  Speedup (oracle vs coord_descent):    {compile_cd_us/oracle_us:.2f}x")
        print(f"{'='*60}")
        print(f"\nNote: The scatter_add is included in both oracle and compile")
        print(f"timings. The Triton kernel fuses where+sum to avoid materializing")
        print(f"the intermediate [128,64,224,224] where result.")

    return oracle_us, compile_us, compile_cd_us


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


@oracle_impl(hardware="H100", shapes="(T([128, 64, 112, 112], f32), T([128, 64, 112, 112], i8, gen=Index(4)), T([128, 64, 224, 224], b8), T([], f32), S([8192, 12544]), S([8192, 12544]), S([128, 64, 224, 224]))")
def oracle_forward(inputs):
    return oracle_full(*inputs)


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
