"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete LayerNorm-backward return tuple by row-tiling the `[1152000, 512]` producer, preserving the fp32 `dy * gamma` row reduction inside the producer and using Triton to finalize fp32 `[512]` column partials, whereas Inductor materializes a row-invariant `[1152000, 1]` `sum(dy * gamma)` buffer before its mix-order reduction; Inductor cannot do this today because its algebraic simplifier does not prove that a reduction over an expanded scalar times a channel vector is independent of the row dimension once it feeds a later multi-output reduction; the fix is ALGEBRAIC_ELIMINATION: canonicalize broadcast-only row reductions to scalar/channel summaries and pass them directly into the fused LayerNorm-backward producer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None

from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 1_152_000
CHANNELS = 512
ROW_TILE = 64
FINAL_BLOCK_CHANNELS = 16
PARTIAL_REDUCE_BLOCK_TILES = 256
PARTIAL_REDUCE_BLOCK_CHANNELS = 16
FINAL_STAGE_BLOCK_TILES = 128


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(shape_param) -> tuple[int, ...]:
    return tuple(int(dim) for dim in shape_param)


if triton is not None:
    from torch._C import _cuda_getCurrentRawStream as get_raw_stream
    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.hints import DeviceProperties, ReductionHint

    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.persistent_reduction(
        size_hints={"x": 2097152, "r0_": 512},
        reduction_hint=ReductionHint.INNER,
        filename=__file__,
        triton_meta={
            "signature": {
                "dy_scalar_ptr": "*bf16",
                "gamma_ptr": "*fp32",
                "x_ptr": "*bf16",
                "mean_ptr": "*fp32",
                "invstd_ptr": "*fp32",
                "out_grad_ptr": "*bf16",
                "partial_sum_ptr": "*fp32",
                "xnumel": "i32",
                "r0_numel": "i32",
                "XBLOCK": "constexpr",
                "RSPLIT_SIZE": "constexpr",
                "NUM_STAGES": "constexpr",
            },
            "device": DeviceProperties.create(torch.device("cuda", 0)),
            "constants": {},
            "native_matmul": False,
            "enable_fp_fusion": True,
            "launch_pdl": False,
            "disable_ftz": False,
            "configs": [
                {
                    (0,): [["tt.divisibility", 16]],
                    (1,): [["tt.divisibility", 16]],
                    (2,): [["tt.divisibility", 16]],
                    (3,): [["tt.divisibility", 16]],
                    (4,): [["tt.divisibility", 16]],
                    (5,): [["tt.divisibility", 16]],
                    (6,): [["tt.divisibility", 16]],
                    (7,): [["tt.divisibility", 16]],
                    (8,): [["tt.divisibility", 16]],
                }
            ],
        },
        inductor_meta={
            "grid_type": "MixOrderReductionGrid",
            "kernel_name": "oracle_layernorm_bwd_mix_order",
            "mutated_arg_names": [],
            "optimize_mem": True,
            "no_x_dim": None,
            "atomic_add_found": False,
            "num_load": 5,
            "num_store": 0,
            "num_reduction": 1,
            "autotune_hints": set(),
            "RSPLIT_SIZE": 64,
            "backend_hash": "oracle",
            "assert_indirect_indexing": True,
            "autotune_local_cache": False,
            "autotune_pointwise": True,
            "autotune_remote_cache": None,
            "force_disable_caches": False,
            "dynamic_scale_rblock": True,
            "incremental_autotune": False,
            "max_autotune": False,
            "max_autotune_pointwise": False,
            "min_split_scan_rblock": 256,
            "spill_threshold": 16,
            "store_cubin": False,
            "deterministic": False,
            "batch_invariant": False,
            "force_filter_reduction_configs": False,
            "mix_order_reduction_allow_multi_stages": True,
            "dynamic_disable_pipelining": True,
            "are_deterministic_algorithms_enabled": False,
            "coordinate_descent_tuning": True,
            "coordinate_descent_search_radius": 1,
            "coordinate_descent_check_all_directions": False,
        },
    )
    @triton.jit
    def _inductor_grad_partial_kernel(
        dy_scalar_ptr,
        gamma_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        out_grad_ptr,
        partial_sum_ptr,
        xnumel,
        r0_numel,
        XBLOCK: tl.constexpr,
        RSPLIT_SIZE: tl.constexpr,
        NUM_STAGES: tl.constexpr,
    ):
        xnumel = 1152000
        r0_numel = 512
        R0_BLOCK: tl.constexpr = 512
        xoffset = tl.program_id(0) * RSPLIT_SIZE
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        r0_index = tl.arange(0, R0_BLOCK)[None, :]
        r0_1 = r0_index
        accum0 = tl.full([R0_BLOCK], 0, tl.float32)[None, :]
        tmp0 = tl.load(dy_scalar_ptr + 0).to(tl.float32)
        tmp1 = tl.broadcast_to(tmp0, [1, 1])
        tmp3 = tl.load(gamma_ptr + r0_1, None, eviction_policy="evict_last")
        tmp2 = tmp1.to(tl.float32)
        tmp4 = tmp2 * tmp3
        tmp18 = tl.full([1, 1], 512.0, tl.float32)
        tmp19 = tmp4 * tmp18
        tmp20 = tl.sum(tmp4, 1)[:, None].to(tl.float32)
        split_size = min(RSPLIT_SIZE, xnumel - xoffset)
        for _ in tl.range(0, split_size, XBLOCK, num_stages=NUM_STAGES):
            xmask = xindex < xnumel
            x0 = xindex
            xindex += XBLOCK
            tmp5 = tl.load(
                x_ptr + (r0_1 + 512 * x0),
                xmask,
                eviction_policy="evict_first",
                other=0.0,
            ).to(tl.float32)
            tmp7 = tl.load(mean_ptr + x0, xmask, eviction_policy="evict_last")
            tmp9 = tl.load(invstd_ptr + x0, xmask, eviction_policy="evict_last")
            tmp6 = tmp5.to(tl.float32)
            tmp8 = tmp6 - tmp7
            tmp10 = tmp8 * tmp9
            tmp11 = tmp4 * tmp10
            tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
            tmp14 = tl.where(xmask, tmp12, 0)
            tmp15 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
            tmp16 = tl.full([1, 1], 0.001953125, tl.float32)
            tmp17 = tmp9 * tmp16
            tmp21 = tmp19 - tmp20
            tmp22 = tmp10 * tmp15
            tmp23 = tmp21 - tmp22
            tmp24 = tmp17 * tmp23
            tmp25 = tmp24.to(tl.float32)
            tmp26 = tmp2 * tmp10
            tl.store(out_grad_ptr + (r0_1 + 512 * x0), tmp25, xmask)
            tmp27 = tl.sum(tmp26, 0)
            tmp28 = accum0 + tmp27
            accum0 = tmp28
        tl.store(
            partial_sum_ptr
            + (tl.program_id(0) + 0 * tl.num_programs(0)) * r0_numel
            + r0_index,
            accum0,
            tl.full([R0_BLOCK], True, tl.int1)[None, :],
        )

    @triton.jit
    def _reduce_partials_stage_kernel(
        partial_sum_ptr,
        stage_sum_ptr,
        NUM_ROW_TILES: tl.constexpr,
        CHANNELS_: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
        BLOCK_CHANNELS_: tl.constexpr,
    ):
        tile_block = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK_CHANNELS_ + tl.arange(0, BLOCK_CHANNELS_)
        tiles = tile_block * BLOCK_TILES + tl.arange(0, BLOCK_TILES)
        mask = tiles[:, None] < NUM_ROW_TILES
        offsets = tiles[:, None] * CHANNELS_ + cols[None, :]
        values = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        reduced = tl.sum(values, axis=0)
        tl.store(stage_sum_ptr + tile_block * CHANNELS_ + cols, reduced)


    @triton.jit
    def _finalize_stage_sum_kernel(
        stage_sum_ptr,
        out_sum_ptr,
        NUM_STAGE_TILES: tl.constexpr,
        CHANNELS_: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
        BLOCK_CHANNELS_: tl.constexpr,
    ):
        cols = tl.program_id(0) * BLOCK_CHANNELS_ + tl.arange(0, BLOCK_CHANNELS_)
        tiles = tl.arange(0, BLOCK_TILES)
        mask = tiles[:, None] < NUM_STAGE_TILES
        offsets = tiles[:, None] * CHANNELS_ + cols[None, :]
        values = tl.load(stage_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_sum_ptr + cols, tl.sum(values, axis=0))


def _validate_inputs(inputs) -> None:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 6:
        raise ValueError(f"expected 6 Repro.forward inputs, got {len(inputs)}")

    dy_scalar, gamma, x, mean, invstd, shape_param = inputs
    expected = {
        "arg4_1": (dy_scalar, (), torch.bfloat16),
        "arg1_1": (gamma, (CHANNELS,), torch.float32),
        "arg0_1": (x, (ROWS, CHANNELS), torch.bfloat16),
        "arg2_1": (mean, (ROWS, 1), torch.float32),
        "arg3_1": (invstd, (ROWS, 1), torch.float32),
    }
    for name, (tensor, shape, dtype) in expected.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.device.type != "cuda":
            raise RuntimeError("CUDA tensors are required for this Triton oracle")
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(
                f"{name} expected {dtype} shape={list(shape)}, "
                f"got dtype={tensor.dtype} shape={list(tensor.shape)}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous for the canonical oracle")

    if _shape_tuple(shape_param) != (ROWS, CHANNELS):
        raise ValueError(f"unexpected _shape_param_0: {shape_param}")


def oracle_forward(inputs):
    """Run the full-scope oracle computation."""
    _validate_inputs(inputs)
    dy_scalar, gamma, x, mean, invstd, shape_param = inputs

    num_row_tiles = triton.cdiv(ROWS, ROW_TILE)
    out_sum = torch.empty(
        (CHANNELS,),
        device=x.device,
        dtype=torch.float32,
    )
    out_grad = torch.empty(
        _shape_tuple(shape_param),
        device=x.device,
        dtype=torch.bfloat16,
    )
    partial_sum = torch.empty(
        (num_row_tiles, CHANNELS),
        device=x.device,
        dtype=torch.float32,
    )
    num_stage_tiles = triton.cdiv(num_row_tiles, PARTIAL_REDUCE_BLOCK_TILES)
    stage_sum = torch.empty(
        (num_stage_tiles, CHANNELS),
        device=x.device,
        dtype=torch.float32,
    )

    stream = get_raw_stream(x.device.index if x.device.index is not None else 0)
    _inductor_grad_partial_kernel.run(
        dy_scalar,
        gamma,
        x,
        mean,
        invstd,
        out_grad,
        partial_sum,
        ROWS,
        CHANNELS,
        stream=stream,
    )

    _reduce_partials_stage_kernel[
        (
            num_stage_tiles,
            triton.cdiv(CHANNELS, PARTIAL_REDUCE_BLOCK_CHANNELS),
        )
    ](
        partial_sum,
        stage_sum,
        NUM_ROW_TILES=num_row_tiles,
        CHANNELS_=CHANNELS,
        BLOCK_TILES=PARTIAL_REDUCE_BLOCK_TILES,
        BLOCK_CHANNELS_=PARTIAL_REDUCE_BLOCK_CHANNELS,
        num_warps=8,
    )

    _finalize_stage_sum_kernel[(triton.cdiv(CHANNELS, FINAL_BLOCK_CHANNELS),)](
        stage_sum,
        out_sum,
        NUM_STAGE_TILES=num_stage_tiles,
        CHANNELS_=CHANNELS,
        BLOCK_TILES=FINAL_STAGE_BLOCK_TILES,
        BLOCK_CHANNELS_=FINAL_BLOCK_CHANNELS,
        num_warps=8,
    )

    return out_sum, out_grad


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

    inputs = get_inputs()
    instance = get_repro_instance()

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
