"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Swin layernorm-backward-style region by fusing the window-unpartition clone, the two per-row 256-wide reductions, the final gradient-input backing store, and both per-channel reduction partials into one Triton pass, then finalizes both channel sums in one Triton reduction kernel, whereas Inductor emits a generic mix-order reduction that materializes a workspace and lets the two final channel reductions run as separate wrapper-level sums; Inductor cannot do this today because its scheduler/codegen has no fused finalizer for multiple sibling reduction accumulators produced by a layout-changing pointwise epilogue; the fix is SCHEDULER_FUSION: teach the persistent reduction template to generate a multi-output workspace finalizer and return the final strided view directly."""
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

N_ROWS = 100352
CHANNELS = 256
ROWS_PER_BLOCK = 64
REDUCE_BLOCK_C = 16


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:
    from torch._C import _cuda_getCurrentRawStream as get_raw_stream
    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.hints import DeviceProperties, ReductionHint

    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.persistent_reduction(
        size_hints={"x": 131072, "r0_": 256},
        reduction_hint=ReductionHint.DEFAULT,
        filename=__file__,
        triton_meta={
            "signature": {
                "in_ptr0": "*fp32",
                "in_ptr1": "*fp32",
                "in_ptr2": "*fp32",
                "in_ptr3": "*fp32",
                "in_ptr4": "*fp32",
                "in_ptr5": "*fp32",
                "out_ptr2": "*fp32",
                "ws_ptr": "*fp32",
                "xnumel": "i32",
                "r0_numel": "i32",
                "XBLOCK": "constexpr",
                "RSPLIT_SIZE": "constexpr",
                "NUM_STAGES": "constexpr",
            },
            "device": DeviceProperties(
                type="cuda",
                index=0,
                multi_processor_count=132,
                cc=90,
                major=9,
                regs_per_multiprocessor=65536,
                max_threads_per_multi_processor=2048,
                max_threads_per_block=1024,
                warp_size=32,
            ),
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
                    (9,): [["tt.divisibility", 16]],
                }
            ],
        },
        inductor_meta={
            "grid_type": "MixOrderReductionGrid",
            "kernel_name": "_swin_layernorm_bwd_partials_persistent_kernel",
            "mutated_arg_names": [],
            "optimize_mem": True,
            "no_x_dim": None,
            "atomic_add_found": False,
            "num_load": 6,
            "num_store": -1,
            "num_reduction": 2,
            "autotune_hints": set(),
            "RSPLIT_SIZE": 64,
            "backend_hash": "2FF7A49C450C045FF2AC7A730B5A8D4CD6DF40914BC38271CC70D96733C19B52",
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
        },
    )
    @triton.jit
    def _swin_layernorm_bwd_partials_persistent_kernel(
        in_ptr0,
        in_ptr1,
        in_ptr2,
        in_ptr3,
        in_ptr4,
        in_ptr5,
        out_ptr2,
        ws_ptr,
        xnumel,
        r0_numel,
        XBLOCK: tl.constexpr,
        RSPLIT_SIZE: tl.constexpr,
        NUM_STAGES: tl.constexpr,
    ):
        xnumel = 100352
        r0_numel = 256
        R0_BLOCK: tl.constexpr = 256
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = tl.program_id(0) * RSPLIT_SIZE
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        r0_index = tl.arange(0, R0_BLOCK)[None, :]
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        r0_3 = r0_index
        accum0 = tl.full([R0_BLOCK], 0, tl.float32)[None, :]
        accum1 = tl.full([R0_BLOCK], 0, tl.float32)[None, :]
        split_size = min(RSPLIT_SIZE, xnumel - xoffset)
        for _ in tl.range(0, split_size, XBLOCK, num_stages=NUM_STAGES):
            xmask = xindex < xnumel
            x0 = xindex % 28
            x1 = (xindex // 28) % 28
            x2 = xindex // 784
            x4 = xindex
            xindex += XBLOCK
            tmp0 = tl.load(
                in_ptr0
                + (
                    r0_3
                    + 256 * (x0 % 7)
                    + 1792 * (x1 % 7)
                    + 12544 * (x0 // 7)
                    + 50176 * (x1 // 7)
                    + 200704 * x2
                ),
                xmask,
                eviction_policy="evict_first",
                other=0.0,
            )
            tmp1 = tl.load(in_ptr1 + r0_3, None, eviction_policy="evict_last")
            tmp7 = tl.load(
                in_ptr2 + (r0_3 + 256 * x4),
                xmask,
                eviction_policy="evict_first",
                other=0.0,
            )
            tmp8 = tl.load(in_ptr3 + x4, xmask, eviction_policy="evict_last")
            tmp10 = tl.load(in_ptr4 + x4, xmask, eviction_policy="evict_last")
            tmp17 = tl.load(
                in_ptr5 + (r0_3 + 256 * x4),
                xmask,
                eviction_policy="evict_first",
                other=0.0,
            )
            tmp2 = tmp0 * tmp1
            tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
            tmp5 = tl.where(xmask, tmp3, 0)
            tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
            tmp9 = tmp7 - tmp8
            tmp11 = tmp9 * tmp10
            tmp12 = tmp2 * tmp11
            tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
            tmp15 = tl.where(xmask, tmp13, 0)
            tmp16 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
            tmp18 = tl.full([1, 1], 0.00390625, tl.float32)
            tmp19 = tmp10 * tmp18
            tmp20 = tl.full([1, 1], 256.0, tl.float32)
            tmp21 = tmp2 * tmp20
            tmp22 = tmp21 - tmp6
            tmp23 = tmp11 * tmp16
            tmp24 = tmp22 - tmp23
            tmp25 = tmp19 * tmp24
            tmp26 = tmp17 + tmp25
            tmp27 = tmp0 * tmp11
            tl.store(out_ptr2 + (r0_3 + 256 * x4), tmp26, xmask)
            tmp28 = tl.sum(tmp27, 0)
            accum0 += tmp28
            tmp30 = tl.sum(tmp0, 0)
            accum1 += tmp30
        tl.store(
            ws_ptr + (tl.program_id(0) + 0 * tl.num_programs(0)) * r0_numel + r0_index,
            accum0,
            r0_mask,
        )
        tl.store(
            ws_ptr + (tl.program_id(0) + 1 * tl.num_programs(0)) * r0_numel + r0_index,
            accum1,
            r0_mask,
        )

    @triton.jit
    def _finalize_channel_sums_kernel(
        partials_ptr,
        out_weight_ptr,
        out_bias_ptr,
        NUM_BLOCKS_: tl.constexpr,
        CHANNELS_: tl.constexpr,
        BLOCK_B_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        channel_block = tl.program_id(0)
        blocks = tl.arange(0, BLOCK_B_)[:, None]
        cols = channel_block * BLOCK_C_ + tl.arange(0, BLOCK_C_)[None, :]
        mask = (blocks < NUM_BLOCKS_) & (cols < CHANNELS_)
        offsets = blocks * CHANNELS_ + cols

        partial_weight = tl.load(partials_ptr + offsets, mask=mask, other=0.0)
        partial_bias = tl.load(
            partials_ptr + (NUM_BLOCKS_ * CHANNELS_) + offsets,
            mask=mask,
            other=0.0,
        )
        sum_weight = tl.sum(partial_weight, axis=0)
        sum_bias = tl.sum(partial_bias, axis=0)
        out_cols = channel_block * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        out_mask = out_cols < CHANNELS_
        tl.store(out_weight_ptr + out_cols, sum_weight, mask=out_mask)
        tl.store(out_bias_ptr + out_cols, sum_bias, mask=out_mask)


def _validate_inputs(inputs: tuple[object, ...] | list[object]) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 12:
        raise ValueError(f"expected 12 inputs, got {len(inputs)}")

    (
        mm_180,
        arg20_1,
        arg205_1,
        arg206_1,
        arg207_1,
        view_712,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
        shape5,
    ) = inputs

    expected_shapes = (
        [2048, 49, 256],
        [2048, 7, 7, 256],
        [128, 4, 4, 7, 7, 256],
        [128, 28, 28, 256],
        [128, 28, 28, 256],
        [100352, 256],
    )
    if (shape0, shape1, shape2, shape3, shape4, shape5) != expected_shapes:
        raise ValueError("unexpected symbolic shape parameters for this fixed-shape oracle")

    tensor_inputs = (mm_180, arg20_1, arg205_1, arg206_1, arg207_1, view_712)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensor_inputs):
        raise TypeError("first six inputs must be tensors")
    if mm_180.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")

    expected = (
        ((N_ROWS, CHANNELS), (CHANNELS, 1)),
        ((CHANNELS,), (1,)),
        ((N_ROWS, CHANNELS), (CHANNELS, 1)),
        ((128, 28, 28, 1), (784, 28, 1, 1)),
        ((128, 28, 28, 1), (784, 28, 1, 1)),
        ((128, 28, 28, CHANNELS), (200704, 7168, CHANNELS, 1)),
    )
    for idx, (tensor, (shape, stride)) in enumerate(zip(tensor_inputs, expected)):
        if tensor.dtype != torch.float32:
            raise ValueError(f"input {idx} expected torch.float32, got {tensor.dtype}")
        if tuple(tensor.shape) != shape or tuple(tensor.stride()) != stride:
            raise ValueError(
                f"input {idx} expected shape={shape} stride={stride}, "
                f"got shape={tuple(tensor.shape)} stride={tuple(tensor.stride())}"
            )

    return tensor_inputs


def oracle_forward(inputs):
    """Compute the exact full Repro.forward scope for the default captured shape."""
    mm_180, arg20_1, arg205_1, arg206_1, arg207_1, view_712 = _validate_inputs(inputs)

    num_blocks = triton.cdiv(N_ROWS, ROWS_PER_BLOCK)
    reduce_block_b = triton.next_power_of_2(num_blocks)

    out_weight = torch.empty_strided(
        (CHANNELS,),
        (1,),
        device=mm_180.device,
        dtype=torch.float32,
    )
    out_bias = torch.empty_strided(
        (CHANNELS,),
        (1,),
        device=mm_180.device,
        dtype=torch.float32,
    )
    out_base = torch.empty_strided(
        (N_ROWS, CHANNELS),
        (CHANNELS, 1),
        device=mm_180.device,
        dtype=torch.float32,
    )
    partials = torch.empty_strided(
        (2, num_blocks, CHANNELS),
        (num_blocks * CHANNELS, CHANNELS, 1),
        device=mm_180.device,
        dtype=torch.float32,
    )

    raw_stream = get_raw_stream(mm_180.device.index or 0)
    _swin_layernorm_bwd_partials_persistent_kernel.run(
        mm_180,
        arg20_1,
        arg205_1,
        arg206_1,
        arg207_1,
        view_712,
        out_base,
        partials,
        N_ROWS,
        CHANNELS,
        stream=raw_stream,
    )
    _finalize_channel_sums_kernel[(triton.cdiv(CHANNELS, REDUCE_BLOCK_C),)](
        partials,
        out_weight,
        out_bias,
        NUM_BLOCKS_=num_blocks,
        CHANNELS_=CHANNELS,
        BLOCK_B_=reduce_block_b,
        BLOCK_C_=REDUCE_BLOCK_C,
        num_warps=8,
    )

    return (
        out_weight,
        out_bias,
        out_base.as_strided((CHANNELS, N_ROWS), (1, CHANNELS)),
    )


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
