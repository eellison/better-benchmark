"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete M2M100 attention backward row-reduction scope in one Triton kernel, including the score view, broadcast mask fill, exp/div probability producer, element mask scale, row sum, fma epilogue, and final contiguous view, whereas Inductor already emits the same fused persistent row reduction for this captured graph; Inductor cannot materially improve this local case through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recomputation because the remaining work is dominated by mandatory f32/bool traffic, one 128-wide reduction, and one exponential per score lane; the fix is BANDWIDTH_BOUND: record this repro as at floor unless broader math or memory-system improvements move both implementations together."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
    from torch._inductor.runtime.triton_helpers import libdevice
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None
    libdevice = None

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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"
ROWS = 64 * 16 * 128
COLS = 128
XBLOCK = 4


def get_inputs():
    """Load inputs from the repro's make_inputs."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create Repro() for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _attention_backward_row_kernel(
        in_out_ptr0,
        in_ptr0,
        in_ptr1,
        in_ptr2,
        in_ptr3,
        in_ptr4,
        in_ptr5,
        in_ptr6,
        in_ptr7,
        xnumel,
        r0_numel,
        XBLOCK: tl.constexpr,
    ):
        xnumel = 131072
        r0_numel = 128
        R0_BLOCK: tl.constexpr = 128
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
        r0_index = tl.arange(0, R0_BLOCK)[None, :]
        r0_offset = 0
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        x4 = xindex
        r0_3 = r0_index
        x0 = xindex % 128
        x2 = xindex // 2048
        tmp0 = tl.load(in_ptr0 + x4, None, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (r0_3 + 128 * x4), None, eviction_policy='evict_first')
        tmp2 = tl.load(in_ptr2 + (r0_3 + 128 * x0 + 16384 * x2), None, eviction_policy='evict_last').to(tl.int1)
        tmp3 = tl.load(in_ptr3 + 0)
        tmp4 = tl.broadcast_to(tmp3, [1, 1])
        tmp8 = tl.load(in_ptr4 + x4, None, eviction_policy='evict_last')
        tmp11 = tl.load(in_ptr5 + x4, None, eviction_policy='evict_last')
        tmp15 = tl.load(in_ptr6 + (r0_3 + 128 * x4), None, eviction_policy='evict_first')
        tmp16 = tl.load(in_ptr7 + (r0_3 + 128 * x4), None, eviction_policy='evict_first').to(tl.int1)
        tmp5 = tl.full([1, 1], float("-inf"), tl.float32)
        tmp6 = tl.where(tmp2, tmp4, tmp5)
        tmp7 = tmp1 + tmp6
        tmp9 = tmp7 - tmp8
        tmp10 = libdevice.exp(tmp9)
        tmp12 = tmp10 / tmp11
        tmp13 = tl.full([1, 1], 0.0, tl.float32)
        tmp14 = tl.where(tmp0, tmp13, tmp12)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tl.full([1, 1], 1.1111111111111112, tl.float32)
        tmp19 = tmp17 * tmp18
        tmp20 = tmp15 * tmp19
        tmp21 = tmp20 * tmp14
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
        tmp24 = tl.sum(tmp22, 1)[:, None].to(tl.float32)
        tmp25 = -tmp14
        tmp26 = tl.fma(tmp25, tmp24, tmp21)
        tl.store(in_out_ptr0 + (r0_3 + 128 * x4), tmp26, None)


def _expect_tensor(
    name: str,
    value: object,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} shape {tuple(value.shape)} != {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} dtype {value.dtype} != {dtype}")
    if value.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if not value.is_contiguous():
        raise ValueError(f"{name} must use the captured contiguous layout")
    return value


def _expect_shape(name: str, value: object, expected: tuple[int, ...]) -> tuple[int, ...]:
    if tuple(value) != expected:
        raise ValueError(f"{name} {value} != {list(expected)}")
    return expected


@oracle_impl(hardware="H100", shapes="(T([1024, 128, 128], f32), T([64, 16, 128, 128], b8), T([64, 1, 128, 128], b8), T([], f32), T([1024, 128, 128], f32), T([64, 16, 128, 1], f32), T([64, 16, 128, 1], f32), T([64, 16, 128, 1], b8), S([64, 16, 128, 128]), S([64, 16, 128, 128]), S([1024, 128, 128]))")
def oracle_forward(inputs):
    """Run the same computation as Repro.forward for the captured inputs."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 11:
        raise ValueError(f"expected 11 inputs, got {len(inputs)}")

    bmm_1 = _expect_tensor("bmm_1", inputs[0], (1024, 128, 128), torch.float32)
    arg17_1 = _expect_tensor("arg17_1", inputs[1], (64, 16, 128, 128), torch.bool)
    arg5_1 = _expect_tensor("arg5_1", inputs[2], (64, 1, 128, 128), torch.bool)
    full = _expect_tensor("full", inputs[3], (), torch.float32)
    arg13_1 = _expect_tensor("arg13_1", inputs[4], (1024, 128, 128), torch.float32)
    arg14_1 = _expect_tensor("arg14_1", inputs[5], (64, 16, 128, 1), torch.float32)
    arg15_1 = _expect_tensor("arg15_1", inputs[6], (64, 16, 128, 1), torch.float32)
    arg16_1 = _expect_tensor("arg16_1", inputs[7], (64, 16, 128, 1), torch.bool)
    _expect_shape("_shape_param_0", inputs[8], (64, 16, 128, 128))
    _expect_shape("_shape_param_1", inputs[9], (64, 16, 128, 128))
    out_shape = _expect_shape("_shape_param_2", inputs[10], (1024, 128, 128))

    out = torch.empty_strided(
        out_shape,
        (out_shape[1] * out_shape[2], out_shape[2], 1),
        device=bmm_1.device,
        dtype=torch.float32,
    )
    grid = (triton.cdiv(ROWS, XBLOCK),)
    _attention_backward_row_kernel[grid](
        out,
        arg16_1,
        arg13_1,
        arg5_1,
        full,
        arg14_1,
        arg15_1,
        bmm_1,
        arg17_1,
        ROWS,
        COLS,
        XBLOCK=XBLOCK,
        num_warps=2,
        num_stages=1,
    )
    return out


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
