"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete SqueezeNet masked average-pool backward channel sum by cooperatively split-K reducing the broadcast `arg66_1 / 169` and boolean `where` mask over the full `[512, 13, 13]` reduction domain for every class, whereas Inductor currently lowers the view/expand/div/where/sum chain as a generic fused reduction over the logical `[512, 1000, 13, 13]` producer; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K template that recognizes a broadcasted average-pool gradient with a mask as a tiny per-channel output reduction and partitions the large N/H/W reduction domain into coordinated partials; the fix is COOPERATIVE_SPLIT_K: add a split-K reduction template for broadcast-plus-where average-pool backward sums that emits per-channel partial accumulators and finalizes the returned vector without materializing the expanded producer."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Callable

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps syntax checks usable without Triton.
    triton = None
    tl = None



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

REPRO_ID = "sum_0becf9609ad7"
SHAPE_LABEL = "torchbench_squeezenet1_1_train_001_e99dcbfa"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 512
C = 1000
H = 13
W = 13
HW = H * W
K_TOTAL = N * HW
INV_HW = 1.0 / HW

BLOCK_K = 512
BLOCK_C = 8



def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_inputs(device: torch.device) -> tuple[object, ...]:
    torch.manual_seed(0)
    module = _load_repro_module()
    inputs = module.make_inputs()
    moved: list[object] = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


def reference_outputs(inputs: tuple[object, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().to(device)
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def _assert_inputs(arg66_1: torch.Tensor, arg48_1: torch.Tensor) -> None:
    if arg66_1.shape != (N, C):
        raise ValueError(f"expected arg66_1 shape {(N, C)}, got {tuple(arg66_1.shape)}")
    if arg48_1.shape != (N, C, H, W):
        raise ValueError(f"expected arg48_1 shape {(N, C, H, W)}, got {tuple(arg48_1.shape)}")
    if arg66_1.dtype != torch.float32:
        raise ValueError(f"expected arg66_1 dtype torch.float32, got {arg66_1.dtype}")
    if arg48_1.dtype != torch.bool:
        raise ValueError(f"expected arg48_1 dtype torch.bool, got {arg48_1.dtype}")
    if not arg66_1.is_contiguous() or not arg48_1.is_contiguous():
        raise ValueError("this fixed-shape oracle expects contiguous captured input layouts")
    if arg66_1.device.type != "cuda" or arg48_1.device.type != "cuda":
        raise RuntimeError("the Triton cooperative split-K oracle requires CUDA tensors")


if triton is not None:

    @triton.jit
    def _masked_avgpool_partial_kernel(
        arg66_ptr,
        mask_ptr,
        partial_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        K_TOTAL_: tl.constexpr,
        INV_HW_: tl.constexpr,
        BLOCK_K_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_k = tl.program_id(1)

        c = pid_c * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        k = pid_k * BLOCK_K_ + tl.arange(0, BLOCK_K_)
        active = (k[:, None] < K_TOTAL_) & (c[None, :] < C_)

        n = k // HW_
        spatial = k - n * HW_

        arg66_offsets = n[:, None] * C_ + c[None, :]
        mask_offsets = (n[:, None] * C_ + c[None, :]) * HW_ + spatial[:, None]

        grad = tl.load(arg66_ptr + arg66_offsets, mask=active, other=0.0).to(tl.float32)
        masked = tl.load(mask_ptr + mask_offsets, mask=active, other=1)
        value = tl.where(masked == 0, grad * INV_HW_, 0.0)
        value = tl.where(active, value, 0.0)

        partial = tl.sum(value, axis=0)
        tl.store(partial_ptr + pid_k * C_ + c, partial, mask=c < C_)

    @triton.jit
    def _masked_avgpool_finalize_kernel(
        partial_ptr,
        out_ptr,
        C_: tl.constexpr,
        NUM_K_TILES_: tl.constexpr,
        BLOCK_TILES_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        c = pid_c * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        tiles = tl.arange(0, BLOCK_TILES_)
        active = (tiles[:, None] < NUM_K_TILES_) & (c[None, :] < C_)

        offsets = tiles[:, None] * C_ + c[None, :]
        values = tl.load(partial_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        out = tl.sum(values, axis=0)
        tl.store(out_ptr + c, out, mask=c < C_)


def oracle_full(
    arg66_1: torch.Tensor,
    arg48_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> torch.Tensor:
    del _shape_param_0, _shape_param_1
    if triton is None:
        raise RuntimeError("triton is not available")
    _assert_inputs(arg66_1, arg48_1)

    num_k_tiles = triton.cdiv(K_TOTAL, BLOCK_K)
    block_tiles = triton.next_power_of_2(num_k_tiles)
    partial = torch.empty((num_k_tiles, C), device=arg66_1.device, dtype=torch.float32)
    out = torch.empty((C,), device=arg66_1.device, dtype=torch.float32)

    grid = (triton.cdiv(C, BLOCK_C), num_k_tiles)
    _masked_avgpool_partial_kernel[grid](
        arg66_1,
        arg48_1,
        partial,
        C_=C,
        HW_=HW,
        K_TOTAL_=K_TOTAL,
        INV_HW_=INV_HW,
        BLOCK_K_=BLOCK_K,
        BLOCK_C_=BLOCK_C,
        num_warps=8,
    )
    _masked_avgpool_finalize_kernel[(triton.cdiv(C, BLOCK_C),)](
        partial,
        out,
        C_=C,
        NUM_K_TILES_=num_k_tiles,
        BLOCK_TILES_=block_tiles,
        BLOCK_C_=BLOCK_C,
        num_warps=8,
    )
    return out


def _compare_outputs(
    got_outputs: tuple[torch.Tensor, ...],
    ref_outputs: tuple[torch.Tensor, ...],
    rtol: float,
    atol: float,
) -> bool:
    if len(got_outputs) != len(ref_outputs):
        print(f"output_count: got={len(got_outputs)} expected={len(ref_outputs)}")
        return False

    ok = True
    for idx, (got, ref) in enumerate(zip(got_outputs, ref_outputs)):
        shape_ok = tuple(got.shape) == tuple(ref.shape)
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        value_ok = torch.allclose(got, ref, rtol=rtol, atol=atol)
        diff = (got.float() - ref.float()).abs()
        max_abs = diff.max().item() if diff.numel() else 0.0
        denom = ref.float().abs().clamp_min(1e-12)
        max_rel = (diff / denom).max().item() if diff.numel() else 0.0
        output_ok = shape_ok and dtype_ok and stride_ok and value_ok
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} stride={got.stride()} "
            f"expected_shape={list(ref.shape)} expected_dtype={ref.dtype} expected_stride={ref.stride()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} allclose={value_ok} "
            f"shape_match={shape_ok} dtype_match={dtype_ok} stride_match={stride_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def _time_cuda_us(fn: Callable[[], object], device: torch.device, warmup: int, rep: int) -> tuple[float, float]:
    if device.type != "cuda":
        raise RuntimeError("benchmarking requires a CUDA device")
    for _ in range(warmup):
        fn()
    synchronize(device)

    times: list[float] = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    return times[0], times[len(times) // 2]


@oracle_impl(hardware="H100", shapes="(T([512, 1000], f32), T([512, 1000, 13, 13], b8), S([512, 1000, 1, 1]), S([512, 1000, 13, 13]))")
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
