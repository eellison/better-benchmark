"""
Full-scope Triton oracle for sum_bd0a5d156952.

The compiled repro computes one contiguous f32[128] output:

    out[c] = sum_{n,h,w} (x[n,c,h,w] + y[n,c,h,w])
                         * sigmoid(a[n,c,h,w])
                         * (a[n,c,h,w] * (1 - sigmoid(a[n,c,h,w])) + 1)

for the same three contiguous f32[128, 128, 56, 56] inputs as repro.py. This
oracle covers that complete fused pointwise plus channel-reduction scope; it is
not a reduction-subset microbenchmark.

Gap diagnosis (classification: BANDWIDTH_BOUND): the oracle differs from
Inductor by using a hand-written split reduction that reads each original input
once, computes the sigmoid-derived pointwise producer inside the reduction
tile, stores per-channel partials, and finalizes directly into the returned
vector. Inductor already emits a fused reduction plus finalizer for this graph,
so it cannot unlock a separate missing multi-output, scatter-reduce,
algebraic-elimination, or recompute-fusion win here; any remaining difference
is reduction-template tiling and memory/SFU throughput. The relevant Inductor
change would be narrower tuning of the existing pointwise-reduction template
for this channel-reduction layout, and if this oracle does not beat both local
required compile configs plus the historical best, it is diagnosis-only rather
than a true floor.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

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


REPRO_ID = "sum_bd0a5d156952"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 128
C = 128
H = 56
W = 56
HW = H * W
REDUCTION_NUMEL = N * HW
HISTORICAL_BEST_COMPILE_US = 104.47999835014345
DEFAULT_PARTIAL_CHUNKS = 32
DEFAULT_BLOCK_R = 512
DEFAULT_NUM_WARPS = 8

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
    (
        (
            "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
            "coordinate_descent_tuning=True,benchmark_combo_kernel=True,"
            "triton.multi_kernel=3"
        ),
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]

sys.path.insert(0, str(REPO_ROOT))


@triton.jit
def _silu_grad_sum_partials_kernel(
    x_ptr,
    y_ptr,
    a_ptr,
    partials_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    REDUCTION_NUMEL_: tl.constexpr,
    NUM_PARTIALS_PER_CHANNEL: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    c = tl.program_id(0)
    chunk = tl.program_id(1)
    r_offsets = tl.arange(0, BLOCK_R)
    chunk_start = chunk * CHUNK_SIZE
    acc = tl.zeros([BLOCK_R], dtype=tl.float32)

    for r_base in tl.range(0, CHUNK_SIZE, BLOCK_R):
        chunk_r = r_base + r_offsets
        r = chunk_start + chunk_r
        active = (chunk_r < CHUNK_SIZE) & (r < REDUCTION_NUMEL_)
        n = r // HW_
        hw = r - n * HW_
        offsets = n * C_ * HW_ + c * HW_ + hw

        x = tl.load(x_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        a = tl.load(a_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        sigmoid = 1.0 / (libdevice.exp(-a) + 1.0)
        value = (x + y) * sigmoid * (a * (1.0 - sigmoid) + 1.0)
        acc += tl.where(active, value, 0.0)

    tl.store(partials_ptr + c * NUM_PARTIALS_PER_CHANNEL + chunk, tl.sum(acc, axis=0))


@triton.jit
def _finalize_partials_kernel(
    partials_ptr,
    out_ptr,
    NUM_PARTIALS_PER_CHANNEL: tl.constexpr,
    BLOCK_PARTIALS: tl.constexpr,
):
    c = tl.program_id(0)
    partial_offsets = tl.arange(0, BLOCK_PARTIALS)
    mask = partial_offsets < NUM_PARTIALS_PER_CHANNEL
    values = tl.load(
        partials_ptr + c * NUM_PARTIALS_PER_CHANNEL + partial_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(out_ptr + c, tl.sum(values, axis=0))


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def _chunk_size(partial_chunks: int) -> int:
    return triton.cdiv(REDUCTION_NUMEL, partial_chunks)


def _num_partials_per_channel(partial_chunks: int) -> int:
    return partial_chunks


def _make_workspace(
    device: torch.device,
    partial_chunks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    partials = torch.empty(
        (C, _num_partials_per_channel(partial_chunks)),
        device=device,
        dtype=torch.float32,
    )
    out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    return partials, out


def _validate_inputs(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    out: torch.Tensor | None = None,
) -> None:
    for name, tensor in (("x", x), ("y", y), ("a", a)):
        if tensor.device.type != "cuda":
            raise RuntimeError("Triton oracle requires CUDA inputs")
        if tuple(tensor.shape) != (N, C, H, W):
            raise ValueError(f"{name} has unexpected shape {tuple(tensor.shape)}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} has unexpected dtype {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous, got stride {tensor.stride()}")
    if out is not None:
        if tuple(out.shape) != (C,) or out.stride() != (1,) or out.dtype != torch.float32:
            raise ValueError("unexpected output buffer layout")


def _validate_tiling(block_r: int, partial_chunks: int, num_warps: int) -> None:
    if block_r <= 0 or (block_r & (block_r - 1)) != 0:
        raise ValueError("--block-r must be a positive power of two")
    if partial_chunks <= 0:
        raise ValueError("--partial-chunks must be positive")
    if num_warps <= 0:
        raise ValueError("--num-warps must be positive")


def _oracle_into(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    partials: torch.Tensor,
    out: torch.Tensor,
    block_r: int,
    partial_chunks: int,
    num_warps: int,
) -> torch.Tensor:
    _validate_tiling(block_r, partial_chunks, num_warps)
    _validate_inputs(x, y, a, out)

    chunk_size = _chunk_size(partial_chunks)
    num_partials = _num_partials_per_channel(partial_chunks)
    if tuple(partials.shape) != (C, num_partials) or partials.dtype != torch.float32:
        raise ValueError("unexpected partials workspace layout")

    _silu_grad_sum_partials_kernel[(C, partial_chunks)](
        x,
        y,
        a,
        partials,
        C_=C,
        HW_=HW,
        REDUCTION_NUMEL_=REDUCTION_NUMEL,
        NUM_PARTIALS_PER_CHANNEL=num_partials,
        CHUNK_SIZE=chunk_size,
        BLOCK_R=block_r,
        num_warps=num_warps,
    )
    _finalize_partials_kernel[(C,)](
        partials,
        out,
        NUM_PARTIALS_PER_CHANNEL=num_partials,
        BLOCK_PARTIALS=triton.next_power_of_2(num_partials),
        num_warps=1,
    )
    return out


def oracle_fused(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    *,
    block_r: int = DEFAULT_BLOCK_R,
    partial_chunks: int = DEFAULT_PARTIAL_CHUNKS,
    num_warps: int = DEFAULT_NUM_WARPS,
) -> torch.Tensor:
    partials, out = _make_workspace(x.device, partial_chunks)
    return _oracle_into(x, y, a, partials, out, block_r, partial_chunks, num_warps)


@oracle_impl(hardware="H100", shapes="(T([128, 128, 56, 56], f32), T([128, 128, 56, 56], f32), T([128, 128, 56, 56], f32))")
def oracle_forward(inputs: tuple[object, ...] | list[object]) -> torch.Tensor:
    x, y, a = inputs
    return oracle_fused(x, y, a)


def reference_output(inputs: tuple[object, ...]) -> torch.Tensor:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


def main() -> None:
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
        status = "PASS" if ok else "FAIL"
        print(f"Correctness: {status}")
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
