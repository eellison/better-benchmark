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


def oracle_forward(inputs: tuple[object, ...] | list[object]) -> torch.Tensor:
    x, y, a = inputs
    return oracle_fused(x, y, a)


def reference_output(inputs: tuple[object, ...]) -> torch.Tensor:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1.0e-8)
    return float(diff.max().item()), float(rel.max().item())


def run_check(
    rtol: float,
    atol: float,
    block_r: int,
    partial_chunks: int,
    num_warps: int,
) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        ref = reference_output(inputs)
        actual = oracle_fused(
            *inputs,
            block_r=block_r,
            partial_chunks=partial_chunks,
            num_warps=num_warps,
        )
        torch.cuda.synchronize()

    shape_ok = actual.shape == ref.shape
    dtype_ok = actual.dtype == ref.dtype
    stride_ok = actual.stride() == ref.stride()
    max_abs, max_rel = _max_diff(actual, ref)
    close_ok = torch.allclose(actual.float(), ref.float(), rtol=rtol, atol=atol)
    ok = shape_ok and dtype_ok and stride_ok and close_ok
    print(
        f"output[0]: shape={list(actual.shape)} dtype={actual.dtype} stride={actual.stride()} "
        f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} allclose={close_ok} "
        f"shape_match={shape_ok} stride_match={stride_ok} dtype_match={dtype_ok}"
    )
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def _compile_with_config(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
    config: dict[str, object],
):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(3):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def run_bench(
    rep: int,
    warmup: int,
    no_compile: bool,
    block_r: int,
    partial_chunks: int,
    num_warps: int,
) -> dict[str, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    x, y, a = inputs
    partials, out = _make_workspace(x.device, partial_chunks)

    timings: dict[str, float] = {}
    with torch.no_grad():
        _oracle_into(x, y, a, partials, out, block_r, partial_chunks, num_warps)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: _oracle_into(x, y, a, partials, out, block_r, partial_chunks, num_warps),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0
    timings["oracle_fused"] = oracle_us
    print(
        f"oracle_fused full-scope sigmoid-product sum: {oracle_us:.3f} us "
        f"(block_r={block_r}, partial_chunks={partial_chunks}, num_warps={num_warps})"
    )

    if not no_compile:
        module = _load_repro_module()
        for label, config in COMPILE_CONFIGS:
            model = module.Repro().cuda()
            with torch.no_grad():
                compiled = _compile_with_config(model, inputs, config)
                compiled_us = triton.testing.do_bench(
                    lambda: compiled(*inputs),
                    warmup=warmup,
                    rep=rep,
                    return_mode="min",
                ) * 1000.0
            timings[label] = compiled_us
            print(f"torch.compile {label}: {compiled_us:.3f} us")

    best_required_compile = min(
        (timings[label] for label, _ in COMPILE_CONFIGS if label in timings),
        default=float("inf"),
    )
    best_gate = min(best_required_compile, HISTORICAL_BEST_COMPILE_US)
    valid_floor = oracle_us < best_gate
    print(f"historical best_compile_us: {HISTORICAL_BEST_COMPILE_US:.3f} us")
    print(f"Valid floor: {'yes' if valid_floor else 'no (diagnosis-only)'}")
    print(
        json.dumps(
            {
                "repro_id": REPRO_ID,
                "oracle_us": round(oracle_us, 3),
                "best_required_compile_us": (
                    round(best_required_compile, 3)
                    if best_required_compile != float("inf")
                    else None
                ),
                "historical_best_compile_us": round(HISTORICAL_BEST_COMPILE_US, 3),
                "valid_floor": valid_floor,
                "classification": "BANDWIDTH_BOUND",
            }
        )
    )
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check against repro.py")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=2e-1)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--block-r", type=int, default=DEFAULT_BLOCK_R)
    parser.add_argument("--partial-chunks", type=int, default=DEFAULT_PARTIAL_CHUNKS)
    parser.add_argument("--num-warps", type=int, default=DEFAULT_NUM_WARPS)
    parser.add_argument("--no-compile", action="store_true", help="only benchmark the oracle")
    args = parser.parse_args()

    if args.rep <= 0:
        raise ValueError("--rep must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(
        rtol=args.rtol,
        atol=args.atol,
        block_r=args.block_r,
        partial_chunks=args.partial_chunks,
        num_warps=args.num_warps,
    ):
        sys.exit(1)
    if args.bench:
        run_bench(
            rep=args.rep,
            warmup=args.warmup,
            no_compile=args.no_compile,
            block_r=args.block_r,
            partial_chunks=args.partial_chunks,
            num_warps=args.num_warps,
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
