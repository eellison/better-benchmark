"""Gap diagnosis (classification: NEW_PATTERN): this oracle materializes the full Visformer reshape/permute/clone/reshape layout transform with specialized Triton pack and unpack kernels for the captured attention-to-image and image-to-attention shapes, whereas Inductor lowers the chain as a generic layout copy with rank-polymorphic indexing; Inductor cannot do this today because its layout-copy codegen does not recognize this fixed head/channel/spatial transpose family and emit the direct affine pack/unpack mapping for both captured directions; the fix is NEW_PATTERN: add a Visformer-style layout materialization specialization that dispatches from the reshape parameters and writes the final contiguous output layout directly."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
    triton = None
    tl = None


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"


from oracle_harness import (
    oracle_impl,  # noqa: E402
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _attention_to_image_flat_kernel(
        x_ptr,
        out_ptr,
        H: tl.constexpr,
        P: tl.constexpr,
        C: tl.constexpr,
        TOTAL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < TOTAL

        p = offsets % P
        hidden_channel = (offsets // P) % (H * C)
        batch = offsets // (H * C * P)
        head = hidden_channel // C
        channel = hidden_channel - head * C

        input_offsets = ((batch * H + head) * P + p) * C + channel
        values = tl.load(x_ptr + input_offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, values, mask=mask)

    @triton.jit
    def _attention_to_image_tile_kernel(
        x_ptr,
        out_ptr,
        H: tl.constexpr,
        P: tl.constexpr,
        C: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        matrix = tl.program_id(0)
        c_block = tl.program_id(1)
        p_block = tl.program_id(2)

        p = p_block * BLOCK_P + tl.arange(0, BLOCK_P)
        c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)

        batch = matrix // H
        head = matrix - batch * H

        input_base = matrix * P * C
        output_base = batch * H * C * P + head * C * P

        input_offsets = input_base + c[:, None] + p[None, :] * C
        output_offsets = output_base + c[:, None] * P + p[None, :]
        mask = (c[:, None] < C) & (p[None, :] < P)

        values = tl.load(x_ptr + input_offsets, mask=mask, other=0.0)
        tl.store(out_ptr + output_offsets, values, mask=mask)

    @triton.jit
    def _image_to_attention_kernel(
        x_ptr,
        out_ptr,
        H: tl.constexpr,
        P: tl.constexpr,
        C: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        matrix = tl.program_id(0)
        c_block = tl.program_id(1)
        p_block = tl.program_id(2)

        p = p_block * BLOCK_P + tl.arange(0, BLOCK_P)
        c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)

        batch = matrix // H
        head = matrix - batch * H
        hidden = H * C

        input_base = batch * hidden * P + head * C
        output_base = matrix * P * C

        input_offsets = input_base + p[:, None] * hidden + c[None, :]
        output_offsets = output_base + p[:, None] * C + c[None, :]
        mask = (p[:, None] < P) & (c[None, :] < C)

        values = tl.load(x_ptr + input_offsets, mask=mask, other=0.0)
        tl.store(out_ptr + output_offsets, values, mask=mask)


def _as_shape(value) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


def _require_cuda_tensor(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype is not torch.float32:
        raise ValueError(f"{name} must be torch.float32, got {tensor.dtype}")


def _attention_to_image(x: torch.Tensor, shape0, shape1) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")

    batch, heads, positions, channels = _as_shape(shape0)
    out_shape = _as_shape(shape1)
    expected_input = (batch * heads, positions, channels)
    expected_output = (batch, heads * channels, int(positions**0.5), int(positions**0.5))

    if tuple(x.shape) != expected_input:
        raise ValueError(f"unexpected input shape {tuple(x.shape)}, expected {expected_input}")
    if out_shape != expected_output:
        raise ValueError(f"unexpected output shape parameter {out_shape}, expected {expected_output}")
    if not x.is_contiguous():
        raise ValueError(f"attention input must be contiguous, got stride={tuple(x.stride())}")

    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    block_p = 64 if positions <= 64 else 32
    block_c = 128 if channels == 128 else 64
    grid = (batch * heads, triton.cdiv(channels, block_c), triton.cdiv(positions, block_p))
    _attention_to_image_tile_kernel[grid](
        x,
        out,
        H=heads,
        P=positions,
        C=channels,
        BLOCK_P=block_p,
        BLOCK_C=block_c,
        num_warps=8,
    )
    return out


def _image_to_attention(x: torch.Tensor, shape0, shape1) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")

    batch, heads, channels, positions = _as_shape(shape0)
    out_shape = _as_shape(shape1)
    side = int(positions**0.5)
    expected_input = (batch, heads * channels, side, side)
    expected_output = (batch * heads, positions, channels)
    expected_stride = (heads * channels * positions, 1, side * heads * channels, heads * channels)

    if tuple(x.shape) != expected_input:
        raise ValueError(f"unexpected input shape {tuple(x.shape)}, expected {expected_input}")
    if out_shape != expected_output:
        raise ValueError(f"unexpected output shape parameter {out_shape}, expected {expected_output}")
    if tuple(x.stride()) != expected_stride:
        raise ValueError(f"image input must have stride {expected_stride}, got {tuple(x.stride())}")

    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    block_p = 16
    block_c = 32
    grid = (batch * heads, triton.cdiv(channels, block_c), triton.cdiv(positions, block_p))
    _image_to_attention_kernel[grid](
        x,
        out,
        H=heads,
        P=positions,
        C=channels,
        BLOCK_P=block_p,
        BLOCK_C=block_c,
        num_warps=4,
    )
    return out


@oracle_impl(hardware="H100", shapes="(T([768, 49, 128], f32), S([128, 6, 49, 128]), S([128, 768, 7, 7]))")
def oracle_forward(inputs):
    """Run the full Repro.forward scope for all captured Visformer layout shapes."""
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects three inputs, got {len(inputs)}")

    x, shape0, shape1 = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"first input must be a tensor, got {type(x)!r}")
    _require_cuda_tensor("first input", x)

    if x.ndim == 3:
        return _attention_to_image(x, shape0, shape1)
    if x.ndim == 4:
        return _image_to_attention(x, shape0, shape1)
    raise ValueError(f"unsupported input rank {x.ndim} for shape {tuple(x.shape)}")


def _check_output_layout(output: torch.Tensor, shape1) -> bool:
    expected_shape = _as_shape(shape1)
    expected_stride = _contiguous_stride(expected_shape)
    return (
        tuple(output.shape) == expected_shape
        and tuple(output.stride()) == expected_stride
        and output.storage_offset() == 0
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
        with torch.no_grad():
            layout_output = oracle_forward(inputs)
            if layout_output.is_cuda:
                torch.cuda.synchronize()
        layout_ok = _check_output_layout(layout_output, inputs[2])
        print(
            f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(layout_output.shape)} stride={list(layout_output.stride())})"
        )
        ok = ok and layout_ok
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
