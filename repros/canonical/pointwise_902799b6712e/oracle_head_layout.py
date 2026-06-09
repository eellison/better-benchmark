"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete transformer attention-head layout materialization by copying the captured contiguous `[B*S,H*D]` projection into the final contiguous `[B*H,S,D]` clone/view layout with a shape-specialized Triton layout-copy kernel, whereas Inductor already lowers the view/permute/expand/clone/view chain to the same required full-tensor layout materialization; Inductor cannot materially do less local work here because the captured region has no arithmetic producer to fuse, no removable scatter/reduction, and is dominated by mandatory f32 reads, f32 stores, and launch overhead; the fix is BANDWIDTH_BOUND: record this as an at-floor full-scope layout-copy case unless broader layout-copy bandwidth or launch-overhead work moves both paths."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
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

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_HEADS": 1, "BLOCK_SEQ": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_HEADS": 2, "BLOCK_SEQ": 16}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_HEADS": 4, "BLOCK_SEQ": 8}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_HEADS": 8, "BLOCK_SEQ": 4}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_HEADS": 16, "BLOCK_SEQ": 2}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_HEADS": 4, "BLOCK_SEQ": 4}, num_warps=4, num_stages=3),
        ],
        key=["B", "H", "S", "D"],
    )
    @triton.jit
    def _head_layout_kernel(
        input_ptr,
        output_ptr,
        B: tl.constexpr,
        H: tl.constexpr,
        S: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_HEADS: tl.constexpr,
        BLOCK_SEQ: tl.constexpr,
    ):
        batch = tl.program_id(0)
        head = tl.program_id(1) * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)[:, None, None]
        seq = tl.program_id(2) * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)[None, :, None]
        dim = tl.arange(0, BLOCK_D)[None, None, :]

        mask = (head < H) & (seq < S) & (dim < D)
        input_offsets = ((batch * S + seq) * H + head) * D + dim
        output_offsets = ((batch * H + head) * S + seq) * D + dim

        values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        tl.store(output_ptr + output_offsets, values, mask=mask)


def _shape_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _resolve_view_shape(shape: tuple[int, ...], numel: int, name: str) -> tuple[int, ...]:
    unknowns = [idx for idx, dim in enumerate(shape) if dim == -1]
    if len(unknowns) > 1:
        raise ValueError(f"{name} has more than one inferred dimension: {shape}")
    if not unknowns:
        return shape

    known_product = 1
    for dim in shape:
        if dim != -1:
            if dim <= 0:
                raise ValueError(f"{name} has non-positive dimension: {shape}")
            known_product *= dim
    if numel % known_product != 0:
        raise ValueError(f"{name}={shape} is incompatible with numel={numel}")

    resolved = list(shape)
    resolved[unknowns[0]] = numel // known_product
    return tuple(resolved)


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


def _next_power_of_2(value: int) -> int:
    if value <= 0:
        raise ValueError(f"expected positive value, got {value}")
    return 1 << (value - 1).bit_length()


def _shares_storage(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    return lhs.untyped_storage().data_ptr() == rhs.untyped_storage().data_ptr()


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, int, int, int, int, tuple[int, int, int]]:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    addmm_69, shape0_raw, shape1_raw, shape2_raw, shape3_raw = inputs
    if not isinstance(addmm_69, torch.Tensor):
        raise TypeError(f"expected tensor input, got {type(addmm_69)!r}")
    if not addmm_69.is_cuda:
        raise RuntimeError("oracle_head_layout.py expects CUDA tensor inputs")
    if addmm_69.dtype != torch.float32:
        raise TypeError(f"unexpected input dtype {addmm_69.dtype}, expected torch.float32")
    if addmm_69.ndim != 2:
        raise ValueError(f"expected rank-2 input, got shape={tuple(addmm_69.shape)}")
    if not addmm_69.is_contiguous():
        raise ValueError(f"expected contiguous input, got stride={tuple(addmm_69.stride())}")

    view0 = _shape_tuple(shape0_raw, "_shape_param_0")
    view1_raw = _shape_tuple(shape1_raw, "_shape_param_1")
    permuted_shape = _shape_tuple(shape2_raw, "_shape_param_2")
    output_shape = _shape_tuple(shape3_raw, "_shape_param_3")

    if len(view0) != 3:
        raise ValueError(f"_shape_param_0 must be rank 3, got {view0}")
    if len(view1_raw) != 4:
        raise ValueError(f"_shape_param_1 must be rank 4, got {view1_raw}")
    if len(permuted_shape) != 4:
        raise ValueError(f"_shape_param_2 must be rank 4, got {permuted_shape}")
    if len(output_shape) != 3:
        raise ValueError(f"_shape_param_3 must be rank 3, got {output_shape}")

    b, h, s, d = permuted_shape
    hidden = h * d
    if min(b, h, s, d) <= 0:
        raise ValueError(f"invalid expanded shape {permuted_shape}")

    expected_view0 = (b, s, hidden)
    if view0 != expected_view0:
        raise ValueError(f"_shape_param_0 is {view0}, expected {expected_view0}")

    view1 = _resolve_view_shape(view1_raw, b * s * hidden, "_shape_param_1")
    expected_view1 = (b, s, h, d)
    if view1 != expected_view1:
        raise ValueError(f"_shape_param_1 resolves to {view1}, expected {expected_view1}")

    expected_input_shape = (b * s, hidden)
    if tuple(addmm_69.shape) != expected_input_shape:
        raise ValueError(f"input shape is {tuple(addmm_69.shape)}, expected {expected_input_shape}")

    expected_output_shape = (b * h, s, d)
    if output_shape != expected_output_shape:
        raise ValueError(f"_shape_param_3 is {output_shape}, expected {expected_output_shape}")

    return addmm_69, b, h, s, d, output_shape


@oracle_impl(hardware="H100", shapes="(T([4096, 4096], f32), S([8, 512, 4096]), S([8, 512, -1, 64]), S([8, 64, 512, 64]), S([512, 512, 64]))")
def oracle_forward(inputs):
    """Run the full Repro.forward scope and return the final cloned layout."""
    addmm_69, b, h, s, d, output_shape = _validate_inputs(inputs)
    output = torch.empty_strided(
        output_shape,
        _contiguous_stride(output_shape),
        device=addmm_69.device,
        dtype=addmm_69.dtype,
    )
    block_d = _next_power_of_2(d)
    grid = lambda meta: (
        b,
        triton.cdiv(h, meta["BLOCK_HEADS"]),
        triton.cdiv(s, meta["BLOCK_SEQ"]),
    )
    _head_layout_kernel[grid](
        addmm_69,
        output,
        B=b,
        H=h,
        S=s,
        D=d,
        BLOCK_D=block_d,
    )
    return output


def _check_layout_alias_and_exact_values(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
) -> bool:
    with torch.no_grad():
        eager = instance(*inputs)
        actual = oracle_forward(inputs)
        if actual.is_cuda:
            torch.cuda.synchronize()

    tensor_inputs = [value for value in inputs if isinstance(value, torch.Tensor)]
    layout_ok = (
        tuple(actual.shape) == tuple(eager.shape)
        and tuple(actual.stride()) == tuple(eager.stride())
        and actual.dtype == eager.dtype
        and actual.storage_offset() == eager.storage_offset()
    )
    alias_ok = all(
        _shares_storage(actual, value) == _shares_storage(eager, value)
        for value in tensor_inputs
    )
    values_ok = torch.equal(eager, actual)
    print(
        f"  output 0 layout/storage: {'PASS' if layout_ok else 'FAIL'} "
        f"(shape={list(actual.shape)} stride={list(actual.stride())} "
        f"dtype={actual.dtype} storage_offset={actual.storage_offset()})"
    )
    print(f"  output 0 input-alias contract: {'PASS' if alias_ok else 'FAIL'}")
    print(f"  output 0 exact: {'PASS' if values_ok else 'FAIL'}")
    return layout_ok and alias_ok and values_ok


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
        ok = _check_layout_alias_and_exact_values(instance, inputs) and ok
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
