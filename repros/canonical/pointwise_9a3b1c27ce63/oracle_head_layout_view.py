"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete DeBERTa attention projection layout scope by directly materializing the required fresh contiguous `[B*H, S, D]` clone buffer from the contiguous `[B*S, H*D]` input and returning the final `[B*H, D, S]` permuted view over that buffer, whereas Inductor lowers the captured view/view/permute/clone/view/permute chain through generic layout materialization; Inductor cannot do this today because it does not recognize this attention head-splitting transpose with a non-contiguous returned view as a specialized copy pattern, so the fix is NEW_PATTERN: add a guarded attention-QKV head-layout materialization template that writes the clone storage directly and preserves the final view metadata."""
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
CLASSIFICATION = "NEW_PATTERN"


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _resolve_view_shape(value: Any, numel: int) -> tuple[int, ...]:
    dims = list(_shape_tuple(value))
    neg_one_count = dims.count(-1)
    if neg_one_count > 1:
        raise ValueError(f"only one inferred dimension is valid, got {dims}")
    if neg_one_count == 1:
        known = 1
        for dim in dims:
            if dim != -1:
                known *= dim
        if known == 0 or numel % known != 0:
            raise ValueError(f"cannot infer shape {dims} for numel={numel}")
        dims[dims.index(-1)] = numel // known
    resolved = tuple(dims)
    product = 1
    for dim in resolved:
        product *= dim
    if product != numel:
        raise ValueError(f"shape {resolved} has {product} elements, expected {numel}")
    return resolved


def _validate_and_layout(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, tuple[int, int, int], tuple[int, int, int], int, int, int, int]:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects four inputs, got {len(inputs)}")

    addmm_140, shape0, shape1, shape2 = inputs
    if not isinstance(addmm_140, torch.Tensor):
        raise TypeError(f"{REPRO_ID} first input must be a tensor")
    if addmm_140.dtype is not torch.float32:
        raise ValueError(f"{REPRO_ID} expects torch.float32 input, got {addmm_140.dtype}")
    if addmm_140.ndim != 2:
        raise ValueError(f"{REPRO_ID} expects a rank-2 input, got shape={tuple(addmm_140.shape)}")
    if not addmm_140.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects contiguous input, got stride={tuple(addmm_140.stride())}")

    input_rows = int(addmm_140.shape[0])
    hidden = int(addmm_140.shape[1])
    numel = int(addmm_140.numel())

    batch, seq, shape_hidden = _resolve_view_shape(shape0, numel)
    if (batch * seq, shape_hidden) != (input_rows, hidden):
        raise ValueError(
            f"first view shape {(batch, seq, shape_hidden)} does not match "
            f"input shape={tuple(addmm_140.shape)}"
        )

    batch1, seq1, heads, head_dim = _resolve_view_shape(shape1, numel)
    if (batch1, seq1) != (batch, seq) or heads * head_dim != hidden:
        raise ValueError(
            f"head view shape {(batch1, seq1, heads, head_dim)} is incompatible "
            f"with first view {(batch, seq, hidden)}"
        )

    clone_shape = _resolve_view_shape(shape2, numel)
    expected_clone_shape = (batch * heads, seq, head_dim)
    if clone_shape != expected_clone_shape:
        raise ValueError(f"unexpected clone view shape {clone_shape}, expected {expected_clone_shape}")

    clone_stride = (seq * head_dim, head_dim, 1)
    total_rows = batch * heads * seq
    return addmm_140, clone_shape, clone_stride, total_rows, seq, heads, head_dim


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
        ],
        key=["N", "S", "H", "D"],
    )
    @triton.jit
    def _head_layout_flat_kernel(
        input_ptr,
        clone_ptr,
        N: tl.constexpr,
        S: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        dim = offsets % D
        tmp = offsets // D
        head = tmp % H
        tmp = tmp // H
        seq = tmp % S
        batch = tmp // S

        clone_offsets = ((batch * H + head) * S + seq) * D + dim
        values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        tl.store(clone_ptr + clone_offsets, values, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([4096, 1536], f32), S([8, 512, 1536]), S([8, 512, 24, -1]), S([-1, 512, 64]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward scope and return the same output layout."""
    addmm_140, clone_shape, clone_stride, total_rows, seq, heads, head_dim = _validate_and_layout(inputs)

    if triton is None or not addmm_140.is_cuda:
        view_default = torch.ops.aten.view.default(addmm_140, inputs[1])
        view_default_1 = torch.ops.aten.view.default(view_default, inputs[2])
        permute_default = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3])
        clone_default = torch.ops.aten.clone.default(permute_default, memory_format=torch.contiguous_format)
        view_default_2 = torch.ops.aten.view.default(clone_default, inputs[3])
        return torch.ops.aten.permute.default(view_default_2, [0, 2, 1])

    clone = torch.empty_strided(
        clone_shape,
        clone_stride,
        device=addmm_140.device,
        dtype=addmm_140.dtype,
    )
    n_elements = total_rows * head_dim
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _head_layout_flat_kernel[grid](
        addmm_140,
        clone,
        N=n_elements,
        S=seq,
        H=heads,
        D=head_dim,
    )
    return clone.permute(0, 2, 1)


def _check_layout(instance: torch.nn.Module, inputs: list[Any] | tuple[Any, ...]) -> bool:
    with torch.no_grad():
        eager_out = instance(*inputs)
        oracle_out = oracle_forward(inputs)
        if oracle_out.is_cuda:
            torch.cuda.synchronize()

    layout_ok = (
        tuple(oracle_out.shape) == tuple(eager_out.shape)
        and oracle_out.dtype == eager_out.dtype
        and tuple(oracle_out.stride()) == tuple(eager_out.stride())
        and oracle_out.storage_offset() == eager_out.storage_offset()
        and oracle_out._base is not None
        and eager_out._base is not None
        and oracle_out.untyped_storage().data_ptr() != inputs[0].untyped_storage().data_ptr()
    )
    print(
        f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
        f"(shape={list(oracle_out.shape)} dtype={oracle_out.dtype} "
        f"stride={tuple(oracle_out.stride())} storage_offset={oracle_out.storage_offset()} "
        f"view_base={oracle_out._base is not None})"
    )
    return bool(layout_ok)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
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
        ok = ok and _check_layout(instance, inputs)
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
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
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
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
