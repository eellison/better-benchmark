"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Reformer masked layout-reduction scope returned by Repro.forward, including the [8,4096,256] to [8,64,64,256] view, both logical permutes, bool-to-f32 dropout-mask scaling by 1/0.95, the two last-dimension slices, and both reshaped sum outputs with direct Triton reductions from the original contiguous input, whereas Inductor currently lowers the decomposed view/permute/mul/permute/slice/sum/view graph as generic layout and reduction work for two sibling reductions; Inductor cannot do this today because its scheduler/codegen pattern library does not recognize the shared masked transpose producer feeding two different fixed-axis reductions as one layout-aware multi-output reduction template; the fix is NEW_PATTERN: add a guarded lowering that keeps the view/permute/masked-scale producer virtual and emits direct reductions for both slice consumers without materializing the intermediate layout."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"
CLASSIFICATION = "NEW_PATTERN"


from oracle_harness import (  # noqa: E402
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


BATCH = 8
I_DIM = 64
J_DIM = 64
D_DIM = 256
OUT0_D = 192
OUT1_D = 64
REDUCE = BATCH * I_DIM
MASK_SCALE = 1.0 / 0.95
OUT0_SHAPE = (1, J_DIM, OUT0_D)
OUT0_STRIDE = (J_DIM * OUT0_D, OUT0_D, 1)
OUT1_SHAPE = (I_DIM, 1, OUT1_D)
OUT1_STRIDE = (OUT1_D, OUT1_D, 1)


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _two_sum_kernel(
        x_ptr,
        mask_ptr,
        out0_ptr,
        out1_ptr,
        I_DIM_: tl.constexpr,
        J_DIM_: tl.constexpr,
        D_DIM_: tl.constexpr,
        OUT0_D_: tl.constexpr,
        OUT1_D_: tl.constexpr,
        MASK_SCALE_: tl.constexpr,
        TAIL_D_BLOCKS: tl.constexpr,
        HEAD_D_BLOCKS: tl.constexpr,
        TAIL_TILES: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        tile = tl.program_id(0)
        is_tail = tile < TAIL_TILES

        tail_j = tile // TAIL_D_BLOCKS
        tail_d_block = tile - tail_j * TAIL_D_BLOCKS
        head_tile = tile - TAIL_TILES
        head_i = head_tile // HEAD_D_BLOCKS
        head_d_block = head_tile - head_i * HEAD_D_BLOCKS

        d_base = tl.arange(0, BLOCK_D)
        tail_d = tail_d_block * BLOCK_D + d_base
        head_d = head_d_block * BLOCK_D + d_base
        load_d = tl.where(is_tail, tail_d + OUT1_D_, head_d)
        valid_d = tl.where(is_tail, tail_d < OUT0_D_, head_d < OUT1_D_)

        r = tl.arange(0, BLOCK_R)
        b = r // I_DIM_
        reduce_lane = r - b * I_DIM_

        tail_x_offsets = (
            b[:, None] * (I_DIM_ * J_DIM_ * D_DIM_)
            + (reduce_lane[:, None] * J_DIM_ + tail_j) * D_DIM_
            + load_d[None, :]
        )
        head_x_offsets = (
            b[:, None] * (I_DIM_ * J_DIM_ * D_DIM_)
            + (head_i * J_DIM_ + reduce_lane[:, None]) * D_DIM_
            + load_d[None, :]
        )
        x_offsets = tl.where(is_tail, tail_x_offsets, head_x_offsets)
        mask_offsets = tl.where(is_tail, b * J_DIM_ + tail_j, b * J_DIM_ + reduce_lane)
        scales = tl.load(mask_ptr + mask_offsets).to(tl.float32) * MASK_SCALE_
        values = tl.load(x_ptr + x_offsets, mask=valid_d[None, :], other=0.0).to(tl.float32)
        values *= scales[:, None]
        totals = tl.sum(values, axis=0)
        tl.store(out0_ptr + tail_j * OUT0_D_ + tail_d, totals, mask=is_tail & (tail_d < OUT0_D_))
        tl.store(out1_ptr + head_i * OUT1_D_ + head_d, totals, mask=(tile >= TAIL_TILES) & (head_d < OUT1_D_))


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    x, mask, shape0, shape1, shape2 = inputs
    if not isinstance(x, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError("first two inputs must be tensors")
    if x.dtype is not torch.float32:
        raise TypeError(f"expected float32 activation input, got {x.dtype}")
    if mask.dtype is not torch.bool:
        raise TypeError(f"expected bool mask input, got {mask.dtype}")
    if tuple(x.shape) != (BATCH, I_DIM * J_DIM, D_DIM):
        raise ValueError(f"unexpected activation shape {tuple(x.shape)}")
    if tuple(mask.shape) != (BATCH, J_DIM, 1, 1):
        raise ValueError(f"unexpected mask shape {tuple(mask.shape)}")
    expected_shapes = (
        (BATCH, I_DIM, J_DIM, D_DIM),
        OUT0_SHAPE,
        OUT1_SHAPE,
    )
    actual_shapes = (_shape_tuple(shape0), _shape_tuple(shape1), _shape_tuple(shape2))
    if actual_shapes != expected_shapes:
        raise ValueError(f"unexpected shape params actual={actual_shapes} expected={expected_shapes}")
    if not x.is_contiguous():
        raise ValueError(f"expected contiguous activation input, got stride={x.stride()}")
    if not mask.is_contiguous():
        raise ValueError(f"expected contiguous mask input, got stride={mask.stride()}")
    return x, mask


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    x, mask, shape0, shape1, shape2 = inputs
    view_default = torch.ops.aten.view.default(x, shape0)
    permute_default = torch.ops.aten.permute.default(view_default, [0, 2, 1, 3])
    convert_element_type_default = torch.ops.prims.convert_element_type.default(mask, torch.float32)
    div_scalar = torch.ops.aten.div.Scalar(convert_element_type_default, 0.95)
    mul_tensor = torch.ops.aten.mul.Tensor(permute_default, div_scalar)
    permute_default_1 = torch.ops.aten.permute.default(mul_tensor, [0, 2, 1, 3])
    slice_tensor = torch.ops.aten.slice.Tensor(permute_default_1, 3, 0, OUT1_D)
    slice_tensor_1 = torch.ops.aten.slice.Tensor(permute_default_1, 3, OUT1_D, D_DIM)
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(slice_tensor_1, [0, 1], True)
    view_default_1 = torch.ops.aten.view.default(sum_dim_int_list, shape1)
    sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(slice_tensor, [0, 2], True)
    view_default_2 = torch.ops.aten.view.default(sum_dim_int_list_1, shape2)
    return view_default_1, view_default_2


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full masked two-reduction Repro.forward scope."""
    x, mask = _validate_inputs(inputs)
    if triton is None or not x.is_cuda:
        return _torch_full_scope(inputs)

    out0 = torch.empty_strided(OUT0_SHAPE, OUT0_STRIDE, device=x.device, dtype=x.dtype)
    out1 = torch.empty_strided(OUT1_SHAPE, OUT1_STRIDE, device=x.device, dtype=x.dtype)
    block_d = 32
    tail_d_blocks = triton.cdiv(OUT0_D, block_d)
    head_d_blocks = triton.cdiv(OUT1_D, block_d)
    tail_tiles = J_DIM * tail_d_blocks
    total_tiles = tail_tiles + I_DIM * head_d_blocks
    _two_sum_kernel[(total_tiles,)](
        x,
        mask,
        out0,
        out1,
        I_DIM_=I_DIM,
        J_DIM_=J_DIM,
        D_DIM_=D_DIM,
        OUT0_D_=OUT0_D,
        OUT1_D_=OUT1_D,
        MASK_SCALE_=MASK_SCALE,
        TAIL_D_BLOCKS=tail_d_blocks,
        HEAD_D_BLOCKS=head_d_blocks,
        TAIL_TILES=tail_tiles,
        BLOCK_D=block_d,
        BLOCK_R=REDUCE,
        num_warps=8,
    )
    return out0, out1


# --- CLI entry point ---
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

    # Handle --show-hw early
    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
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
            # The shared harness owns timing so graph capture, GPU locking, and
            # interleaved oracle/compile measurement stay intact.
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
