"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full MobileViT permute -> contiguous clone -> unsafe_view -> final view Repro.forward contract by directly materializing the contiguous float32[8192, 240] output from the captured float32[512, 4, 16, 60] input in one Triton layout-copy kernel, whereas Inductor currently lowers the decomposed permute/clone/view chain as a generic pointwise layout materialization with scalar index decoding; Inductor cannot do this today because its pointwise scheduler/codegen does not recognize the static permute(0, 2, 1, 3).clone().view(B*S, H*D) layout-copy idiom and emit head-vectorized stores directly in final storage order; the fix is NEW_PATTERN: add a specialized layout-copy lowering for permute-contiguous-flatten materializations with non-power-of-two inner dimensions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps py_compile useful.
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

BATCH = 512
HEADS = 4
SEQ = 16
DIM = 60
PADDED_DIM = 64
HIDDEN = HEADS * DIM
TOTAL_ROWS = BATCH * SEQ
OUT_SHAPE = (TOTAL_ROWS, HIDDEN)
OUT_STRIDE = (HIDDEN, 1)
CLASSIFICATION = "NEW_PATTERN"


def get_inputs() -> tuple[object, ...]:
    """Load inputs from the repro's make_inputs."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create Repro() for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": 4}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 16}, num_warps=8, num_stages=3),
        ],
        key=["TOTAL_ROWS_", "SEQ_"],
    )
    @triton.jit
    def _permute_clone_view_kernel(
        input_ptr,
        output_ptr,
        TOTAL_ROWS_: tl.constexpr,
        SEQ_: tl.constexpr,
        DIM_: tl.constexpr,
        PADDED_DIM_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        dim = tl.arange(0, PADDED_DIM_)
        mask = (rows[:, None] < TOTAL_ROWS_) & (dim[None, :] < DIM_)

        batch = rows // SEQ_
        seq_idx = rows - batch * SEQ_

        input_base = batch[:, None] * (4 * SEQ_ * DIM_) + seq_idx[:, None] * DIM_ + dim[None, :]
        output_base = rows[:, None] * HIDDEN_ + dim[None, :]

        head0 = tl.load(input_ptr + input_base, mask=mask, other=0.0)
        head1 = tl.load(input_ptr + input_base + SEQ_ * DIM_, mask=mask, other=0.0)
        head2 = tl.load(input_ptr + input_base + 2 * SEQ_ * DIM_, mask=mask, other=0.0)
        head3 = tl.load(input_ptr + input_base + 3 * SEQ_ * DIM_, mask=mask, other=0.0)

        tl.store(output_ptr + output_base, head0, mask=mask)
        tl.store(output_ptr + output_base + DIM_, head1, mask=mask)
        tl.store(output_ptr + output_base + 2 * DIM_, head2, mask=mask)
        tl.store(output_ptr + output_base + 3 * DIM_, head3, mask=mask)


def _shape_tuple(value: object, name: str) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable shape, got {type(value)!r}") from exc


def _validate_inputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, tuple[int, int]]:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    x, shape_param = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected arg347_1 to be a tensor, got {type(x)!r}")
    if tuple(x.shape) != (BATCH, HEADS, SEQ, DIM):
        raise ValueError(f"expected input shape {(BATCH, HEADS, SEQ, DIM)}, got {tuple(x.shape)}")
    if x.dtype != torch.float32:
        raise TypeError(f"expected float32 input, got {x.dtype}")
    if not x.is_cuda:
        raise ValueError("oracle requires CUDA input")
    if not x.is_contiguous():
        raise ValueError(f"expected contiguous input layout, got stride={tuple(x.stride())}")

    out_shape = _shape_tuple(shape_param, "_shape_param_0")
    if out_shape != OUT_SHAPE:
        raise ValueError(f"unexpected _shape_param_0 {out_shape}, expected {OUT_SHAPE}")

    return x, out_shape


def oracle_forward(inputs: tuple[object, ...] | list[object]) -> torch.Tensor:
    """Run the full permute -> contiguous clone -> unsafe_view -> view scope."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    x, out_shape = _validate_inputs(tuple(inputs))
    output = torch.empty_strided(
        out_shape,
        OUT_STRIDE,
        device=x.device,
        dtype=x.dtype,
    )
    grid = lambda META: (triton.cdiv(TOTAL_ROWS, META["BLOCK_ROWS"]),)
    _permute_clone_view_kernel[grid](
        x,
        output,
        TOTAL_ROWS_=TOTAL_ROWS,
        SEQ_=SEQ,
        DIM_=DIM,
        PADDED_DIM_=PADDED_DIM,
        HIDDEN_=HIDDEN,
    )
    return output


def _as_tensor_tuple(value: object) -> tuple[torch.Tensor, ...]:
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(item for item in value if isinstance(item, torch.Tensor))
    raise TypeError(f"expected tensor output, got {type(value)!r}")


def _check_output_layout(instance: torch.nn.Module, inputs: tuple[object, ...]) -> bool:
    with torch.no_grad():
        eager_outputs = _as_tensor_tuple(instance(*inputs))
        oracle_outputs = _as_tensor_tuple(oracle_forward(inputs))
        if any(output.is_cuda for output in oracle_outputs):
            torch.cuda.synchronize()

    if len(eager_outputs) != len(oracle_outputs):
        print(
            f"  layout: FAIL output count oracle={len(oracle_outputs)} "
            f"eager={len(eager_outputs)}"
        )
        return False

    ok = True
    for index, (eager, oracle) in enumerate(zip(eager_outputs, oracle_outputs)):
        layout_ok = (
            tuple(oracle.shape) == tuple(eager.shape)
            and oracle.dtype == eager.dtype
            and tuple(oracle.stride()) == tuple(eager.stride())
            and oracle.storage_offset() == eager.storage_offset()
        )
        print(
            f"  output {index} layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(oracle.shape)} dtype={oracle.dtype} "
            f"stride={tuple(oracle.stride())} storage_offset={oracle.storage_offset()})"
        )
        ok = ok and layout_ok
    return ok


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
        ok = _check_output_layout(instance, inputs) and ok
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
