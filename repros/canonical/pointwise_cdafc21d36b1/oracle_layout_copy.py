"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete permute-clone-view layout materialization as one shape-specialized Triton copy from contiguous `[128, 12, 256, 64]` input storage into the required fresh contiguous `[32768, 768]` output, whereas Inductor lowers the same captured layout-only region through its generic pointwise/layout-copy scheduler; Inductor cannot materially do less local work today because the clone contract requires a non-aliasing output and the operation is dominated by mandatory full-tensor read/write traffic rather than removable arithmetic or fusion boundaries; the fix is BANDWIDTH_BOUND: record this as a bandwidth floor unless broader transpose-copy codegen or launch/memory-system improvements move both implementations."""
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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 128
HEADS = 12
SEQ = 256
DIM = 64
OUT_ROWS = BATCH * SEQ
OUT_COLS = HEADS * DIM
OUT_SHAPE = (OUT_ROWS, OUT_COLS)
OUT_STRIDE = (OUT_COLS, 1)
IN_SHAPE = (BATCH, HEADS, SEQ, DIM)
IN_STRIDE = (HEADS * SEQ * DIM, SEQ * DIM, DIM, 1)
BLOCK_ROWS = 4


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
    def _layout_copy_kernel(
        x,
        out,
        BLOCK_M: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < 32768
        batch = rows // 256
        seq = rows - batch * 256

        cols0 = tl.arange(0, 512)
        head0 = cols0 // 64
        dim0 = cols0 - head0 * 64
        in0 = (
            batch[:, None] * 196608
            + head0[None, :] * 16384
            + seq[:, None] * 64
            + dim0[None, :]
        )
        out0 = rows[:, None] * 768 + cols0[None, :]
        vals0 = tl.load(x + in0, mask=row_mask[:, None])
        tl.store(out + out0, vals0, mask=row_mask[:, None])

        cols1 = 512 + tl.arange(0, 256)
        head1 = cols1 // 64
        dim1 = cols1 - head1 * 64
        in1 = (
            batch[:, None] * 196608
            + head1[None, :] * 16384
            + seq[:, None] * 64
            + dim1[None, :]
        )
        out1 = rows[:, None] * 768 + cols1[None, :]
        vals1 = tl.load(x + in1, mask=row_mask[:, None])
        tl.store(out + out1, vals1, mask=row_mask[:, None])


@oracle_impl(hardware="H100", shapes="(T([128, 12, 256, 64], f32), S([32768, 768]))")
def oracle_forward(inputs):
    """Run the full Repro.forward scope with one Triton layout-copy kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    x, shape_param = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected tensor input 0, got {type(x)!r}")
    if tuple(x.shape) != IN_SHAPE:
        raise ValueError(f"unexpected input shape: {tuple(x.shape)}")
    if tuple(x.stride()) != IN_STRIDE:
        raise ValueError(f"unexpected input stride: {tuple(x.stride())}")
    if x.dtype is not torch.float32:
        raise ValueError(f"unexpected input dtype: {x.dtype}")
    if not x.is_cuda:
        raise ValueError("oracle_layout_copy.py expects CUDA inputs")
    if tuple(shape_param) != OUT_SHAPE:
        raise ValueError(f"unexpected shape input: {shape_param!r}")

    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=x.device,
        dtype=x.dtype,
    )
    _layout_copy_kernel[(triton.cdiv(OUT_ROWS, BLOCK_ROWS),)](
        x,
        out,
        BLOCK_M=BLOCK_ROWS,
        num_warps=8,
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
