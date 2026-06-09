"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Blenderbot causal-mask fanout as one Triton writer, generating the `[32,1,128,128]` lower-triangular float mask once per physical element and storing the 24 required fresh expanded `[32,32,128,128]` zero-head-stride outputs with distinct storage, whereas Inductor lowers the captured repeated iota/le/where/expand branches through generic pointwise output scheduling; Inductor cannot do this today because the scheduler does not coalesce many identical metadata-expanded causal-mask siblings into one multi-output store while preserving non-aliasing result storages; the fix is SCHEDULER_FUSION: add a guarded multi-output pointwise fusion/CSE path for repeated generated causal masks with zero-stride expand outputs."""
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

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


OUTPUT_COUNT = 24
SOURCE_MASK_SHAPE = (1, 1, 128, 128)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
        ],
        key=["N_ELEMENTS", "Q_LEN", "K_LEN"],
    )
    @triton.jit
    def _causal_mask_fanout_kernel(
        out0,
        out1,
        out2,
        out3,
        out4,
        out5,
        out6,
        out7,
        out8,
        out9,
        out10,
        out11,
        out12,
        out13,
        out14,
        out15,
        out16,
        out17,
        out18,
        out19,
        out20,
        out21,
        out22,
        out23,
        N_ELEMENTS: tl.constexpr,
        Q_LEN: tl.constexpr,
        K_LEN: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_ELEMENTS
        key = offsets % K_LEN
        query = (offsets // K_LEN) % Q_LEN
        values = tl.where(key <= query, 0.0, -float("inf"))

        tl.store(out0 + offsets, values, mask=mask)
        tl.store(out1 + offsets, values, mask=mask)
        tl.store(out2 + offsets, values, mask=mask)
        tl.store(out3 + offsets, values, mask=mask)
        tl.store(out4 + offsets, values, mask=mask)
        tl.store(out5 + offsets, values, mask=mask)
        tl.store(out6 + offsets, values, mask=mask)
        tl.store(out7 + offsets, values, mask=mask)
        tl.store(out8 + offsets, values, mask=mask)
        tl.store(out9 + offsets, values, mask=mask)
        tl.store(out10 + offsets, values, mask=mask)
        tl.store(out11 + offsets, values, mask=mask)
        tl.store(out12 + offsets, values, mask=mask)
        tl.store(out13 + offsets, values, mask=mask)
        tl.store(out14 + offsets, values, mask=mask)
        tl.store(out15 + offsets, values, mask=mask)
        tl.store(out16 + offsets, values, mask=mask)
        tl.store(out17 + offsets, values, mask=mask)
        tl.store(out18 + offsets, values, mask=mask)
        tl.store(out19 + offsets, values, mask=mask)
        tl.store(out20 + offsets, values, mask=mask)
        tl.store(out21 + offsets, values, mask=mask)
        tl.store(out22 + offsets, values, mask=mask)
        tl.store(out23 + offsets, values, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _resolve_expand_shape(value: Any, source_shape: tuple[int, ...]) -> tuple[int, ...]:
    dims = _shape_tuple(value)
    if len(dims) != len(source_shape):
        raise ValueError(f"expected rank-{len(source_shape)} shape, got {dims}")
    resolved = []
    for dim, source_dim in zip(dims, source_shape):
        if dim == -1:
            resolved.append(source_dim)
        elif dim < 0:
            raise ValueError(f"invalid expand dimension {dim} in {dims}")
        else:
            resolved.append(dim)
    return tuple(resolved)


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], int, int, int]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_causal_mask_fanout.py")
    if len(inputs) != OUTPUT_COUNT + 1:
        raise ValueError(f"{REPRO_ID} expects 25 shape inputs, got {len(inputs)}")

    mask_shape = _resolve_expand_shape(inputs[0], SOURCE_MASK_SHAPE)
    output_shape = _shape_tuple(inputs[1])
    if len(output_shape) != 4:
        raise ValueError(f"expected rank-4 output shape, got {output_shape}")

    batch, heads, query_len, key_len = output_shape
    if heads <= 0:
        raise ValueError(f"expanded head dimension must be positive, got {heads}")
    if mask_shape != (batch, 1, query_len, key_len):
        raise ValueError(
            f"mask expand shape {mask_shape} is incompatible with output shape {output_shape}"
        )

    for index, shape in enumerate(inputs[1:], start=1):
        current = _shape_tuple(shape)
        if current != output_shape:
            raise ValueError(
                f"output shape parameter {index} is {current}, expected {output_shape}"
            )

    output_stride = (query_len * key_len, 0, key_len, 1)
    n_elements = batch * query_len * key_len
    return output_shape, output_stride, n_elements, query_len, key_len


def _make_output(
    output_shape: tuple[int, int, int, int],
    output_stride: tuple[int, int, int, int],
    device: torch.device,
) -> torch.Tensor:
    return torch.empty_strided(
        output_shape,
        output_stride,
        device=device,
        dtype=torch.float32,
    )


@oracle_impl(hardware="H100", shapes="(S([32, -1, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]), S([32, 32, 128, 128]))")
def oracle_forward(inputs):
    """Run the full repeated causal-mask expand scope."""
    output_shape, output_stride, n_elements, query_len, key_len = _validate_inputs(inputs)
    device = torch.device("cuda", torch.cuda.current_device())
    outputs = tuple(
        _make_output(output_shape, output_stride, device)
        for _ in range(OUTPUT_COUNT)
    )

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _causal_mask_fanout_kernel[grid](
        outputs[0],
        outputs[1],
        outputs[2],
        outputs[3],
        outputs[4],
        outputs[5],
        outputs[6],
        outputs[7],
        outputs[8],
        outputs[9],
        outputs[10],
        outputs[11],
        outputs[12],
        outputs[13],
        outputs[14],
        outputs[15],
        outputs[16],
        outputs[17],
        outputs[18],
        outputs[19],
        outputs[20],
        outputs[21],
        outputs[22],
        outputs[23],
        N_ELEMENTS=n_elements,
        Q_LEN=query_len,
        K_LEN=key_len,
    )
    return outputs


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
