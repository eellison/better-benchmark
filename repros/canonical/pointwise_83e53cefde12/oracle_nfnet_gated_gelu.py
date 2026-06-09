"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete captured NFNet inference gated exact-GELU scope in one shape-specialized Triton pointwise kernel, including the `[128, C, 1, 1]` sigmoid gate broadcast over spatial positions, every fp16 rounding boundary before the fp32 GELU, the final two fp16 scale multiplies, and the contiguous `[128, C, H, W]` output contract, whereas Inductor already lowers the same full graph to a single fused pointwise kernel with the broadcast indexing folded into codegen; Inductor cannot materially improve this repro through local scheduler fusion because there is no remaining stencil, reduction, scatter, or layout materialization to eliminate and the work is dominated by required activation/residual reads, transcendental sigmoid/erf math, and the output store; the fix is BANDWIDTH_BOUND: treat this row as at the pointwise math/memory floor unless broader pointwise math codegen or launch-overhead changes move both implementations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


CLASSIFICATION = "BANDWIDTH_BOUND"


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
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
        ],
        key=["n_elements", "spatial"],
    )
    @triton.jit
    def _nfnet_gated_gelu_kernel(
        gate_ptr,
        payload_ptr,
        scalar_ptr,
        residual_ptr,
        output_ptr,
        n_elements: tl.constexpr,
        spatial: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        gate_offsets = offsets // spatial
        gate = tl.sigmoid(tl.load(gate_ptr + gate_offsets, mask=mask, other=0.0).to(tl.float32)).to(tl.float16)
        payload = tl.load(payload_ptr + offsets, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
        scalar = tl.load(scalar_ptr).to(tl.float16)

        x = (payload * gate).to(tl.float16)
        x = (x * 2.0).to(tl.float16)
        x = (x * scalar).to(tl.float16)
        x = (x * 0.2).to(tl.float16)
        x = (x + residual).to(tl.float16)

        x_f32 = x.to(tl.float32)
        erf_term = tl.math.erf(x_f32 * 0.7071067811865476) + 1.0
        gelu = x_f32 * 0.5 * erf_term
        gelu_h = gelu.to(tl.float16)
        scaled = (gelu_h * 1.7015043497085571).to(tl.float16)
        output = (scaled * 0.9622504486493761).to(tl.float16)
        tl.store(output_ptr + offsets, output, mask=mask)


def _validate_inputs(inputs):
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    gate, payload, scalar, residual = inputs
    for index, value in enumerate((gate, payload, scalar, residual)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{REPRO_ID} input {index} must be a tensor")
        if value.dtype != torch.float16:
            raise TypeError(f"{REPRO_ID} input {index} must be f16, got {value.dtype}")
        if not value.is_cuda:
            raise ValueError(f"{REPRO_ID} input {index} must be CUDA")

    if gate.dim() != 4 or payload.dim() != 4 or residual.dim() != 4:
        raise ValueError(f"{REPRO_ID} expects 4D gate/payload/residual tensors")
    if scalar.dim() != 0:
        raise ValueError(f"{REPRO_ID} expects a scalar third input")
    if gate.shape[:2] != payload.shape[:2] or tuple(gate.shape[2:]) != (1, 1):
        raise ValueError(
            f"{REPRO_ID} gate shape {tuple(gate.shape)} is incompatible with "
            f"payload shape {tuple(payload.shape)}"
        )
    if residual.shape != payload.shape:
        raise ValueError(
            f"{REPRO_ID} residual shape {tuple(residual.shape)} must match "
            f"payload shape {tuple(payload.shape)}"
        )
    if not gate.is_contiguous() or not payload.is_contiguous() or not residual.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects the captured contiguous input layouts")

    return gate, payload, scalar, residual


@oracle_impl(hardware="H100", shapes="(T([128, 1536, 1, 1], f16), T([128, 1536, 6, 6], f16), T([], f16), T([128, 1536, 6, 6], f16))")
def oracle_forward(inputs):
    """Run the complete Repro.forward computation for the captured pointwise scope."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_nfnet_gated_gelu.py")

    gate, payload, scalar, residual = _validate_inputs(inputs)
    output = torch.empty_strided(
        tuple(payload.shape),
        tuple(payload.stride()),
        device=payload.device,
        dtype=torch.float16,
    )

    n_elements = payload.numel()
    spatial = payload.shape[2] * payload.shape[3]
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _nfnet_gated_gelu_kernel[grid](
        gate,
        payload,
        scalar,
        residual,
        output,
        n_elements=n_elements,
        spatial=spatial,
    )
    return output


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
