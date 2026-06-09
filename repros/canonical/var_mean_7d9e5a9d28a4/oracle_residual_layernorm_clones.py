"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Longformer residual-add LayerNorm and the three separate contiguous clone returns in one Triton row kernel, using the same fp32 population mean, centered variance, epsilon placement, affine order, transpose-clone layout, and libdevice inverse square root as the generated Inductor lowering, whereas Inductor already emits the same fused full-scope schedule; Inductor cannot materially improve this repro today because the required activation, residual, affine reads plus three full output stores dominate the local work rather than an avoidable scheduler split; the fix is BANDWIDTH_BOUND: record this as a full-scope floor unless a broader memory-traffic or multi-output clone optimization changes the required clone contract."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
    from torch._inductor.runtime.triton_helpers import libdevice
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    libdevice = None

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
ROWS = 8192
SEQ = 1024
BATCH = 8
HIDDEN = 768
R_BLOCK = 1024
EPS = 1.0e-5


def get_inputs():
    """Load inputs from the repro's make_inputs."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create Repro() for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _residual_layernorm_clone_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out0_ptr,
        out1_ptr,
        out2_ptr,
        HIDDEN_: tl.constexpr,
        SEQ_: tl.constexpr,
        BATCH_: tl.constexpr,
        EPS_: tl.constexpr,
        R_BLOCK_: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, R_BLOCK_)
        mask = cols < HIDDEN_

        src_offsets = row * HIDDEN_ + cols
        addmm = tl.load(addmm_ptr + src_offsets, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + src_offsets, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_last")
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_last")

        added = addmm + residual
        mean_sum = tl.sum(tl.where(mask, added, 0.0), axis=0).to(tl.float32)
        mean = mean_sum / tl.full((), HIDDEN_, tl.int32).to(tl.float32)
        centered = added - mean
        var_sum = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0).to(tl.float32)
        variance = var_sum / tl.full((), HIDDEN_, tl.float32)
        invstd = libdevice.rsqrt(variance + tl.full((), EPS_, tl.float32))
        out = (centered * invstd) * weight + bias

        seq = row % SEQ_
        batch = row // SEQ_
        dst_offsets = cols + HIDDEN_ * batch + (HIDDEN_ * BATCH_) * seq
        tl.store(out0_ptr + dst_offsets, out, mask=mask)
        tl.store(out1_ptr + dst_offsets, out, mask=mask)
        tl.store(out2_ptr + dst_offsets, out, mask=mask)


def _validate_inputs(inputs):
    if triton is None or libdevice is None:
        raise RuntimeError("Triton with Inductor libdevice helpers is required")
    if len(inputs) != 8:
        raise ValueError(f"expected 8 inputs, got {len(inputs)}")

    addmm, residual, weight, bias, shape0, shape1, shape2, shape3 = inputs
    if addmm.device.type != "cuda":
        raise RuntimeError("oracle requires CUDA inputs")
    expected = (
        (tuple(addmm.shape), addmm.stride(), addmm.dtype),
        (tuple(residual.shape), residual.stride(), residual.dtype),
        (tuple(weight.shape), weight.stride(), weight.dtype),
        (tuple(bias.shape), bias.stride(), bias.dtype),
    )
    actual = (
        ((ROWS, HIDDEN), (HIDDEN, 1), torch.float32),
        ((BATCH, SEQ, HIDDEN), (SEQ * HIDDEN, HIDDEN, 1), torch.float32),
        ((HIDDEN,), (1,), torch.float32),
        ((HIDDEN,), (1,), torch.float32),
    )
    if expected != actual:
        raise ValueError(f"unexpected tensor metadata: {expected}")
    if tuple(shape0) != (BATCH, SEQ, HIDDEN):
        raise ValueError(f"unexpected input view shape: {shape0}")
    for shape in (shape1, shape2, shape3):
        if tuple(shape) != (ROWS, HIDDEN):
            raise ValueError(f"unexpected output view shape: {shape}")
    return addmm, residual, weight, bias, tuple(shape1), tuple(shape2), tuple(shape3)


def oracle_forward(inputs):
    """Compute exactly Repro()(*make_inputs()) for this repro."""
    addmm, residual, weight, bias, shape1, shape2, shape3 = _validate_inputs(inputs)
    out0_3d = torch.empty_strided(
        (SEQ, BATCH, HIDDEN),
        (BATCH * HIDDEN, HIDDEN, 1),
        device=addmm.device,
        dtype=torch.float32,
    )
    out1_3d = torch.empty_strided(
        (SEQ, BATCH, HIDDEN),
        (BATCH * HIDDEN, HIDDEN, 1),
        device=addmm.device,
        dtype=torch.float32,
    )
    out2_3d = torch.empty_strided(
        (SEQ, BATCH, HIDDEN),
        (BATCH * HIDDEN, HIDDEN, 1),
        device=addmm.device,
        dtype=torch.float32,
    )

    _residual_layernorm_clone_kernel[(ROWS,)](
        addmm,
        residual,
        weight,
        bias,
        out0_3d,
        out1_3d,
        out2_3d,
        HIDDEN_=HIDDEN,
        SEQ_=SEQ,
        BATCH_=BATCH,
        EPS_=EPS,
        R_BLOCK_=R_BLOCK,
    )
    return out0_3d.view(shape1), out1_3d.view(shape2), out2_3d.view(shape3)


def _as_list(outputs):
    if isinstance(outputs, (tuple, list)):
        return list(outputs)
    return [outputs]


def _storage_id(tensor):
    return tensor.untyped_storage().data_ptr()


def check_layout_and_alias(instance, inputs):
    with torch.no_grad():
        eager = _as_list(instance(*inputs))
        oracle = _as_list(oracle_forward(inputs))

    ok = True
    for i, (expected, actual) in enumerate(zip(eager, oracle)):
        layout_ok = (
            tuple(actual.shape) == tuple(expected.shape)
            and actual.dtype == expected.dtype
            and actual.stride() == expected.stride()
            and actual.storage_offset() == expected.storage_offset()
            and actual.is_contiguous() == expected.is_contiguous()
        )
        print(
            f"  layout {i}: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(actual.shape)} stride={actual.stride()} "
            f"offset={actual.storage_offset()} contiguous={actual.is_contiguous()})"
        )
        ok = ok and layout_ok

    for i in range(len(oracle)):
        for j in range(i + 1, len(oracle)):
            no_alias = _storage_id(oracle[i]) != _storage_id(oracle[j])
            print(f"  alias {i}-{j}: {'PASS' if no_alias else 'FAIL'}")
            ok = ok and no_alias
    return ok


def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations")
    parser.add_argument("--rep", type=int, default=200, help="Benchmark repetitions")
    parser.add_argument("--no-skip-stochastic", action="store_true", help="Disable stochastic skipping")
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark all shapes from shapes.txt")
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
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        values_ok = check_oracle(
            oracle_forward,
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
            skip_stochastic=not args.no_skip_stochastic,
        )
        layout_ok = check_layout_and_alias(instance, inputs)
        ok = values_ok and layout_ok
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
                    print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")
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
