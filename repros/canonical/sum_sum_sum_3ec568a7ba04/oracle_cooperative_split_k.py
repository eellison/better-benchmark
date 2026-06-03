"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle split-K-reduces the NFNet sigmoid-gated spatial product once per channel/batch tile to produce both the returned scalar all-channel sum and the returned `[1536]` sigmoid-gradient channel vector, whereas Inductor currently schedules the scalar sum, spatial `[128, 1536, 1, 1]` reduction, sigmoid-derivative pointwise chain, and final channel sum as separate generic reductions and pointwise kernels over materialized intermediates; Inductor cannot do this today because its scheduler cannot form a cooperative split-K multi-output reduction when one sibling output is a scalar all-channel sum and another is a per-channel reduction with a dependent sigmoid-gradient epilogue; the fix is COOPERATIVE_SPLIT_K: teach Inductor to tile the shared `(N, H, W)` reduction, accumulate multiple output ranks from the same producer, and finalize the scalar and channel reductions together."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_sum_3ec568a7ba04"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 128
C = 1536
H = 6
W = 6
HW = H * W
BLOCK_N = 16
N_TILES = triton.cdiv(N, BLOCK_N)


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


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    getitem, arg409_1, arg406_1, arg147_1 = inputs
    return (
        getitem.contiguous(),
        arg409_1.contiguous(),
        arg406_1.contiguous(),
        arg147_1.contiguous(),
    )


@triton.jit
def _partial_channel_batch_kernel(
    getitem_ptr,
    arg409_ptr,
    arg406_ptr,
    arg147_ptr,
    partial_scalar_ptr,
    partial_channel_ptr,
    N_: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    N_TILES_: tl.constexpr,
    BLOCK_N_: tl.constexpr,
    BLOCK_HW_: tl.constexpr,
):
    c = tl.program_id(0)
    n_tile = tl.program_id(1)

    hw = tl.arange(0, BLOCK_HW_)[:, None]
    n_vec = n_tile * BLOCK_N_ + tl.arange(0, BLOCK_N_)
    n = n_vec[None, :]
    mask = (n < N_) & (hw < HW_)

    offsets = n * (C_ * HW_) + c * HW_ + hw
    getitem = tl.load(getitem_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    arg406 = tl.load(arg406_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sigmoid = tl.sigmoid(tl.load(arg409_ptr + n_vec * C_ + c, mask=n_vec < N_, other=0.0).to(tl.float32))

    scaled_getitem = getitem * 0.2
    scalar_terms = scaled_getitem * ((arg406 * sigmoid) * 2.0)

    scale = tl.load(arg147_ptr).to(tl.float32)
    spatial_grad_terms = (scaled_getitem * scale * 2.0) * arg406
    spatial_grad_sum = tl.sum(spatial_grad_terms, axis=0)
    sigmoid_deriv = sigmoid * (1.0 - sigmoid)
    channel_terms = spatial_grad_sum * sigmoid_deriv

    partial_idx = c * N_TILES_ + n_tile
    tl.store(partial_scalar_ptr + partial_idx, tl.sum(tl.sum(scalar_terms, axis=0), axis=0))
    tl.store(partial_channel_ptr + partial_idx, tl.sum(channel_terms, axis=0))


@triton.jit
def _finalize_channel_kernel(
    partial_scalar_ptr,
    partial_channel_ptr,
    scalar_by_channel_ptr,
    channel_out_ptr,
    N_TILES_: tl.constexpr,
    BLOCK_TILES_: tl.constexpr,
):
    c = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES_)
    mask = tiles < N_TILES_
    offsets = c * N_TILES_ + tiles

    scalar_partials = tl.load(partial_scalar_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    channel_partials = tl.load(partial_channel_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    tl.store(scalar_by_channel_ptr + c, tl.sum(scalar_partials, axis=0))
    tl.store(channel_out_ptr + c, tl.sum(channel_partials, axis=0))


@triton.jit
def _finalize_scalar_kernel(
    scalar_by_channel_ptr,
    scalar_out_ptr,
    C_: tl.constexpr,
    BLOCK_C_: tl.constexpr,
):
    c = tl.arange(0, BLOCK_C_)
    mask = c < C_
    values = tl.load(scalar_by_channel_ptr + c, mask=mask, other=0.0).to(tl.float32)
    tl.store(scalar_out_ptr, tl.sum(values, axis=0))


def oracle_fused(
    getitem: torch.Tensor,
    arg409_1: torch.Tensor,
    arg406_1: torch.Tensor,
    arg147_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert getitem.shape == (N, C, H, W)
    assert arg409_1.shape == (N, C, 1, 1)
    assert arg406_1.shape == (N, C, H, W)
    assert arg147_1.shape == ()
    assert getitem.is_contiguous()
    assert arg409_1.is_contiguous()
    assert arg406_1.is_contiguous()
    assert arg147_1.is_contiguous()

    device = getitem.device
    partial_scalar = torch.empty((C, N_TILES), device=device, dtype=torch.float32)
    partial_channel = torch.empty((C, N_TILES), device=device, dtype=torch.float32)

    _partial_channel_batch_kernel[(C, N_TILES)](
        getitem,
        arg409_1,
        arg406_1,
        arg147_1,
        partial_scalar,
        partial_channel,
        N_=N,
        C_=C,
        HW_=HW,
        N_TILES_=N_TILES,
        BLOCK_N_=BLOCK_N,
        BLOCK_HW_=64,
        num_warps=2,
    )

    scalar_by_channel = torch.empty((C,), device=device, dtype=torch.float32)
    channel_out = torch.empty((C,), device=device, dtype=torch.float32)
    _finalize_channel_kernel[(C,)](
        partial_scalar,
        partial_channel,
        scalar_by_channel,
        channel_out,
        N_TILES_=N_TILES,
        BLOCK_TILES_=8,
        num_warps=1,
    )

    scalar_out = torch.empty((), device=device, dtype=torch.float32)
    _finalize_scalar_kernel[(1,)](
        scalar_by_channel,
        scalar_out,
        C_=C,
        BLOCK_C_=2048,
        num_warps=8,
    )

    return scalar_out, channel_out


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), rel.max().item()


def run_check(rtol: float, atol: float) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        ref = reference_outputs(inputs)
        actual = oracle_fused(*prepare_oracle_inputs(*inputs))
        torch.cuda.synchronize()

    ok = True
    for idx, (got, expected) in enumerate(zip(actual, ref)):
        max_abs, max_rel = _max_diff(got, expected)
        output_ok = torch.allclose(got.float(), expected.float(), rtol=rtol, atol=atol)
        stride_ok = got.stride() == expected.stride()
        ok = ok and output_ok and stride_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} stride={got.stride()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={output_ok} stride_match={stride_ok}"
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


def run_bench(rep: int, warmup: int, no_compile: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    inputs = make_inputs()
    oracle_inputs = prepare_oracle_inputs(*inputs)
    with torch.no_grad():
        oracle_fused(*oracle_inputs)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: oracle_fused(*oracle_inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0

    print(f"oracle_fused cooperative split-k reduction: {oracle_us:.3f} us")

    if no_compile:
        return

    module = _load_repro_module()
    compile_configs = [
        ("default", {}),
        ("coordinate_descent_tuning", {"coordinate_descent_tuning": True}),
    ]

    for label, config in compile_configs:
        model = module.Repro().cuda()
        with torch.no_grad():
            compiled = _compile_with_config(model, inputs, config)
            compiled_us = triton.testing.do_bench(
                lambda: compiled(*inputs),
                warmup=warmup,
                rep=rep,
                return_mode="min",
            ) * 1000.0
        print(f"torch.compile {label}: {compiled_us:.3f} us")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--no-compile", action="store_true", help="only benchmark oracle")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(rep=args.rep, warmup=args.warmup, no_compile=args.no_compile)


if __name__ == "__main__":
    with torch.no_grad():
        main()
