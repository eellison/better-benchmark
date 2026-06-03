"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full Whisper layer-norm backward return tuple by reducing each row's hidden-dimension summaries inside a row-tiled producer that writes the returned `[8, 1500, 384]` input-gradient add while cooperatively accumulating both returned `[384]` column reductions, whereas Inductor currently schedules the row sums, the input-gradient pointwise epilogue, and the two sibling `sum([0, 1])` reductions as separate generic reduction/pointwise kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps row-local reductions, a materialized side output, and sibling column accumulators in one coordinated producer; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split the large row dimension for compatible layer-norm-backward column reductions, coordinate the partial accumulators, and fuse the dependent input-gradient side store."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_sum_d574fc7bdc59"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 8
SEQ = 1500
M = BATCH * SEQ
D = 384
INV_D = 1.0 / D


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
    (
        mm_6,
        mm_9,
        mm_10,
        arg1_1,
        arg9_1,
        arg10_1,
        arg0_1,
        add_3,
        *_shape_params,
    ) = inputs

    return (
        mm_6.contiguous(),
        mm_9.contiguous(),
        mm_10.contiguous(),
        arg1_1.reshape(M, D).contiguous(),
        arg9_1.reshape(M).contiguous(),
        arg10_1.reshape(M).contiguous(),
        arg0_1.contiguous(),
        add_3.reshape(M, D).contiguous(),
    )


@triton.jit
def _layernorm_backward_tile_kernel(
    mm6_ptr,
    mm9_ptr,
    mm10_ptr,
    x_ptr,
    mean_ptr,
    rstd_ptr,
    weight_ptr,
    residual_ptr,
    partial_sum_dy_xhat_ptr,
    partial_sum_dy_ptr,
    out_md_ptr,
    M_: tl.constexpr,
    D_: tl.constexpr,
    INV_D_: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    d = tl.arange(0, BLOCK_D)
    m_mask = m < M_
    d_mask = d < D_
    mask = m_mask[:, None] & d_mask[None, :]
    offsets = m[:, None] * D_ + d[None, :]

    dy = (
        tl.load(mm6_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        + tl.load(mm9_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        + tl.load(mm10_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    )
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    mean = tl.load(mean_ptr + m, mask=m_mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + m, mask=m_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + d, mask=d_mask, other=0.0).to(tl.float32)

    xhat = (x - mean[:, None]) * rstd[:, None]
    weighted = dy * weight[None, :]
    row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
    row_dot = tl.sum(tl.where(mask, weighted * xhat, 0.0), axis=1)
    ln_grad = rstd[:, None] * INV_D_ * (
        weighted * D_ - row_sum[:, None] - xhat * row_dot[:, None]
    )
    out = residual + ln_grad
    tl.store(out_md_ptr + offsets, out, mask=mask)

    sum_dy_xhat = tl.sum(tl.where(mask, dy * xhat, 0.0), axis=0)
    sum_dy = tl.sum(tl.where(mask, dy, 0.0), axis=0)
    partial_offsets = tl.program_id(0) * D_ + d
    tl.store(partial_sum_dy_xhat_ptr + partial_offsets, sum_dy_xhat, mask=d_mask)
    tl.store(partial_sum_dy_ptr + partial_offsets, sum_dy, mask=d_mask)


@triton.jit
def _finalize_column_sums_kernel(
    partial_sum_dy_xhat_ptr,
    partial_sum_dy_ptr,
    out_sum_dy_xhat_ptr,
    out_sum_dy_ptr,
    NUM_M_TILES: tl.constexpr,
    D_: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tile = tl.arange(0, BLOCK_TILES)
    d = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = (tile[:, None] < NUM_M_TILES) & (d[None, :] < D_)
    offsets = tile[:, None] * D_ + d[None, :]

    partial_xhat = tl.load(partial_sum_dy_xhat_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    partial_dy = tl.load(partial_sum_dy_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    d_mask = d < D_
    tl.store(out_sum_dy_xhat_ptr + d, tl.sum(partial_xhat, axis=0), mask=d_mask)
    tl.store(out_sum_dy_ptr + d, tl.sum(partial_dy, axis=0), mask=d_mask)


def oracle_full(
    mm_6: torch.Tensor,
    mm_9: torch.Tensor,
    mm_10: torch.Tensor,
    x_md: torch.Tensor,
    mean_m: torch.Tensor,
    rstd_m: torch.Tensor,
    weight_d: torch.Tensor,
    residual_md: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert mm_6.shape == (M, D)
    assert mm_9.shape == (M, D)
    assert mm_10.shape == (M, D)
    assert x_md.shape == (M, D)
    assert mean_m.shape == (M,)
    assert rstd_m.shape == (M,)
    assert weight_d.shape == (D,)
    assert residual_md.shape == (M, D)
    assert mm_6.is_contiguous()
    assert mm_9.is_contiguous()
    assert mm_10.is_contiguous()
    assert x_md.is_contiguous()
    assert mean_m.is_contiguous()
    assert rstd_m.is_contiguous()
    assert weight_d.is_contiguous()
    assert residual_md.is_contiguous()

    device = mm_6.device
    block_m = 8
    block_d = 512
    num_m_tiles = triton.cdiv(M, block_m)
    partial_sum_dy_xhat = torch.empty((num_m_tiles, D), device=device, dtype=torch.float32)
    partial_sum_dy = torch.empty((num_m_tiles, D), device=device, dtype=torch.float32)
    out_md = torch.empty((M, D), device=device, dtype=torch.float32)

    _layernorm_backward_tile_kernel[(num_m_tiles,)](
        mm_6,
        mm_9,
        mm_10,
        x_md,
        mean_m,
        rstd_m,
        weight_d,
        residual_md,
        partial_sum_dy_xhat,
        partial_sum_dy,
        out_md,
        M_=M,
        D_=D,
        INV_D_=INV_D,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        num_warps=4,
    )

    sum_dy_xhat = torch.empty((D,), device=device, dtype=torch.float32)
    sum_dy = torch.empty((D,), device=device, dtype=torch.float32)
    block_final_d = 16
    block_tiles = 1 << (num_m_tiles - 1).bit_length()
    _finalize_column_sums_kernel[(triton.cdiv(D, block_final_d),)](
        partial_sum_dy_xhat,
        partial_sum_dy,
        sum_dy_xhat,
        sum_dy,
        NUM_M_TILES=num_m_tiles,
        D_=D,
        BLOCK_TILES=block_tiles,
        BLOCK_D=block_final_d,
        num_warps=8,
    )

    return sum_dy_xhat, sum_dy, out_md.reshape(BATCH, SEQ, D)


def _as_tuple(out: object) -> tuple[torch.Tensor, ...]:
    if isinstance(out, tuple):
        return out
    return (out,)


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


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
        expected = reference_outputs(inputs)
        actual = _as_tuple(oracle_full(*prepare_oracle_inputs(*inputs)))
        torch.cuda.synchronize()

    ok = len(actual) == len(expected)
    if not ok:
        print(f"output_count: actual={len(actual)} expected={len(expected)} allclose=False")

    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, max_rel = _max_diff(got, ref)
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        output_ok = value_ok and dtype_ok and stride_ok
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} "
            f"stride={got.stride()} max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={value_ok} dtype_match={dtype_ok} stride_match={stride_ok}"
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

    torch.manual_seed(0)
    inputs = make_inputs()
    oracle_inputs = prepare_oracle_inputs(*inputs)
    with torch.no_grad():
        oracle_full(*oracle_inputs)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: oracle_full(*oracle_inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0

    print(f"oracle_full cooperative split-k layernorm backward: {oracle_us:.3f} us")

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
