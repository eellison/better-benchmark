"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full Reformer LSH routing block by deriving signed argmax buckets from bmm_12 without materializing cat([x, -x]), sorting bucket*4096+position, and carrying both sorted gathers through the RMS mean normalization and cyclic concatenation outputs returned by Repro.forward, whereas Inductor currently materializes the 128-wide signed cat, emits generic argmax and key arithmetic, crosses an external sort boundary, then schedules the gather, mean, rsqrt, permute, and cat epilogues as separate kernels; Inductor cannot do this today because the scheduler/codegen pattern library has no Reformer LSH routing pattern that canonicalizes signed-argmax bucket construction, bounded-key sort, sorted gathers, and the downstream RMS/cyclic-concat consumers across the sort boundary; the fix is NEW_PATTERN: add an Inductor lowering for this Reformer LSH routing idiom that computes signed buckets directly and fuses sorted-gather epilogues for all returned tensors."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
import triton


REPRO_ID = "argmax_mean_9cf9e9271ff1"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 1
HEADS = 12
SEQ = 4096
DIM = 64
CHUNKS = 64
SIGNED_BUCKETS = 128
FLAT_CHUNKS = BATCH * HEADS * CHUNKS


def _load_repro_module() -> Any:
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this oracle")


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


def make_inputs() -> tuple[Any, ...]:
    module = _load_repro_module()
    return tuple(value.cuda() if isinstance(value, torch.Tensor) else value for value in module.make_inputs())


def prepare_oracle_inputs(*inputs: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bmm_12, permute_62, mm_19, *_shape_params = inputs
    return bmm_12, permute_62, mm_19


def _signed_lsh_bucket(bmm_12: torch.Tensor) -> torch.Tensor:
    x = (
        bmm_12.view(HEADS, SEQ, 1, 1, 1, DIM)
        .permute(3, 0, 4, 1, 5, 2)
        .reshape(BATCH, HEADS, SEQ, DIM)
    )
    pos_val, pos_idx = x.max(dim=-1)
    neg_val, neg_idx = (-x).max(dim=-1)
    return torch.where(pos_val >= neg_val, pos_idx, neg_idx + DIM).to(torch.int64)


def oracle_reformer_lsh_routing(
    bmm_12: torch.Tensor,
    permute_62: torch.Tensor,
    mm_19: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bucket = _signed_lsh_bucket(bmm_12)
    positions = torch.arange(SEQ, device=bmm_12.device, dtype=torch.int64).view(BATCH, 1, SEQ)
    sort_keys = bucket * SEQ + positions
    sorted_keys, sorted_pos = torch.sort(sort_keys, dim=-1)
    gather_index = sorted_pos.unsqueeze(-1).expand(BATCH, HEADS, SEQ, DIM)

    gathered = torch.gather(permute_62, 2, gather_index)
    gathered_view = gathered.view(BATCH, HEADS, CHUNKS, CHUNKS, DIM)
    gathered_out = gathered_view.view(FLAT_CHUNKS, CHUNKS, DIM)

    squared = torch.pow(gathered_view, 2)
    mean_square = squared.mean(dim=-1, keepdim=True)
    rms = torch.rsqrt(mean_square + 1e-6)
    normalized = gathered_view * rms
    div_tensor = normalized / torch.full((), 8.0, dtype=torch.float64)
    shifted = torch.cat([div_tensor[:, :, -1:, :, :], div_tensor[:, :, :-1, :, :]], dim=2)
    normalized_out = (
        torch.cat([shifted, div_tensor], dim=3)
        .permute(0, 1, 2, 4, 3)
        .view(FLAT_CHUNKS, DIM, SIGNED_BUCKETS)
    )

    mm_view = mm_19.view(BATCH, SEQ, HEADS, DIM).permute(0, 2, 1, 3)
    gathered_mm = torch.gather(mm_view, 2, gather_index)
    gathered_mm_view = gathered_mm.view(BATCH, HEADS, CHUNKS, CHUNKS, DIM)
    shifted_mm = torch.cat([gathered_mm_view[:, :, -1:, :, :], gathered_mm_view[:, :, :-1, :, :]], dim=2)
    mm_out = torch.cat([shifted_mm, gathered_mm_view], dim=3).view(FLAT_CHUNKS, SIGNED_BUCKETS, DIM)

    return sorted_keys, gathered_out, normalized_out, mm_out


class ReformerLSHRoutingOracle(torch.nn.Module):
    def forward(
        self,
        bmm_12: torch.Tensor,
        permute_62: torch.Tensor,
        mm_19: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return oracle_reformer_lsh_routing(bmm_12, permute_62, mm_19)


def reference_outputs(inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    if actual.dtype.is_floating_point:
        actual_f32 = actual.float()
        expected_f32 = expected.float()
        diff = (actual_f32 - expected_f32).abs()
        rel = diff / (expected_f32.abs() + 1e-8)
        return diff.max().item(), rel.max().item()

    diff = (actual.to(torch.int64) - expected.to(torch.int64)).abs()
    return diff.max().item(), 0.0


def _metadata_ok(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    return (
        actual.shape == expected.shape
        and actual.dtype == expected.dtype
        and actual.device.type == expected.device.type
        and actual.stride() == expected.stride()
    )


def run_check(rtol: float, atol: float) -> bool:
    _require_cuda()
    torch.manual_seed(0)

    inputs = make_inputs()
    oracle_inputs = prepare_oracle_inputs(*inputs)

    with torch.no_grad():
        expected = reference_outputs(inputs)
        actual = oracle_reformer_lsh_routing(*oracle_inputs)
        torch.cuda.synchronize()

    ok = len(actual) == len(expected)
    if not ok:
        print(f"tuple length mismatch: actual={len(actual)} expected={len(expected)}")

    for idx, (got, ref) in enumerate(zip(actual, expected)):
        meta_ok = _metadata_ok(got, ref)
        max_abs, max_rel = _max_diff(got, ref)
        if got.dtype.is_floating_point:
            value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
            exact = torch.equal(got, ref)
        else:
            value_ok = torch.equal(got, ref)
            exact = value_ok
        ok = ok and meta_ok and value_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} "
            f"stride={list(got.stride())} meta_ok={meta_ok} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"value_ok={value_ok} exact={exact}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def _bench_cuda(fn, warmup: int, rep: int) -> float:
    with torch.no_grad():
        for _ in range(max(1, warmup)):
            fn()
        torch.cuda.synchronize()
        return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="min") * 1000.0


def _compile_model(model: torch.nn.Module, inputs: tuple[Any, ...]) -> torch.nn.Module:
    import torch._dynamo

    torch._dynamo.reset()
    compiled = torch.compile(model)
    with torch.no_grad():
        for _ in range(3):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def run_bench(warmup: int, rep: int, compile_models: bool) -> None:
    _require_cuda()
    torch.manual_seed(0)

    inputs = make_inputs()
    oracle_inputs = prepare_oracle_inputs(*inputs)
    module = _load_repro_module()
    repro_model = module.Repro().cuda()

    print(
        f"oracle shape: bmm=f16[{HEADS}, {SEQ}, {DIM}], "
        f"sorted rows={BATCH * HEADS}, seq={SEQ}, dim={DIM}"
    )

    oracle_us = _bench_cuda(lambda: oracle_reformer_lsh_routing(*oracle_inputs), warmup=warmup, rep=rep)
    print(f"oracle eager full-scope: {oracle_us:.3f} us")

    repro_us = _bench_cuda(lambda: repro_model(*inputs), warmup=warmup, rep=rep)
    print(f"repro eager captured graph: {repro_us:.3f} us")

    if not compile_models:
        return

    oracle_model = ReformerLSHRoutingOracle().cuda()
    try:
        compiled_oracle = _compile_model(oracle_model, oracle_inputs)
        compiled_oracle_us = _bench_cuda(lambda: compiled_oracle(*oracle_inputs), warmup=warmup, rep=rep)
        print(f"oracle torch.compile full-scope: {compiled_oracle_us:.3f} us")
    except Exception as exc:
        print(f"oracle torch.compile full-scope: FAILED ({exc})")

    try:
        compiled_repro = _compile_model(repro_model, inputs)
        compiled_repro_us = _bench_cuda(lambda: compiled_repro(*inputs), warmup=warmup, rep=rep)
        print(f"repro torch.compile captured graph: {compiled_repro_us:.3f} us")
    except Exception as exc:
        print(f"repro torch.compile captured graph: FAILED ({exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate oracle against Repro.forward")
    parser.add_argument("--bench", action="store_true", help="benchmark the oracle")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--compile", action="store_true", help="also benchmark torch.compile versions")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(warmup=args.warmup, rep=args.rep, compile_models=args.compile)


if __name__ == "__main__":
    with torch.no_grad():
        main()
