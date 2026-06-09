"""
Gap diagnosis (classification: SCHEDULER_FUSION): this oracle evaluates the Qwen RoPE-backward epilogue once per `[batch, sequence, head, 128]` row, reuses that producer for the RMSNorm hidden-dimension dot product, writes the transposed `[2048, 2048]` input-gradient side output, and forms the `[128]` weight-gradient sum from the same row-structured value, whereas Inductor currently lowers the slice_scatter/permute/RMSNorm-backward graph as separate generic pointwise and reduction work that materializes the RoPE producer and the RMSNorm gradient before the sibling reductions; Inductor cannot do this today because the scheduler/codegen cannot represent a row-persistent reduction whose producer also has a materialized layout-changing side output and a cross-row sibling reduction over only the rotary head dimension; the fix is SCHEDULER_FUSION: add a fused multi-output RMSNorm-backward/rotary template that keeps the per-row dot product in registers, stores the transposed gradient directly, and emits coordinated partial reductions for the weight gradient.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

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


REPRO_ID = "sum_sum_230671a8764d"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs(device: torch.device) -> tuple[object, ...]:
    from repro_harness import load_shape_configs, make_inputs_from_config

    configs = load_shape_configs(str(REPRO_PATH))
    if configs:
        config = next(iter(configs.values()))
        config = {
            "inputs": [
                {**spec, "device": str(device)}
                if isinstance(spec, dict) and spec.get("kind") == "tensor"
                else spec
                for spec in config["inputs"]
            ]
        }
        inputs = make_inputs_from_config(config)
    else:
        module = _load_repro_module()
        inputs = module.make_inputs()

    return tuple(value.to(device=device) if isinstance(value, torch.Tensor) else value for value in inputs)


def structured_oracle(
    getitem_81: torch.Tensor,
    unsqueeze_6: torch.Tensor,
    full_3: torch.Tensor,
    unsqueeze_7: torch.Tensor,
    arg5_1: torch.Tensor,
    arg315_1: torch.Tensor,
    arg316_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
    _shape_param_5,
) -> tuple[torch.Tensor, torch.Tensor]:
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5

    batch, heads, seq, dims = getitem_81.shape
    half = dims // 2
    tokens = batch * seq
    hidden = heads * dims

    rotary_product = getitem_81 * unsqueeze_6
    rotary_term = torch.cat(
        (rotary_product[..., half:], -rotary_product[..., :half]),
        dim=-1,
    )
    rotary_grad = full_3 + rotary_term + getitem_81 * unsqueeze_7
    row_grad = rotary_grad.permute(0, 2, 1, 3)

    x = arg315_1.view(batch, seq, heads, dims)
    x_f32 = x.float()
    xhat_bf16 = (x_f32 * arg316_1).to(torch.bfloat16)
    weight_grad = (row_grad * xhat_bf16).sum(dim=(0, 1, 2), keepdim=True).view(dims)

    dy_weight = (row_grad * arg5_1).float()
    dot = (dy_weight * x_f32).sum(dim=3, keepdim=True)
    correction = ((dot * -0.5) * (arg316_1**3)).expand_as(dy_weight) / dims
    input_grad = dy_weight * arg316_1 + correction * (x_f32 * 2.0)
    input_grad_t = (
        input_grad.to(torch.bfloat16)
        .contiguous()
        .view(batch, seq, hidden)
        .view(tokens, hidden)
        .permute(1, 0)
    )
    return weight_grad, input_grad_t


@oracle_impl(hardware="H100", shapes="(T([4, 16, 512, 128], bf16), T([1, 1, 512, 128], bf16), T([4, 16, 512, 128], bf16), T([1, 1, 512, 128], bf16), T([128], bf16), T([2048, 2048], bf16), T([4, 512, 16, 1], f32), S([4, 512, 2048]), S([4, 512, -1, 128]), S([128]), S([4, 512, 16, 128]), S([4, 512, 2048]), S([2048, 2048]))")
def oracle_forward(inputs):
    """Thin wrapper for oracle_harness compatibility."""
    return structured_oracle(*inputs)


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

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

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
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
