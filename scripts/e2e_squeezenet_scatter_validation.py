"""End-to-end validation: does the scatter_add gather-reduce elimination
(pytorch-work 5489b8c2bb9, sum_sum_3219a09ab96a) compose to the FULL
SqueezeNet backward graph, or was the repro-level gap a partitioner artifact?

Runs ONE condition per process (fresh inductor state). Drive with:

  CUDA_VISIBLE_DEVICES=1 INDUCTOR_GPU_BENCH_LOCK=1 \
  TORCHINDUCTOR_SCATTER_ADD_REDUCE_ELIMINATION={1|0} \
  TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache_sqfix_{on|off} \
  TORCH_LOGS=output_code \
  python scripts/e2e_squeezenet_scatter_validation.py \
      --mode full_graph --label fix_on --out /tmp/sq_e2e_fix_on.json 2> /tmp/sq_e2e_fix_on.stderr.log

Kernel count: grep -c "def triton" on the stderr log.

NOTE on inputs: the sweep skips this graph as unsafe_integer_input because the
int8 maxpool-offset inputs (getitem_1/3/5) have no graph-inferred constraints.
They are _low_memory_max_pool offsets for a 3x3 window and MUST be in [0, 9);
the sidecar default (index high=512, clamped to int8) would produce
out-of-bounds scatter indices. We override those generators here.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch._inductor.config as inductor_config  # noqa: E402
import torch._inductor.metrics as inductor_metrics  # noqa: E402
from torch._dynamo.utils import counters  # noqa: E402

from oracle_harness import _capture_graph, _gpu_exclusive_lock, _time_graph  # noqa: E402

FULL_GRAPH = REPO / "repros/models/torchbench/train/squeezenet1_1/full_graph_001.py"
CANONICAL_REPRO = REPO / "repros/canonical/sum_sum_3219a09ab96a/repro.py"

# int8 maxpool window offsets: 3x3 kernel -> valid range [0, 9)
MAXPOOL_OFFSET_INPUTS = {"getitem_1", "getitem_3", "getitem_5"}

N_ROUNDS = 5


def _load_full_graph_workload():
    from full_graph_harness import (
        instantiate_full_graph,
        load_full_graph_definition,
        make_inputs_from_full_graph_specs,
    )

    definition = load_full_graph_definition(FULL_GRAPH)
    specs = []
    for spec in definition.input_specs:
        spec = dict(spec)
        if spec.get("name") in MAXPOOL_OFFSET_INPUTS:
            # _generation_spec checks "gen" before "generator": override both.
            spec.pop("gen", None)
            spec["generator"] = {"kind": "index", "low": 0, "high": 9}
        specs.append(spec)
    instance = instantiate_full_graph(definition)
    inputs = make_inputs_from_full_graph_specs(specs)
    return instance, inputs


def _load_canonical_repro_workload():
    import importlib.util
    import math

    spec = importlib.util.spec_from_file_location("repro", CANONICAL_REPRO)
    mod = importlib.util.module_from_spec(spec)
    mod.device = torch.device
    mod.inf = math.inf
    mod.nan = math.nan
    spec.loader.exec_module(mod)
    return mod.Repro(), mod.make_inputs()


def _flatten_tensors(out):
    if torch.is_tensor(out):
        return [out]
    result = []
    if isinstance(out, (tuple, list)):
        for item in out:
            result.extend(_flatten_tensors(item))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full_graph", "repro"], required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    # Surface the pass's chain-detection / skip decisions.
    logging.getLogger("torch._inductor.fx_passes.scatter_reduce_fusion").setLevel(
        logging.DEBUG
    )

    torch.manual_seed(0)

    flag = getattr(inductor_config, "scatter_add_reduce_elimination", None)
    print(
        f"[{args.label}] scatter_add_reduce_elimination={flag} "
        f"env={os.environ.get('TORCHINDUCTOR_SCATTER_ADD_REDUCE_ELIMINATION')!r} "
        f"cache_dir={os.environ.get('TORCHINDUCTOR_CACHE_DIR')!r}",
        file=sys.stderr,
    )

    if args.mode == "full_graph":
        instance, inputs = _load_full_graph_workload()
    else:
        instance, inputs = _load_canonical_repro_workload()

    inductor_config.coordinate_descent_tuning = True

    inductor_metrics.reset()
    counters.clear()
    torch._dynamo.reset()
    compiled = torch.compile(instance)

    with torch.no_grad():
        compiled_out = compiled(*inputs)
        torch.cuda.synchronize()
    n_kernels = inductor_metrics.generated_kernel_count
    pass_applied = counters["inductor"].get("scatter_add_gather_reduce_applied", 0)

    # Numerical sanity vs eager (worst relative error across tensor outputs).
    with torch.no_grad():
        eager_out = instance(*inputs)
        torch.cuda.synchronize()
    worst_rel = 0.0
    compiled_tensors = _flatten_tensors(compiled_out)
    eager_tensors = _flatten_tensors(eager_out)
    for a, b in zip(compiled_tensors, eager_tensors):
        denom = b.abs().max().item() or 1.0
        rel = (a - b).abs().max().item() / denom
        worst_rel = max(worst_rel, rel)
    del eager_out, eager_tensors, compiled_out, compiled_tensors
    torch.cuda.empty_cache()

    # Extra warmup before CUDAGraph capture.
    with torch.no_grad():
        for _ in range(3):
            compiled(*inputs)
        torch.cuda.synchronize()

    graph = _capture_graph(lambda: compiled(*inputs))

    with _gpu_exclusive_lock(label=f"sq_e2e_{args.label}"):
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        times_us = [_time_graph(graph, warmup=3, rep=20) for _ in range(N_ROUNDS)]

    result = {
        "mode": args.mode,
        "label": args.label,
        "scatter_add_reduce_elimination": flag,
        "compiled_us_min": min(times_us),
        "compiled_us_rounds": times_us,
        "n_kernels": n_kernels,
        "scatter_add_gather_reduce_applied": pass_applied,
        "worst_rel_error_vs_eager": worst_rel,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
