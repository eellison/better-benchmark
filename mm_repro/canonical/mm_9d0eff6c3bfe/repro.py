"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/infer/T5ForConditionalGeneration/full_graph_000.py
Pattern hash: 9d0eff6c3bfe
Shape hash: aad97ac9
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[2048, 512]", arg1_1: "bf16[8192, 512]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        permute: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        mm: "bf16[8192, 2048]" = torch.ops.aten.mm.default(arg1_1, permute);  arg1_1 = permute = None
        view: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        relu: "bf16[8, 1024, 2048]" = torch.ops.aten.relu.default(view);  view = None
        view_1: "bf16[8192, 2048]" = torch.ops.aten.view.default(relu, _shape_param_1);  relu = _shape_param_1 = None
        return view_1



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
