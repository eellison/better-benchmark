"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/GoogleFnet/full_graph_001.py
Pattern hash: 44940d51a61a
Shape hash: b6a593c0
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
    def forward(self, arg0_1: "f32[768, 768]", arg1_1: "f32[16384, 768]", arg2_1: "f32[768]", arg3_1: "f32[32, 512, 768]", _shape_param_0):
        # No stacktrace found for following nodes
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm: "f32[16384, 768]" = torch.ops.aten.mm.default(arg1_1, permute_1);  arg1_1 = permute_1 = None
        view: "f32[32, 512, 768]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view, arg2_1);  arg2_1 = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg3_1)
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view, arg3_1);  view = arg3_1 = None
        return (mul, mul_1, mul_2, mul_3)



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
