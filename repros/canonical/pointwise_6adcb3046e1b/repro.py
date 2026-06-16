"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_infer
Pattern hash: 6adcb3046e1b
Shape hash: 21720c2b
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
    def forward(self, arg0_1: "bf16[512, 1280, 1, 1]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "f32[512, 1280, 1, 1]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        add: "f32[512, 1280, 1, 1]" = torch.ops.aten.add.Tensor(convert_element_type, 3)
        clamp_min: "f32[512, 1280, 1, 1]" = torch.ops.aten.clamp_min.default(add, 0);  add = None
        clamp_max: "f32[512, 1280, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min, 6);  clamp_min = None
        mul: "f32[512, 1280, 1, 1]" = torch.ops.aten.mul.Tensor(convert_element_type, clamp_max);  convert_element_type = clamp_max = None
        div: "f32[512, 1280, 1, 1]" = torch.ops.aten.div.Tensor(mul, 6);  mul = None
        convert_element_type_1: "bf16[512, 1280, 1, 1]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        view: "bf16[512, 1280]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_0);  convert_element_type_1 = _shape_param_0 = None
        return view



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
