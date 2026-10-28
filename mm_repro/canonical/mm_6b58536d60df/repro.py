"""
Standalone repro captured via capture_hook.
Label: repros/models/timm/train/mobilevit_s/full_graph_001.py
Pattern hash: 6b58536d60df
Shape hash: c1fadf29
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
    def forward(self, arg0_1: "bf16[8192, 240]", arg1_1: "bf16[240, 480]", arg2_1: "bf16[8192, 480]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        mm: "bf16[8192, 480]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[512, 16, 480]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[512, 16, 480]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        view_1: "bf16[512, 16, 480]" = torch.ops.aten.view.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        convert_element_type_1: "f32[512, 16, 480]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        sigmoid: "f32[512, 16, 480]" = torch.ops.aten.sigmoid.default(convert_element_type_1)
        mul: "f32[512, 16, 480]" = torch.ops.aten.mul.Tensor(convert_element_type, sigmoid);  convert_element_type = None
        sub: "f32[512, 16, 480]" = torch.ops.aten.sub.Tensor(1, sigmoid);  sigmoid = None
        mul_1: "f32[512, 16, 480]" = torch.ops.aten.mul.Tensor(convert_element_type_1, sub);  convert_element_type_1 = sub = None
        add: "f32[512, 16, 480]" = torch.ops.aten.add.Tensor(mul_1, 1);  mul_1 = None
        mul_2: "f32[512, 16, 480]" = torch.ops.aten.mul.Tensor(mul, add);  mul = add = None
        convert_element_type_2: "bf16[512, 16, 480]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        view_2: "bf16[8192, 480]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_2);  convert_element_type_2 = _shape_param_2 = None
        permute: "bf16[480, 8192]" = torch.ops.aten.permute.default(view_2, [1, 0])
        return (view_2, permute)



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
