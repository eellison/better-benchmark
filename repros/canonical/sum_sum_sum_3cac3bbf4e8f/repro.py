"""
Standalone repro captured via capture_hook.
Label: genai_LayerNormBackward_static
Pattern hash: 3cac3bbf4e8f
Shape hash: 4e21884e
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
    def forward(self, arg0_1: "f32[1152000, 1]", arg1_1: "bf16[]", arg2_1: "f32[512]", arg3_1: "bf16[1152000, 512]", arg4_1: "f32[1152000, 1]", _shape_param_0):
        # No stacktrace found for following nodes
        div: "f32[1152000, 1]" = torch.ops.aten.div.Tensor(arg0_1, 512)
        expand: "bf16[1152000, 512]" = torch.ops.aten.expand.default(arg1_1, _shape_param_0);  arg1_1 = _shape_param_0 = None
        convert_element_type: "f32[1152000, 512]" = torch.ops.prims.convert_element_type.default(expand, torch.float32);  expand = None
        mul: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(convert_element_type, arg2_1);  arg2_1 = None
        mul_1: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(mul, 512)
        sum_1: "f32[1152000, 1]" = torch.ops.aten.sum.dim_IntList(mul, [1], True)
        sub: "f32[1152000, 512]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        convert_element_type_1: "f32[1152000, 512]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        sub_1: "f32[1152000, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg4_1);  convert_element_type_1 = arg4_1 = None
        mul_2: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(sub_1, arg0_1);  sub_1 = arg0_1 = None
        mul_3: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(mul, mul_2);  mul = None
        sum_2: "f32[1152000, 1]" = torch.ops.aten.sum.dim_IntList(mul_3, [1], True);  mul_3 = None
        mul_4: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(mul_2, sum_2);  sum_2 = None
        sub_2: "f32[1152000, 512]" = torch.ops.aten.sub.Tensor(sub, mul_4);  sub = mul_4 = None
        mul_5: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        convert_element_type_2: "bf16[1152000, 512]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        mul_6: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(convert_element_type, mul_2);  convert_element_type = mul_2 = None
        sum_3: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_6, [0]);  mul_6 = None
        return (convert_element_type_2, sum_3)



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
