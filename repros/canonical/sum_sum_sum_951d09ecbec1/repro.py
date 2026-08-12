"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_train
Pattern hash: 951d09ecbec1
Shape hash: 30b03cad
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
    def forward(self, arg0_1: "bf16[131072, 144]", arg1_1: "f32[144]", arg2_1: "bf16[512, 256, 144]", arg3_1: "f32[512, 256, 1]", arg4_1: "f32[512, 256, 1]", arg5_1: "bf16[512, 256, 144]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view: "bf16[512, 256, 144]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[512, 256, 144]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(convert_element_type, arg1_1);  arg1_1 = None
        mul_1: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(mul, 144)
        sum_1: "f32[512, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        convert_element_type_1: "f32[512, 256, 144]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        sub: "f32[512, 256, 144]" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg3_1);  convert_element_type_1 = arg3_1 = None
        mul_2: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(sub, arg4_1);  sub = None
        mul_3: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(mul, mul_2);  mul = None
        sum_2: "f32[512, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_3, [2], True);  mul_3 = None
        mul_4: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(mul_2, sum_2);  sum_2 = None
        sub_1: "f32[512, 256, 144]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_2: "f32[512, 256, 144]" = torch.ops.aten.sub.Tensor(sub_1, mul_4);  sub_1 = mul_4 = None
        div: "f32[512, 256, 1]" = torch.ops.aten.div.Tensor(arg4_1, 144);  arg4_1 = None
        mul_5: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_6: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(convert_element_type, mul_2);  mul_2 = None
        sum_3: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_6, [0, 1]);  mul_6 = None
        sum_4: "f32[144]" = torch.ops.aten.sum.dim_IntList(convert_element_type, [0, 1]);  convert_element_type = None
        convert_element_type_2: "bf16[512, 256, 144]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        add: "bf16[512, 256, 144]" = torch.ops.aten.add.Tensor(arg5_1, convert_element_type_2);  arg5_1 = convert_element_type_2 = None
        view_1: "bf16[131072, 144]" = torch.ops.aten.view.default(add, _shape_param_1);  _shape_param_1 = None
        permute: "bf16[144, 131072]" = torch.ops.aten.permute.default(view_1, [1, 0])
        sum_5: "f32[1, 144]" = torch.ops.aten.sum.dim_IntList(view_1, [0], True, dtype = torch.float32)
        view_2: "f32[144]" = torch.ops.aten.view.default(sum_5, _shape_param_2);  sum_5 = _shape_param_2 = None
        convert_element_type_3: "bf16[144]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        convert_element_type_4: "f32[144]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        return (sum_3, sum_4, add, view_1, permute, convert_element_type_4)



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
