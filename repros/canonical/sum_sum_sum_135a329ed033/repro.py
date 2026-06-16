"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train
Pattern hash: 135a329ed033
Shape hash: 5a443972
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
    def forward(self, arg0_1: "bf16[8192, 768]", arg1_1: "f32[8, 1024, 768]", arg2_1: "f32[768]", arg3_1: "f32[8, 1024, 768]", arg4_1: "f32[8, 1024, 1]", arg5_1: "b8[8, 1024, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        add: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(arg1_1, convert_element_type);  arg1_1 = convert_element_type = None
        mul: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add, arg2_1);  arg2_1 = None
        mul_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        sum_1: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        mul_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = None
        sum_2: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True);  mul_2 = None
        mul_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg3_1, sum_2);  sum_2 = None
        sub: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_1: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg4_1, sub_1);  arg4_1 = sub_1 = None
        mul_5: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add, arg3_1);  arg3_1 = None
        sum_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 1]);  mul_5 = None
        sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(add, [0, 1]);  add = None
        convert_element_type_1: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16)
        convert_element_type_2: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        mul_6: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_7: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, mul_6);  convert_element_type_1 = mul_6 = None
        sum_5: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_7, [0, 1], True, dtype = torch.float32)
        view_1: "f32[768]" = torch.ops.aten.view.default(sum_5, _shape_param_1);  sum_5 = _shape_param_1 = None
        convert_element_type_3: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        view_2: "bf16[8192, 768]" = torch.ops.aten.view.default(mul_7, _shape_param_2);  mul_7 = _shape_param_2 = None
        permute: "bf16[768, 8192]" = torch.ops.aten.permute.default(view_2, [1, 0])
        convert_element_type_4: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        return (mul_4, sum_3, sum_4, view_2, permute, convert_element_type_4)



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
