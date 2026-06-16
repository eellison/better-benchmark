"""
Standalone repro captured via capture_hook.
Label: hf_OPTForCausalLM_train
Pattern hash: 31ea6dc807dd
Shape hash: 81e4a7b8
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
    def forward(self, arg0_1: "bf16[8192, 768]", arg1_1: "f32[768]", arg2_1: "bf16[8192, 768]", arg3_1: "b8[4, 2048, 768]", arg4_1: "f32[4, 2048, 768]", arg5_1: "f32[8192, 1]", arg6_1: "f32[8192, 1]", arg7_1: "f32[8192, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        convert_element_type: "f32[8192, 768]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        mul: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, arg1_1);  arg1_1 = None
        mul_1: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        sum_1: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(mul, [1], True)
        view: "bf16[4, 2048, 768]" = torch.ops.aten.view.default(arg2_1, _shape_param_0);  arg2_1 = _shape_param_0 = None
        mul_2: "bf16[4, 2048, 768]" = torch.ops.aten.mul.Tensor(arg3_1, view);  view = None
        mul_3: "bf16[4, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None
        add: "f32[4, 2048, 768]" = torch.ops.aten.add.Tensor(arg4_1, mul_3);  arg4_1 = mul_3 = None
        view_1: "f32[8192, 768]" = torch.ops.aten.view.default(add, _shape_param_1);  add = _shape_param_1 = None
        sub: "f32[8192, 768]" = torch.ops.aten.sub.Tensor(view_1, arg5_1);  view_1 = arg5_1 = None
        mul_4: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(sub, arg6_1);  sub = None
        mul_5: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(mul, mul_4);  mul = None
        sum_2: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(mul_5, [1], True);  mul_5 = None
        mul_6: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(mul_4, sum_2);  sum_2 = None
        sub_1: "f32[8192, 768]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_2: "f32[8192, 768]" = torch.ops.aten.sub.Tensor(sub_1, mul_6);  sub_1 = mul_6 = None
        div: "f32[8192, 1]" = torch.ops.aten.div.Tensor(arg6_1, 768);  arg6_1 = None
        mul_7: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_8: "f32[8192, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, mul_4);  mul_4 = None
        sum_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_8, [0]);  mul_8 = None
        sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type, [0]);  convert_element_type = None
        add_1: "f32[8192, 768]" = torch.ops.aten.add.Tensor(arg7_1, mul_7);  arg7_1 = mul_7 = None
        view_2: "f32[4, 2048, 768]" = torch.ops.aten.view.default(add_1, _shape_param_2);  add_1 = _shape_param_2 = None
        convert_element_type_1: "bf16[4, 2048, 768]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16)
        convert_element_type_2: "bf16[4, 2048, 768]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        mul_9: "bf16[4, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_10: "bf16[4, 2048, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, mul_9);  convert_element_type_1 = mul_9 = None
        view_3: "bf16[8192, 768]" = torch.ops.aten.view.default(mul_10, _shape_param_3);  mul_10 = _shape_param_3 = None
        permute: "bf16[768, 8192]" = torch.ops.aten.permute.default(view_3, [1, 0])
        sum_5: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_3, [0], True, dtype = torch.float32)
        view_4: "f32[768]" = torch.ops.aten.view.default(sum_5, _shape_param_4);  sum_5 = _shape_param_4 = None
        convert_element_type_3: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_4, torch.bfloat16);  view_4 = None
        convert_element_type_4: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        return (sum_3, sum_4, view_2, view_3, permute, convert_element_type_4)



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
