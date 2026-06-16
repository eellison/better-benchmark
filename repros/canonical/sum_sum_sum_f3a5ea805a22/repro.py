"""
Standalone repro captured via capture_hook.
Label: hf_BlenderbotForCausalLM_train
Pattern hash: f3a5ea805a22
Shape hash: 1dd4e44b
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
    def forward(self, arg0_1: "bf16[4096, 2560]", arg1_1: "f32[2560]", arg2_1: "bf16[4096, 2560]", arg3_1: "b8[32, 128, 2560]", arg4_1: "f32[32, 128, 2560]", arg5_1: "f32[32, 128, 1]", arg6_1: "f32[32, 128, 1]", arg7_1: "f32[32, 128, 2560]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view: "bf16[32, 128, 2560]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[32, 128, 2560]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type, arg1_1);  arg1_1 = None
        mul_1: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul, 2560)
        sum_1: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        view_1: "bf16[32, 128, 2560]" = torch.ops.aten.view.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        mul_2: "bf16[32, 128, 2560]" = torch.ops.aten.mul.Tensor(arg3_1, view_1);  view_1 = None
        mul_3: "bf16[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None
        add: "f32[32, 128, 2560]" = torch.ops.aten.add.Tensor(arg4_1, mul_3);  arg4_1 = mul_3 = None
        sub: "f32[32, 128, 2560]" = torch.ops.aten.sub.Tensor(add, arg5_1);  add = arg5_1 = None
        mul_4: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(sub, arg6_1);  sub = None
        mul_5: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul, mul_4);  mul = None
        sum_2: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_5, [2], True);  mul_5 = None
        mul_6: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_4, sum_2);  sum_2 = None
        sub_1: "f32[32, 128, 2560]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_2: "f32[32, 128, 2560]" = torch.ops.aten.sub.Tensor(sub_1, mul_6);  sub_1 = mul_6 = None
        div: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(arg6_1, 2560);  arg6_1 = None
        mul_7: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_8: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type, mul_4);  mul_4 = None
        sum_3: "f32[2560]" = torch.ops.aten.sum.dim_IntList(mul_8, [0, 1]);  mul_8 = None
        sum_4: "f32[2560]" = torch.ops.aten.sum.dim_IntList(convert_element_type, [0, 1]);  convert_element_type = None
        add_1: "f32[32, 128, 2560]" = torch.ops.aten.add.Tensor(arg7_1, mul_7);  arg7_1 = mul_7 = None
        convert_element_type_1: "bf16[32, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16)
        convert_element_type_2: "bf16[32, 128, 2560]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        mul_9: "bf16[32, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_10: "bf16[32, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_1, mul_9);  convert_element_type_1 = mul_9 = None
        view_2: "bf16[4096, 2560]" = torch.ops.aten.view.default(mul_10, _shape_param_2);  mul_10 = _shape_param_2 = None
        permute: "bf16[2560, 4096]" = torch.ops.aten.permute.default(view_2, [1, 0])
        sum_5: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_2, [0], True, dtype = torch.float32)
        view_3: "f32[2560]" = torch.ops.aten.view.default(sum_5, _shape_param_3);  sum_5 = _shape_param_3 = None
        convert_element_type_3: "bf16[2560]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        convert_element_type_4: "f32[2560]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        return (sum_3, sum_4, add_1, view_2, permute, convert_element_type_4)



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
