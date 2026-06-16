"""
Standalone repro captured via capture_hook.
Label: hf_YituTechConvBert_train
Pattern hash: 56a68a521d26
Shape hash: 730e8332
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
    def forward(self, arg0_1: "bf16[16384, 768]", arg1_1: "f32[32, 512, 768]", arg2_1: "bf16[16384, 768]", arg3_1: "bf16[32, 768, 512]", arg4_1: "bf16[16384, 768]", arg5_1: "bf16[16384, 768]", arg6_1: "f32[768]", arg7_1: "f32[32, 512, 768]", arg8_1: "f32[32, 512, 1]", arg9_1: "b8[32, 512, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        view: "bf16[32, 512, 768]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        add: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(arg1_1, convert_element_type);  arg1_1 = convert_element_type = None
        view_1: "bf16[32, 512, 768]" = torch.ops.aten.view.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        convert_element_type_1: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        add_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add, convert_element_type_1);  add = convert_element_type_1 = None
        convert_element_type_2: "f32[32, 768, 512]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        permute: "f32[32, 512, 768]" = torch.ops.aten.permute.default(convert_element_type_2, [0, 2, 1]);  convert_element_type_2 = None
        add_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_1, permute);  add_1 = permute = None
        view_2: "bf16[32, 512, 768]" = torch.ops.aten.view.default(arg4_1, _shape_param_2);  arg4_1 = _shape_param_2 = None
        convert_element_type_3: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        add_3: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_2, convert_element_type_3);  add_2 = convert_element_type_3 = None
        view_3: "bf16[32, 512, 768]" = torch.ops.aten.view.default(arg5_1, _shape_param_3);  arg5_1 = _shape_param_3 = None
        convert_element_type_4: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        add_4: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_3, convert_element_type_4);  add_3 = convert_element_type_4 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_4, arg6_1);  arg6_1 = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        sum_1: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg7_1);  mul = None
        sum_2: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True);  mul_2 = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg7_1, sum_2);  sum_2 = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg8_1, sub_1);  arg8_1 = sub_1 = None
        mul_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_4, arg7_1);  arg7_1 = None
        sum_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 1]);  mul_5 = None
        sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_4, [0, 1]);  add_4 = None
        convert_element_type_5: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16)
        convert_element_type_6: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.bfloat16);  arg9_1 = None
        mul_6: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1111111111111112);  convert_element_type_6 = None
        mul_7: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_5, mul_6);  convert_element_type_5 = mul_6 = None
        view_4: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_7, _shape_param_4);  mul_7 = _shape_param_4 = None
        permute_1: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_4, [1, 0])
        sum_5: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_4, [0], True, dtype = torch.float32)
        view_5: "f32[768]" = torch.ops.aten.view.default(sum_5, _shape_param_5);  sum_5 = _shape_param_5 = None
        convert_element_type_7: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        convert_element_type_8: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_7, torch.float32);  convert_element_type_7 = None
        return (mul_4, sum_3, sum_4, view_4, permute_1, convert_element_type_8)



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
