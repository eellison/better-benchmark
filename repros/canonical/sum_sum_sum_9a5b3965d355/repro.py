"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_train
Pattern hash: 9a5b3965d355
Shape hash: 5afc2635
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
    def forward(self, arg0_1: "bf16[128, 144, 32, 32]", arg1_1: "f32[144]", arg2_1: "bf16[512, 256, 144]", arg3_1: "f32[512, 256, 1]", arg4_1: "f32[512, 256, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        convert_element_type: "f32[128, 144, 32, 32]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        clone: "f32[128, 144, 32, 32]" = torch.ops.aten.clone.default(convert_element_type, memory_format = torch.contiguous_format);  convert_element_type = None
        view: "f32[294912, 2, 16, 2]" = torch.ops.aten.view.default(clone, _shape_param_0);  clone = _shape_param_0 = None
        permute: "f32[294912, 16, 2, 2]" = torch.ops.aten.permute.default(view, [0, 2, 1, 3]);  view = None
        clone_1: "f32[294912, 16, 2, 2]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_1: "f32[128, 144, 256, 4]" = torch.ops.aten.view.default(clone_1, _shape_param_1);  clone_1 = _shape_param_1 = None
        permute_1: "f32[128, 4, 256, 144]" = torch.ops.aten.permute.default(view_1, [0, 3, 2, 1]);  view_1 = None
        clone_2: "f32[128, 4, 256, 144]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_2: "f32[512, 256, 144]" = torch.ops.aten.view.default(clone_2, _shape_param_2);  clone_2 = _shape_param_2 = None
        mul: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(view_2, arg1_1);  arg1_1 = None
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
        mul_6: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(view_2, mul_2);  mul_2 = None
        sum_3: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_6, [0, 1]);  mul_6 = None
        sum_4: "f32[144]" = torch.ops.aten.sum.dim_IntList(view_2, [0, 1]);  view_2 = None
        convert_element_type_2: "bf16[512, 256, 144]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        view_3: "bf16[131072, 144]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_3);  _shape_param_3 = None
        permute_2: "bf16[144, 131072]" = torch.ops.aten.permute.default(view_3, [1, 0])
        sum_5: "f32[1, 144]" = torch.ops.aten.sum.dim_IntList(view_3, [0], True, dtype = torch.float32)
        view_4: "f32[144]" = torch.ops.aten.view.default(sum_5, _shape_param_4);  sum_5 = _shape_param_4 = None
        convert_element_type_3: "bf16[144]" = torch.ops.prims.convert_element_type.default(view_4, torch.bfloat16);  view_4 = None
        convert_element_type_4: "f32[144]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        return (sum_3, sum_4, convert_element_type_2, view_3, permute_2, convert_element_type_4)



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
