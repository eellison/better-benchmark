"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_train
Pattern hash: ef8c091a67f7
Shape hash: f3ef90ca
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
    def forward(self, arg0_1: "bf16[8192, 1024]", arg1_1: "b8[512, 16, 1024]", arg2_1: "f32[1024]", arg3_1: "f32[512, 16, 1024]", arg4_1: "f32[512, 16, 1]", arg5_1: "b8[512, 16, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        permute: "f32[512, 16, 1024]" = torch.ops.aten.permute.default(convert_element_type, [1, 0, 2]);  convert_element_type = None
        convert_element_type_1: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        mul: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(permute, mul);  permute = mul = None
        clone: "f32[512, 16, 1024]" = torch.ops.aten.clone.default(mul_1, memory_format = torch.contiguous_format);  mul_1 = None
        mul_2: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(clone, arg2_1);  arg2_1 = None
        mul_3: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_2, 1024)
        sum_1: "f32[512, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True)
        mul_4: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_2, arg3_1);  mul_2 = None
        sum_2: "f32[512, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_4, [2], True);  mul_4 = None
        mul_5: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(arg3_1, sum_2);  sum_2 = None
        sub: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(mul_3, sum_1);  mul_3 = sum_1 = None
        sub_1: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(sub, mul_5);  sub = mul_5 = None
        mul_6: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(arg4_1, sub_1);  arg4_1 = sub_1 = None
        mul_7: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(clone, arg3_1);  arg3_1 = None
        sum_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_7, [0, 1]);  mul_7 = None
        sum_4: "f32[1024]" = torch.ops.aten.sum.dim_IntList(clone, [0, 1]);  clone = None
        convert_element_type_2: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16)
        convert_element_type_3: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        mul_8: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
        mul_9: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_2, mul_8);  convert_element_type_2 = mul_8 = None
        view_1: "bf16[8192, 1024]" = torch.ops.aten.view.default(mul_9, _shape_param_1);  mul_9 = _shape_param_1 = None
        permute_1: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_1, [1, 0])
        sum_5: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1, [0], True, dtype = torch.float32)
        view_2: "f32[1024]" = torch.ops.aten.view.default(sum_5, _shape_param_2);  sum_5 = _shape_param_2 = None
        convert_element_type_4: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        convert_element_type_5: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32);  convert_element_type_4 = None
        return (mul_6, sum_3, sum_4, view_1, permute_1, convert_element_type_5)



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
