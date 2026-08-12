"""
Standalone repro captured via capture_hook.
Label: hf_BartForCausalLM_train
Pattern hash: 00a35e7a9bcb
Shape hash: c040a99e
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
    def forward(self, arg0_1: "f32[8, 1024, 1024]", arg1_1: "f32[1024]", arg2_1: "f32[8, 1024, 1024]", arg3_1: "f32[8, 1024, 1]", arg4_1: "b8[8, 1024, 1024]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        mul: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        mul_1: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul, 1024)
        sum_1: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        mul_2: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = None
        sum_2: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True);  mul_2 = None
        mul_3: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(arg2_1, sum_2);  arg2_1 = sum_2 = None
        sub: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_1: "f32[8, 1024, 1024]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(arg3_1, sub_1);  arg3_1 = sub_1 = None
        convert_element_type: "bf16[8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16)
        convert_element_type_1: "bf16[8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        mul_5: "bf16[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_6: "bf16[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, mul_5);  convert_element_type = mul_5 = None
        view: "bf16[8192, 1024]" = torch.ops.aten.view.default(mul_6, _shape_param_0);  mul_6 = _shape_param_0 = None
        permute: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view, [1, 0])
        sum_3: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view, [0], True, dtype = torch.float32)
        view_1: "f32[1024]" = torch.ops.aten.view.default(sum_3, _shape_param_1);  sum_3 = _shape_param_1 = None
        convert_element_type_2: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        convert_element_type_3: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        return (mul_4, view, permute, convert_element_type_3)



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
