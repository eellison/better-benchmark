"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_train
Pattern hash: b37e1065b3a0
Shape hash: 4fd6f7e0
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
    def forward(self, arg0_1: "bf16[128, 512, 28, 28]", arg1_1: "bf16[128, 512, 1, 1]", arg2_1: "bf16[128, 512, 28, 28]", arg3_1: "bf16[128, 512, 28, 28]", arg4_1: "bf16[128, 512, 28, 28]"):
        # No stacktrace found for following nodes
        mul: "bf16[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(arg0_1, 0.9805806756909201);  arg0_1 = None
        convert_element_type: "f32[128, 512, 28, 28]" = torch.ops.prims.convert_element_type.default(mul, torch.float32);  mul = None
        sigmoid: "bf16[128, 512, 1, 1]" = torch.ops.aten.sigmoid.default(arg1_1);  arg1_1 = None
        mul_1: "bf16[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(arg2_1, sigmoid)
        mul_2: "bf16[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1, 2.0);  mul_1 = None
        mul_3: "bf16[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_2, 0.2);  mul_2 = None
        add: "bf16[128, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_3, arg3_1);  mul_3 = arg3_1 = None
        convert_element_type_1: "f32[128, 512, 28, 28]" = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        sigmoid_1: "f32[128, 512, 28, 28]" = torch.ops.aten.sigmoid.default(convert_element_type_1)
        mul_4: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(convert_element_type, sigmoid_1);  convert_element_type = None
        sub: "f32[128, 512, 28, 28]" = torch.ops.aten.sub.Tensor(1, sigmoid_1);  sigmoid_1 = None
        mul_5: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(convert_element_type_1, sub);  convert_element_type_1 = sub = None
        add_1: "f32[128, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_5, 1);  mul_5 = None
        mul_6: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_4, add_1);  mul_4 = add_1 = None
        convert_element_type_2: "bf16[128, 512, 28, 28]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        add_2: "bf16[128, 512, 28, 28]" = torch.ops.aten.add.Tensor(arg4_1, convert_element_type_2);  arg4_1 = convert_element_type_2 = None
        mul_7: "bf16[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(add_2, 0.2)
        mul_8: "bf16[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_7, 2.0);  mul_7 = None
        mul_9: "bf16[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_8, arg2_1);  arg2_1 = None
        sum_1: "f32[128, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_9, [2, 3], True, dtype = torch.float32);  mul_9 = None
        convert_element_type_3: "bf16[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(sum_1, torch.bfloat16);  sum_1 = None
        convert_element_type_4: "f32[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        convert_element_type_5: "f32[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(sigmoid, torch.float32)
        sub_1: "f32[128, 512, 1, 1]" = torch.ops.aten.sub.Tensor(1, convert_element_type_5)
        mul_10: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_5, sub_1);  convert_element_type_5 = sub_1 = None
        mul_11: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_4, mul_10);  convert_element_type_4 = mul_10 = None
        convert_element_type_6: "bf16[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(mul_11, torch.bfloat16);  mul_11 = None
        sum_2: "bf16[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_6, [0, 2, 3])
        convert_element_type_7: "f32[512]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3: "bf16[512]" = torch.ops.aten.sum.dim_IntList(add_2, [0, 2, 3])
        convert_element_type_8: "f32[512]" = torch.ops.prims.convert_element_type.default(sum_3, torch.float32);  sum_3 = None
        return (sigmoid, add_2, mul_8, convert_element_type_6, convert_element_type_7, convert_element_type_8)



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
