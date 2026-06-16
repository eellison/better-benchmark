"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_train
Pattern hash: 2d74760b80f4
Shape hash: adbc9760
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
    def forward(self, arg0_1: "bf16[128, 512, 24, 24]", arg1_1: "bf16[128, 512, 1, 1]", arg2_1: "bf16[128, 512, 24, 24]", arg3_1: "f32[]", arg4_1: "bf16[128, 512, 24, 24]", arg5_1: "bf16[128, 512, 24, 24]"):
        # No stacktrace found for following nodes
        mul: "bf16[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(arg0_1, 0.9805806756909201);  arg0_1 = None
        mul_1: "bf16[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul, 1.7015043497085571);  mul = None
        convert_element_type: "f32[128, 512, 24, 24]" = torch.ops.prims.convert_element_type.default(mul_1, torch.float32);  mul_1 = None
        sigmoid: "bf16[128, 512, 1, 1]" = torch.ops.aten.sigmoid.default(arg1_1);  arg1_1 = None
        mul_2: "bf16[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(arg2_1, sigmoid)
        mul_3: "bf16[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_2, 2.0);  mul_2 = None
        mul_4: "bf16[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_3, arg3_1)
        mul_5: "bf16[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_4, 0.2);  mul_4 = None
        add: "bf16[128, 512, 24, 24]" = torch.ops.aten.add.Tensor(mul_5, arg4_1);  mul_5 = arg4_1 = None
        convert_element_type_1: "f32[128, 512, 24, 24]" = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        mul_6: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 0.7071067811865476)
        erf: "f32[128, 512, 24, 24]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
        add_1: "f32[128, 512, 24, 24]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_7: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_1, 0.5);  add_1 = None
        mul_8: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(convert_element_type_1, convert_element_type_1)
        mul_9: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_8, -0.5);  mul_8 = None
        exp: "f32[128, 512, 24, 24]" = torch.ops.aten.exp.default(mul_9);  mul_9 = None
        mul_10: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_11: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(convert_element_type_1, mul_10);  convert_element_type_1 = mul_10 = None
        add_2: "f32[128, 512, 24, 24]" = torch.ops.aten.add.Tensor(mul_7, mul_11);  mul_7 = mul_11 = None
        mul_12: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(convert_element_type, add_2);  convert_element_type = add_2 = None
        convert_element_type_2: "bf16[128, 512, 24, 24]" = torch.ops.prims.convert_element_type.default(mul_12, torch.bfloat16);  mul_12 = None
        add_3: "bf16[128, 512, 24, 24]" = torch.ops.aten.add.Tensor(arg5_1, convert_element_type_2);  arg5_1 = convert_element_type_2 = None
        mul_13: "bf16[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_3, 0.2)
        mul_14: "bf16[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_13, mul_3);  mul_3 = None
        mul_15: "bf16[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_13, arg3_1);  mul_13 = arg3_1 = None
        sum_1: "f32[]" = torch.ops.aten.sum.default(mul_14, dtype = torch.float32);  mul_14 = None
        mul_16: "bf16[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_15, 2.0);  mul_15 = None
        mul_17: "bf16[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_16, arg2_1);  arg2_1 = None
        sum_2: "f32[128, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_17, [2, 3], True, dtype = torch.float32);  mul_17 = None
        convert_element_type_3: "bf16[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(sum_2, torch.bfloat16);  sum_2 = None
        convert_element_type_4: "f32[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        convert_element_type_5: "f32[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(sigmoid, torch.float32)
        sub: "f32[128, 512, 1, 1]" = torch.ops.aten.sub.Tensor(1, convert_element_type_5)
        mul_18: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_5, sub);  convert_element_type_5 = sub = None
        mul_19: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_4, mul_18);  convert_element_type_4 = mul_18 = None
        convert_element_type_6: "bf16[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(mul_19, torch.bfloat16);  mul_19 = None
        sum_3: "bf16[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_6, [0, 2, 3])
        convert_element_type_7: "f32[512]" = torch.ops.prims.convert_element_type.default(sum_3, torch.float32);  sum_3 = None
        sum_4: "bf16[512]" = torch.ops.aten.sum.dim_IntList(add_3, [0, 2, 3])
        convert_element_type_8: "f32[512]" = torch.ops.prims.convert_element_type.default(sum_4, torch.float32);  sum_4 = None
        return (sigmoid, add_3, sum_1, mul_16, convert_element_type_6, convert_element_type_7, convert_element_type_8)



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
