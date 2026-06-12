"""
Standalone repro captured via capture_hook.
Label: timm_repvgg_a2_train
Pattern hash: fb3a1658dadb
Shape hash: 7a67de76
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
    def forward(self, arg0_1: "bf16[128, 64, 112, 112]", arg1_1: "bf16[128, 64, 112, 112]", arg2_1: "bf16[128, 64, 112, 112]", arg3_1: "bf16[]", arg4_1: "bf16[128, 64, 112, 112]", arg5_1: "f32[1, 64, 1, 1]", arg6_1: "f32[64]", arg7_1: "f32[64]", arg8_1: "bf16[128, 64, 112, 112]", arg9_1: "f32[1, 64, 1, 1]", arg10_1: "f32[64]", arg11_1: "f32[64]"):
        # No stacktrace found for following nodes
        add: "bf16[128, 64, 112, 112]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        le: "b8[128, 64, 112, 112]" = torch.ops.aten.le.Scalar(arg2_1, 0);  arg2_1 = None
        where: "bf16[128, 64, 112, 112]" = torch.ops.aten.where.self(le, arg3_1, add);  le = arg3_1 = add = None
        convert_element_type: "f32[128, 64, 112, 112]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        sum_1: "f32[64]" = torch.ops.aten.sum.dim_IntList(convert_element_type, [0, 2, 3])
        convert_element_type_1: "f32[128, 64, 112, 112]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        sub: "f32[128, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg5_1);  convert_element_type_1 = arg5_1 = None
        mul: "f32[128, 64, 112, 112]" = torch.ops.aten.mul.Tensor(convert_element_type, sub)
        sum_2: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul, [0, 2, 3]);  mul = None
        mul_1: "f32[64]" = torch.ops.aten.mul.Tensor(sum_1, 6.228077168367346e-07)
        unsqueeze: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1, 0);  mul_1 = None
        unsqueeze_1: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        mul_2: "f32[64]" = torch.ops.aten.mul.Tensor(sum_2, 6.228077168367346e-07)
        mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(arg6_1, arg6_1)
        mul_4: "f32[64]" = torch.ops.aten.mul.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        unsqueeze_3: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_4, 0);  mul_4 = None
        unsqueeze_4: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        unsqueeze_5: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        mul_5: "f32[64]" = torch.ops.aten.mul.Tensor(arg6_1, arg7_1);  arg7_1 = None
        unsqueeze_6: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_5, 0);  mul_5 = None
        unsqueeze_7: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
        unsqueeze_8: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        mul_6: "f32[128, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_5);  sub = unsqueeze_5 = None
        sub_1: "f32[128, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convert_element_type, mul_6);  mul_6 = None
        sub_2: "f32[128, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_1, unsqueeze_2);  sub_1 = unsqueeze_2 = None
        mul_7: "f32[128, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_8);  sub_2 = unsqueeze_8 = None
        mul_8: "f32[64]" = torch.ops.aten.mul.Tensor(sum_2, arg6_1);  sum_2 = arg6_1 = None
        convert_element_type_2: "bf16[128, 64, 112, 112]" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None
        sum_3: "f32[64]" = torch.ops.aten.sum.dim_IntList(convert_element_type, [0, 2, 3])
        convert_element_type_3: "f32[128, 64, 112, 112]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.float32);  arg8_1 = None
        sub_3: "f32[128, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convert_element_type_3, arg9_1);  convert_element_type_3 = arg9_1 = None
        mul_9: "f32[128, 64, 112, 112]" = torch.ops.aten.mul.Tensor(convert_element_type, sub_3)
        sum_4: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_9, [0, 2, 3]);  mul_9 = None
        mul_10: "f32[64]" = torch.ops.aten.mul.Tensor(sum_3, 6.228077168367346e-07)
        unsqueeze_9: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_10, 0);  mul_10 = None
        unsqueeze_10: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 2);  unsqueeze_9 = None
        unsqueeze_11: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 3);  unsqueeze_10 = None
        mul_11: "f32[64]" = torch.ops.aten.mul.Tensor(sum_4, 6.228077168367346e-07)
        mul_12: "f32[64]" = torch.ops.aten.mul.Tensor(arg10_1, arg10_1)
        mul_13: "f32[64]" = torch.ops.aten.mul.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_12: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_13, 0);  mul_13 = None
        unsqueeze_13: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 2);  unsqueeze_12 = None
        unsqueeze_14: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 3);  unsqueeze_13 = None
        mul_14: "f32[64]" = torch.ops.aten.mul.Tensor(arg10_1, arg11_1);  arg11_1 = None
        unsqueeze_15: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_14, 0);  mul_14 = None
        unsqueeze_16: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_15, 2);  unsqueeze_15 = None
        unsqueeze_17: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, 3);  unsqueeze_16 = None
        mul_15: "f32[128, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_14);  sub_3 = unsqueeze_14 = None
        sub_4: "f32[128, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convert_element_type, mul_15);  convert_element_type = mul_15 = None
        sub_5: "f32[128, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_4, unsqueeze_11);  sub_4 = unsqueeze_11 = None
        mul_16: "f32[128, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_17);  sub_5 = unsqueeze_17 = None
        mul_17: "f32[64]" = torch.ops.aten.mul.Tensor(sum_4, arg10_1);  sum_4 = arg10_1 = None
        convert_element_type_4: "bf16[128, 64, 112, 112]" = torch.ops.prims.convert_element_type.default(mul_16, torch.bfloat16);  mul_16 = None
        return (sum_1, mul_8, convert_element_type_2, sum_3, mul_17, convert_element_type_4)



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
