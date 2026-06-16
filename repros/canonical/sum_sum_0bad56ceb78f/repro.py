"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv2_100_train
Pattern hash: 0bad56ceb78f
Shape hash: 8693ecd8
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
    def forward(self, arg0_1: "bf16[128, 32, 112, 112]", arg1_1: "f32[1, 32, 1, 1]", arg2_1: "f32[1, 32, 1, 1]", arg3_1: "f32[32]", arg4_1: "f32[32]", arg5_1: "bf16[]", arg6_1: "bf16[128, 32, 112, 112]"):
        # No stacktrace found for following nodes
        sub: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(arg0_1, arg1_1)
        mul: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub, arg2_1);  sub = None
        unsqueeze: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1)
        unsqueeze_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_1: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add: "f32[128, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_3);  mul_1 = unsqueeze_3 = None
        convert_element_type: "bf16[128, 32, 112, 112]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        le: "b8[128, 32, 112, 112]" = torch.ops.aten.le.Scalar(convert_element_type, 0.0)
        ge: "b8[128, 32, 112, 112]" = torch.ops.aten.ge.Scalar(convert_element_type, 6.0);  convert_element_type = None
        bitwise_or: "b8[128, 32, 112, 112]" = torch.ops.aten.bitwise_or.Tensor(le, ge);  le = ge = None
        where: "bf16[128, 32, 112, 112]" = torch.ops.aten.where.self(bitwise_or, arg5_1, arg6_1);  bitwise_or = arg5_1 = arg6_1 = None
        convert_element_type_1: "f32[128, 32, 112, 112]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        squeeze: "f32[32]" = torch.ops.aten.squeeze.dims(arg1_1, [0, 2, 3]);  arg1_1 = None
        unsqueeze_4: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_5: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None
        unsqueeze_6: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3);  unsqueeze_5 = None
        sum_1: "f32[32]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1, [0, 2, 3])
        convert_element_type_2: "f32[128, 32, 112, 112]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        sub_1: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convert_element_type_2, unsqueeze_6);  convert_element_type_2 = unsqueeze_6 = None
        mul_2: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(convert_element_type_1, sub_1)
        sum_2: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_2, [0, 2, 3]);  mul_2 = None
        mul_3: "f32[32]" = torch.ops.aten.mul.Tensor(sum_1, 6.228077168367346e-07)
        unsqueeze_7: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_3, 0);  mul_3 = None
        unsqueeze_8: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 2);  unsqueeze_7 = None
        unsqueeze_9: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 3);  unsqueeze_8 = None
        mul_4: "f32[32]" = torch.ops.aten.mul.Tensor(sum_2, 6.228077168367346e-07)
        squeeze_1: "f32[32]" = torch.ops.aten.squeeze.dims(arg2_1, [0, 2, 3]);  arg2_1 = None
        mul_5: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_6: "f32[32]" = torch.ops.aten.mul.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze_10: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_6, 0);  mul_6 = None
        unsqueeze_11: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None
        unsqueeze_12: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 3);  unsqueeze_11 = None
        mul_7: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, arg3_1);  arg3_1 = None
        unsqueeze_13: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_7, 0);  mul_7 = None
        unsqueeze_14: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 2);  unsqueeze_13 = None
        unsqueeze_15: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 3);  unsqueeze_14 = None
        mul_8: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_12);  sub_1 = unsqueeze_12 = None
        sub_2: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convert_element_type_1, mul_8);  convert_element_type_1 = mul_8 = None
        sub_3: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_2, unsqueeze_9);  sub_2 = unsqueeze_9 = None
        mul_9: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_15);  sub_3 = unsqueeze_15 = None
        mul_10: "f32[32]" = torch.ops.aten.mul.Tensor(sum_2, squeeze_1);  sum_2 = squeeze_1 = None
        convert_element_type_3: "bf16[128, 32, 112, 112]" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        return (sum_1, mul_10, convert_element_type_3)



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
