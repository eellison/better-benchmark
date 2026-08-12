"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: c91e0b70e149
Shape hash: d3452c96
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
    def forward(self, arg0_1: "bf16[512, 120, 28, 28]", arg1_1: "bf16[512, 60, 28, 28]", arg2_1: "bf16[512, 60, 28, 28]", arg3_1: "bf16[]", arg4_1: "bf16[512, 60, 28, 28]", arg5_1: "f32[1, 60, 1, 1]", arg6_1: "f32[60]", arg7_1: "f32[60]"):
        # No stacktrace found for following nodes
        slice_1: "bf16[512, 60, 28, 28]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 0, 60);  arg0_1 = None
        add: "bf16[512, 60, 28, 28]" = torch.ops.aten.add.Tensor(slice_1, arg1_1);  slice_1 = arg1_1 = None
        le: "b8[512, 60, 28, 28]" = torch.ops.aten.le.Scalar(arg2_1, 0);  arg2_1 = None
        where: "bf16[512, 60, 28, 28]" = torch.ops.aten.where.self(le, arg3_1, add);  le = arg3_1 = add = None
        convert_element_type: "f32[512, 60, 28, 28]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        sum_1: "f32[60]" = torch.ops.aten.sum.dim_IntList(convert_element_type, [0, 2, 3])
        convert_element_type_1: "f32[512, 60, 28, 28]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        sub: "f32[512, 60, 28, 28]" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg5_1);  convert_element_type_1 = arg5_1 = None
        mul: "f32[512, 60, 28, 28]" = torch.ops.aten.mul.Tensor(convert_element_type, sub)
        sum_2: "f32[60]" = torch.ops.aten.sum.dim_IntList(mul, [0, 2, 3]);  mul = None
        mul_1: "f32[60]" = torch.ops.aten.mul.Tensor(sum_1, 2.4912308673469386e-06)
        unsqueeze: "f32[1, 60]" = torch.ops.aten.unsqueeze.default(mul_1, 0);  mul_1 = None
        unsqueeze_1: "f32[1, 60, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2: "f32[1, 60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        mul_2: "f32[60]" = torch.ops.aten.mul.Tensor(sum_2, 2.4912308673469386e-06)
        mul_3: "f32[60]" = torch.ops.aten.mul.Tensor(arg6_1, arg6_1)
        mul_4: "f32[60]" = torch.ops.aten.mul.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        unsqueeze_3: "f32[1, 60]" = torch.ops.aten.unsqueeze.default(mul_4, 0);  mul_4 = None
        unsqueeze_4: "f32[1, 60, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        unsqueeze_5: "f32[1, 60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        mul_5: "f32[60]" = torch.ops.aten.mul.Tensor(arg6_1, arg7_1);  arg7_1 = None
        unsqueeze_6: "f32[1, 60]" = torch.ops.aten.unsqueeze.default(mul_5, 0);  mul_5 = None
        unsqueeze_7: "f32[1, 60, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
        unsqueeze_8: "f32[1, 60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        mul_6: "f32[512, 60, 28, 28]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_5);  sub = unsqueeze_5 = None
        sub_1: "f32[512, 60, 28, 28]" = torch.ops.aten.sub.Tensor(convert_element_type, mul_6);  convert_element_type = mul_6 = None
        sub_2: "f32[512, 60, 28, 28]" = torch.ops.aten.sub.Tensor(sub_1, unsqueeze_2);  sub_1 = unsqueeze_2 = None
        mul_7: "f32[512, 60, 28, 28]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_8);  sub_2 = unsqueeze_8 = None
        mul_8: "f32[60]" = torch.ops.aten.mul.Tensor(sum_2, arg6_1);  sum_2 = arg6_1 = None
        convert_element_type_2: "bf16[512, 60, 28, 28]" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None
        return (sum_1, mul_8, convert_element_type_2)



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
