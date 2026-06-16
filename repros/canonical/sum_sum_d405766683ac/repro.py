"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_train
Pattern hash: d405766683ac
Shape hash: 0e39883f
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
    def forward(self, arg0_1: "bf16[512, 72, 28, 28]", arg1_1: "f32[1, 72, 1, 1]", arg2_1: "f32[1, 72, 1, 1]", arg3_1: "f32[72]", arg4_1: "f32[72]", arg5_1: "bf16[512, 72, 28, 28]", arg6_1: "bf16[512, 72, 1, 1]", arg7_1: "f32[]"):
        # No stacktrace found for following nodes
        sub: "f32[512, 72, 28, 28]" = torch.ops.aten.sub.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        mul: "f32[512, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub, arg2_1);  sub = arg2_1 = None
        unsqueeze: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
        unsqueeze_1: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_1: "f32[512, 72, 28, 28]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_3: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add: "f32[512, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_3);  mul_1 = unsqueeze_3 = None
        convert_element_type: "bf16[512, 72, 28, 28]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        relu: "bf16[512, 72, 28, 28]" = torch.ops.aten.relu.default(convert_element_type);  convert_element_type = None
        mul_2: "bf16[512, 72, 28, 28]" = torch.ops.aten.mul.Tensor(arg5_1, relu);  arg5_1 = None
        convert_element_type_1: "f32[512, 72, 1, 1]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.float32);  arg6_1 = None
        sum_1: "f32[512, 72, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2, 3], True, dtype = torch.float32);  mul_2 = None
        convert_element_type_2: "bf16[512, 72, 1, 1]" = torch.ops.prims.convert_element_type.default(sum_1, torch.bfloat16);  sum_1 = None
        convert_element_type_3: "f32[512, 72, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        gt: "b8[512, 72, 1, 1]" = torch.ops.aten.gt.Scalar(convert_element_type_1, -3.0)
        lt: "b8[512, 72, 1, 1]" = torch.ops.aten.lt.Scalar(convert_element_type_1, 3.0)
        bitwise_and: "b8[512, 72, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt, lt);  gt = lt = None
        mul_3: "f32[512, 72, 1, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 0.16666666666666666);  convert_element_type_3 = None
        where: "f32[512, 72, 1, 1]" = torch.ops.aten.where.self(bitwise_and, mul_3, arg7_1);  bitwise_and = mul_3 = arg7_1 = None
        convert_element_type_4: "bf16[512, 72, 1, 1]" = torch.ops.prims.convert_element_type.default(where, torch.bfloat16);  where = None
        sum_2: "bf16[72]" = torch.ops.aten.sum.dim_IntList(convert_element_type_4, [0, 2, 3])
        convert_element_type_5: "f32[72]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        return (relu, convert_element_type_1, convert_element_type_4, convert_element_type_5)



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
