"""
Standalone repro captured via capture_hook.
Label: torchbench_dcgan_train
Pattern hash: 12fc8ebf8180
Shape hash: a564ddd4
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
    def forward(self, arg0_1: "bf16[32, 128, 16, 16]", arg1_1: "bf16[32, 128, 16, 16]", arg2_1: "bf16[32, 128, 16, 16]", arg3_1: "f32[1, 128, 1, 1]", arg4_1: "f32[128]", arg5_1: "f32[128]"):
        # No stacktrace found for following nodes
        convert_element_type: "f32[32, 128, 16, 16]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        convert_element_type_1: "f32[32, 128, 16, 16]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        gt: "b8[32, 128, 16, 16]" = torch.ops.aten.gt.Scalar(convert_element_type_1, 0);  convert_element_type_1 = None
        mul: "f32[32, 128, 16, 16]" = torch.ops.aten.mul.Tensor(convert_element_type, 0.2)
        where: "f32[32, 128, 16, 16]" = torch.ops.aten.where.self(gt, convert_element_type, mul);  gt = convert_element_type = mul = None
        convert_element_type_2: "bf16[32, 128, 16, 16]" = torch.ops.prims.convert_element_type.default(where, torch.bfloat16);  where = None
        convert_element_type_3: "f32[32, 128, 16, 16]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        sum_1: "f32[128]" = torch.ops.aten.sum.dim_IntList(convert_element_type_3, [0, 2, 3])
        convert_element_type_4: "f32[32, 128, 16, 16]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        sub: "f32[32, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convert_element_type_4, arg3_1);  convert_element_type_4 = arg3_1 = None
        mul_1: "f32[32, 128, 16, 16]" = torch.ops.aten.mul.Tensor(convert_element_type_3, sub)
        sum_2: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1, [0, 2, 3]);  mul_1 = None
        mul_2: "f32[128]" = torch.ops.aten.mul.Tensor(sum_1, 0.0001220703125)
        unsqueeze: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_2, 0);  mul_2 = None
        unsqueeze_1: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        mul_3: "f32[128]" = torch.ops.aten.mul.Tensor(sum_2, 0.0001220703125)
        mul_4: "f32[128]" = torch.ops.aten.mul.Tensor(arg4_1, arg4_1)
        mul_5: "f32[128]" = torch.ops.aten.mul.Tensor(mul_3, mul_4);  mul_3 = mul_4 = None
        unsqueeze_3: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_5, 0);  mul_5 = None
        unsqueeze_4: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        unsqueeze_5: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        mul_6: "f32[128]" = torch.ops.aten.mul.Tensor(arg4_1, arg5_1);  arg5_1 = None
        unsqueeze_6: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_6, 0);  mul_6 = None
        unsqueeze_7: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
        unsqueeze_8: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        mul_7: "f32[32, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_5);  sub = unsqueeze_5 = None
        sub_1: "f32[32, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convert_element_type_3, mul_7);  convert_element_type_3 = mul_7 = None
        sub_2: "f32[32, 128, 16, 16]" = torch.ops.aten.sub.Tensor(sub_1, unsqueeze_2);  sub_1 = unsqueeze_2 = None
        mul_8: "f32[32, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_8);  sub_2 = unsqueeze_8 = None
        mul_9: "f32[128]" = torch.ops.aten.mul.Tensor(sum_2, arg4_1);  sum_2 = arg4_1 = None
        convert_element_type_5: "bf16[32, 128, 16, 16]" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None
        return (sum_1, mul_9, convert_element_type_5)



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
