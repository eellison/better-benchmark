"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_train
Pattern hash: d227dc0811d2
Shape hash: 2570be15
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
    def forward(self, arg0_1: "bf16[128, 512, 14, 14]", arg1_1: "bf16[128, 512, 28, 28]", arg2_1: "bf16[128, 512, 28, 28]", arg3_1: "bf16[128, 512, 28, 28]", arg4_1: "bf16[128, 512, 28, 28]", arg5_1: "bf16[128, 512, 1, 1]"):
        # No stacktrace found for following nodes
        avg_pool2d_backward: "bf16[128, 512, 28, 28]" = torch.ops.aten.avg_pool2d_backward.default(arg0_1, arg1_1, [2, 2], [2, 2], [0, 0], True, False, None);  arg0_1 = arg1_1 = None
        add: "bf16[128, 512, 28, 28]" = torch.ops.aten.add.Tensor(arg2_1, avg_pool2d_backward);  arg2_1 = avg_pool2d_backward = None
        mul: "bf16[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(add, 0.9622504486493761);  add = None
        convert_element_type: "f32[128, 512, 28, 28]" = torch.ops.prims.convert_element_type.default(mul, torch.float32);  mul = None
        convert_element_type_1: "f32[128, 512, 28, 28]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        sigmoid: "f32[128, 512, 28, 28]" = torch.ops.aten.sigmoid.default(convert_element_type_1)
        mul_1: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(convert_element_type, sigmoid);  convert_element_type = None
        sub: "f32[128, 512, 28, 28]" = torch.ops.aten.sub.Tensor(1, sigmoid);  sigmoid = None
        mul_2: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(convert_element_type_1, sub);  convert_element_type_1 = sub = None
        add_1: "f32[128, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_2, 1);  mul_2 = None
        mul_3: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_1, add_1);  mul_1 = add_1 = None
        convert_element_type_2: "bf16[128, 512, 28, 28]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        mul_4: "bf16[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.2)
        mul_5: "bf16[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_4, 2.0);  mul_4 = None
        mul_6: "bf16[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_5, arg4_1);  arg4_1 = None
        sigmoid_1: "bf16[128, 512, 1, 1]" = torch.ops.aten.sigmoid.default(arg5_1);  arg5_1 = None
        sum_1: "f32[128, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_6, [2, 3], True, dtype = torch.float32);  mul_6 = None
        convert_element_type_3: "bf16[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(sum_1, torch.bfloat16);  sum_1 = None
        convert_element_type_4: "f32[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        convert_element_type_5: "f32[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(sigmoid_1, torch.float32)
        sub_1: "f32[128, 512, 1, 1]" = torch.ops.aten.sub.Tensor(1, convert_element_type_5)
        mul_7: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_5, sub_1);  convert_element_type_5 = sub_1 = None
        mul_8: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_4, mul_7);  convert_element_type_4 = mul_7 = None
        convert_element_type_6: "bf16[128, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None
        sum_2: "bf16[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_6, [0, 2, 3])
        convert_element_type_7: "f32[512]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        return (convert_element_type_2, mul_5, sigmoid_1, convert_element_type_6, convert_element_type_7)



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
