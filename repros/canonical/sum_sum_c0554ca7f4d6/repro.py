"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_train
Pattern hash: c0554ca7f4d6
Shape hash: d723c5b4
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
    def forward(self, arg0_1: "bf16[128, 320, 8, 8]", arg1_1: "bf16[128, 160, 8, 8]", arg2_1: "f32[1, 160, 1, 1]", arg3_1: "f32[1, 160, 1, 1]", arg4_1: "f32[160]", arg5_1: "f32[160]"):
        # No stacktrace found for following nodes
        slice_1: "bf16[128, 160, 8, 8]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 160, 320);  arg0_1 = None
        convert_element_type: "f32[128, 160, 8, 8]" = torch.ops.prims.convert_element_type.default(slice_1, torch.float32);  slice_1 = None
        sub: "f32[128, 160, 8, 8]" = torch.ops.aten.sub.Tensor(arg1_1, arg2_1)
        mul: "f32[128, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub, arg3_1);  sub = None
        unsqueeze: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1)
        unsqueeze_1: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_1: "f32[128, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_3: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add: "f32[128, 160, 8, 8]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_3);  mul_1 = unsqueeze_3 = None
        convert_element_type_1: "bf16[128, 160, 8, 8]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        convert_element_type_2: "f32[128, 160, 8, 8]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        sigmoid: "f32[128, 160, 8, 8]" = torch.ops.aten.sigmoid.default(convert_element_type_2)
        mul_2: "f32[128, 160, 8, 8]" = torch.ops.aten.mul.Tensor(convert_element_type, sigmoid);  convert_element_type = None
        sub_1: "f32[128, 160, 8, 8]" = torch.ops.aten.sub.Tensor(1, sigmoid);  sigmoid = None
        mul_3: "f32[128, 160, 8, 8]" = torch.ops.aten.mul.Tensor(convert_element_type_2, sub_1);  convert_element_type_2 = sub_1 = None
        add_1: "f32[128, 160, 8, 8]" = torch.ops.aten.add.Tensor(mul_3, 1);  mul_3 = None
        mul_4: "f32[128, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul_2, add_1);  mul_2 = add_1 = None
        convert_element_type_3: "bf16[128, 160, 8, 8]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        convert_element_type_4: "f32[128, 160, 8, 8]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        squeeze: "f32[160]" = torch.ops.aten.squeeze.dims(arg2_1, [0, 2, 3]);  arg2_1 = None
        unsqueeze_4: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_5: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None
        unsqueeze_6: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3);  unsqueeze_5 = None
        sum_1: "f32[160]" = torch.ops.aten.sum.dim_IntList(convert_element_type_4, [0, 2, 3])
        convert_element_type_5: "f32[128, 160, 8, 8]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        sub_2: "f32[128, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convert_element_type_5, unsqueeze_6);  convert_element_type_5 = unsqueeze_6 = None
        mul_5: "f32[128, 160, 8, 8]" = torch.ops.aten.mul.Tensor(convert_element_type_4, sub_2)
        sum_2: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 2, 3]);  mul_5 = None
        mul_6: "f32[160]" = torch.ops.aten.mul.Tensor(sum_1, 0.0001220703125)
        unsqueeze_7: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_6, 0);  mul_6 = None
        unsqueeze_8: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 2);  unsqueeze_7 = None
        unsqueeze_9: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 3);  unsqueeze_8 = None
        mul_7: "f32[160]" = torch.ops.aten.mul.Tensor(sum_2, 0.0001220703125)
        squeeze_1: "f32[160]" = torch.ops.aten.squeeze.dims(arg3_1, [0, 2, 3]);  arg3_1 = None
        mul_8: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_9: "f32[160]" = torch.ops.aten.mul.Tensor(mul_7, mul_8);  mul_7 = mul_8 = None
        unsqueeze_10: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_9, 0);  mul_9 = None
        unsqueeze_11: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None
        unsqueeze_12: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 3);  unsqueeze_11 = None
        mul_10: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_1, arg4_1);  arg4_1 = None
        unsqueeze_13: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_10, 0);  mul_10 = None
        unsqueeze_14: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 2);  unsqueeze_13 = None
        unsqueeze_15: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 3);  unsqueeze_14 = None
        mul_11: "f32[128, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_12);  sub_2 = unsqueeze_12 = None
        sub_3: "f32[128, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convert_element_type_4, mul_11);  convert_element_type_4 = mul_11 = None
        sub_4: "f32[128, 160, 8, 8]" = torch.ops.aten.sub.Tensor(sub_3, unsqueeze_9);  sub_3 = unsqueeze_9 = None
        mul_12: "f32[128, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_15);  sub_4 = unsqueeze_15 = None
        mul_13: "f32[160]" = torch.ops.aten.mul.Tensor(sum_2, squeeze_1);  sum_2 = squeeze_1 = None
        convert_element_type_6: "bf16[128, 160, 8, 8]" = torch.ops.prims.convert_element_type.default(mul_12, torch.bfloat16);  mul_12 = None
        return (sum_1, mul_13, convert_element_type_6)



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
