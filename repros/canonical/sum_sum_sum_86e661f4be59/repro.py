"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 86e661f4be59
Shape hash: 641bb11a
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
    def forward(self, arg0_1: "bf16[512, 112, 14, 14]", arg1_1: "bf16[512, 112, 14, 14]", arg2_1: "bf16[512, 112, 14, 14]", arg3_1: "f32[1, 112, 1, 1]", arg4_1: "f32[112]", arg5_1: "f32[112]", arg6_1: "bf16[512, 56, 14, 14]", arg7_1: "f32[1, 56, 1, 1]", arg8_1: "f32[56]", arg9_1: "f32[56]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        add: "bf16[512, 112, 14, 14]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        new_empty_strided: "bf16[512, 112, 14, 14]" = torch.ops.aten.new_empty_strided.default(add, _shape_param_0, _shape_param_1);  _shape_param_0 = _shape_param_1 = None
        copy: "bf16[512, 112, 14, 14]" = torch.ops.aten.copy.default(new_empty_strided, add);  new_empty_strided = add = None
        clone: "bf16[512, 112, 14, 14]" = torch.ops.aten.clone.default(copy, memory_format = torch.contiguous_format)
        copy_1: "bf16[512, 112, 14, 14]" = torch.ops.aten.copy.default(copy, clone);  copy = None
        convert_element_type: "f32[512, 112, 14, 14]" = torch.ops.prims.convert_element_type.default(clone, torch.float32);  clone = None
        sum_1: "f32[112]" = torch.ops.aten.sum.dim_IntList(convert_element_type, [0, 2, 3])
        convert_element_type_1: "f32[512, 112, 14, 14]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        sub: "f32[512, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg3_1);  convert_element_type_1 = arg3_1 = None
        mul: "f32[512, 112, 14, 14]" = torch.ops.aten.mul.Tensor(convert_element_type, sub)
        sum_2: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul, [0, 2, 3]);  mul = None
        mul_1: "f32[112]" = torch.ops.aten.mul.Tensor(sum_1, 9.964923469387754e-06)
        unsqueeze: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_1, 0);  mul_1 = None
        unsqueeze_1: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        mul_2: "f32[112]" = torch.ops.aten.mul.Tensor(sum_2, 9.964923469387754e-06)
        mul_3: "f32[112]" = torch.ops.aten.mul.Tensor(arg4_1, arg4_1)
        mul_4: "f32[112]" = torch.ops.aten.mul.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        unsqueeze_3: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_4, 0);  mul_4 = None
        unsqueeze_4: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        unsqueeze_5: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        mul_5: "f32[112]" = torch.ops.aten.mul.Tensor(arg4_1, arg5_1);  arg5_1 = None
        unsqueeze_6: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_5, 0);  mul_5 = None
        unsqueeze_7: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
        unsqueeze_8: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        mul_6: "f32[512, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_5);  sub = unsqueeze_5 = None
        sub_1: "f32[512, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convert_element_type, mul_6);  convert_element_type = mul_6 = None
        sub_2: "f32[512, 112, 14, 14]" = torch.ops.aten.sub.Tensor(sub_1, unsqueeze_2);  sub_1 = unsqueeze_2 = None
        mul_7: "f32[512, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_8);  sub_2 = unsqueeze_8 = None
        mul_8: "f32[112]" = torch.ops.aten.mul.Tensor(sum_2, arg4_1);  sum_2 = arg4_1 = None
        convert_element_type_2: "bf16[512, 112, 14, 14]" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None
        slice_1: "bf16[512, 56, 14, 14]" = torch.ops.aten.slice.Tensor(copy_1, 1, 56, 112)
        convert_element_type_3: "f32[512, 56, 14, 14]" = torch.ops.prims.convert_element_type.default(slice_1, torch.float32);  slice_1 = None
        sum_3: "f32[56]" = torch.ops.aten.sum.dim_IntList(convert_element_type_3, [0, 2, 3])
        convert_element_type_4: "f32[512, 56, 14, 14]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.float32);  arg6_1 = None
        sub_3: "f32[512, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convert_element_type_4, arg7_1);  convert_element_type_4 = arg7_1 = None
        mul_9: "f32[512, 56, 14, 14]" = torch.ops.aten.mul.Tensor(convert_element_type_3, sub_3)
        sum_4: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_9, [0, 2, 3]);  mul_9 = None
        mul_10: "f32[56]" = torch.ops.aten.mul.Tensor(sum_3, 9.964923469387754e-06)
        unsqueeze_9: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_10, 0);  mul_10 = None
        unsqueeze_10: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 2);  unsqueeze_9 = None
        unsqueeze_11: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 3);  unsqueeze_10 = None
        mul_11: "f32[56]" = torch.ops.aten.mul.Tensor(sum_4, 9.964923469387754e-06)
        mul_12: "f32[56]" = torch.ops.aten.mul.Tensor(arg8_1, arg8_1)
        mul_13: "f32[56]" = torch.ops.aten.mul.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_12: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_13, 0);  mul_13 = None
        unsqueeze_13: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 2);  unsqueeze_12 = None
        unsqueeze_14: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 3);  unsqueeze_13 = None
        mul_14: "f32[56]" = torch.ops.aten.mul.Tensor(arg8_1, arg9_1);  arg9_1 = None
        unsqueeze_15: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_14, 0);  mul_14 = None
        unsqueeze_16: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_15, 2);  unsqueeze_15 = None
        unsqueeze_17: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, 3);  unsqueeze_16 = None
        mul_15: "f32[512, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_14);  sub_3 = unsqueeze_14 = None
        sub_4: "f32[512, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convert_element_type_3, mul_15);  convert_element_type_3 = mul_15 = None
        sub_5: "f32[512, 56, 14, 14]" = torch.ops.aten.sub.Tensor(sub_4, unsqueeze_11);  sub_4 = unsqueeze_11 = None
        mul_16: "f32[512, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_17);  sub_5 = unsqueeze_17 = None
        mul_17: "f32[56]" = torch.ops.aten.mul.Tensor(sum_4, arg8_1);  sum_4 = arg8_1 = None
        convert_element_type_5: "bf16[512, 56, 14, 14]" = torch.ops.prims.convert_element_type.default(mul_16, torch.bfloat16);  mul_16 = None
        return (copy_1, sum_1, mul_8, convert_element_type_2, sum_3, mul_17, convert_element_type_5)



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
