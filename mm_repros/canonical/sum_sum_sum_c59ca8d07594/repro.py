"""
Standalone repro captured via capture_hook.
Label: lennard_jones
Pattern hash: c59ca8d07594
Shape hash: 678ee69a
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
    def forward(self, arg0_1: "bf16[128, 1]", arg1_1: "f32[1, 16]", arg2_1: "bf16[128, 16]", arg3_1: "bf16[16, 16]", arg4_1: "bf16[128, 16]", arg5_1: "bf16[16, 16]", arg6_1: "bf16[128, 16]", arg7_1: "bf16[16, 16]", arg8_1: "bf16[128, 16]", arg9_1: "bf16[128, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        convert_element_type: "f32[128, 1]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        convert_element_type_1: "bf16[1, 16]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        permute: "bf16[16, 1]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        permute_1: "bf16[1, 16]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        convert_element_type_2: "f32[1, 16]" = torch.ops.prims.convert_element_type.default(permute_1, torch.float32);  permute_1 = None
        mul: "f32[128, 16]" = torch.ops.aten.mul.Tensor(convert_element_type, convert_element_type_2);  convert_element_type = convert_element_type_2 = None
        convert_element_type_3: "bf16[128, 16]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        convert_element_type_4: "f32[128, 16]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        convert_element_type_5: "f32[128, 16]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        mul_1: "f32[128, 16]" = torch.ops.aten.mul.Tensor(convert_element_type_5, convert_element_type_5);  convert_element_type_5 = None
        sub: "f32[128, 16]" = torch.ops.aten.sub.Tensor(1, mul_1);  mul_1 = None
        mul_2: "f32[128, 16]" = torch.ops.aten.mul.Tensor(convert_element_type_4, sub);  convert_element_type_4 = sub = None
        convert_element_type_6: "bf16[128, 16]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        mm: "bf16[128, 16]" = torch.ops.aten.mm.default(convert_element_type_6, arg3_1);  arg3_1 = None
        permute_2: "bf16[16, 128]" = torch.ops.aten.permute.default(convert_element_type_6, [1, 0])
        mm_1: "bf16[16, 16]" = torch.ops.aten.mm.default(permute_2, arg4_1);  permute_2 = None
        sum_1: "f32[1, 16]" = torch.ops.aten.sum.dim_IntList(convert_element_type_6, [0], True, dtype = torch.float32);  convert_element_type_6 = None
        view: "f32[16]" = torch.ops.aten.view.default(sum_1, _shape_param_0);  sum_1 = _shape_param_0 = None
        convert_element_type_7: "bf16[16]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16);  view = None
        convert_element_type_8: "f32[16, 16]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_9: "f32[16]" = torch.ops.prims.convert_element_type.default(convert_element_type_7, torch.float32);  convert_element_type_7 = None
        convert_element_type_10: "f32[128, 16]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        convert_element_type_11: "f32[128, 16]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        mul_3: "f32[128, 16]" = torch.ops.aten.mul.Tensor(convert_element_type_11, convert_element_type_11);  convert_element_type_11 = None
        sub_1: "f32[128, 16]" = torch.ops.aten.sub.Tensor(1, mul_3);  mul_3 = None
        mul_4: "f32[128, 16]" = torch.ops.aten.mul.Tensor(convert_element_type_10, sub_1);  convert_element_type_10 = sub_1 = None
        convert_element_type_12: "bf16[128, 16]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        mm_2: "bf16[128, 16]" = torch.ops.aten.mm.default(convert_element_type_12, arg5_1);  arg5_1 = None
        permute_3: "bf16[16, 128]" = torch.ops.aten.permute.default(convert_element_type_12, [1, 0])
        mm_3: "bf16[16, 16]" = torch.ops.aten.mm.default(permute_3, arg6_1);  permute_3 = None
        sum_2: "f32[1, 16]" = torch.ops.aten.sum.dim_IntList(convert_element_type_12, [0], True, dtype = torch.float32);  convert_element_type_12 = None
        view_1: "f32[16]" = torch.ops.aten.view.default(sum_2, _shape_param_1);  sum_2 = _shape_param_1 = None
        convert_element_type_13: "bf16[16]" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        convert_element_type_14: "f32[16, 16]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_15: "f32[16]" = torch.ops.prims.convert_element_type.default(convert_element_type_13, torch.float32);  convert_element_type_13 = None
        convert_element_type_16: "f32[128, 16]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_17: "f32[128, 16]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.float32);  arg6_1 = None
        mul_5: "f32[128, 16]" = torch.ops.aten.mul.Tensor(convert_element_type_17, convert_element_type_17);  convert_element_type_17 = None
        sub_2: "f32[128, 16]" = torch.ops.aten.sub.Tensor(1, mul_5);  mul_5 = None
        mul_6: "f32[128, 16]" = torch.ops.aten.mul.Tensor(convert_element_type_16, sub_2);  convert_element_type_16 = sub_2 = None
        convert_element_type_18: "bf16[128, 16]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        mm_4: "bf16[128, 16]" = torch.ops.aten.mm.default(convert_element_type_18, arg7_1);  arg7_1 = None
        permute_4: "bf16[16, 128]" = torch.ops.aten.permute.default(convert_element_type_18, [1, 0])
        mm_5: "bf16[16, 16]" = torch.ops.aten.mm.default(permute_4, arg8_1);  permute_4 = None
        sum_3: "f32[1, 16]" = torch.ops.aten.sum.dim_IntList(convert_element_type_18, [0], True, dtype = torch.float32);  convert_element_type_18 = None
        view_2: "f32[16]" = torch.ops.aten.view.default(sum_3, _shape_param_2);  sum_3 = _shape_param_2 = None
        convert_element_type_19: "bf16[16]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        convert_element_type_20: "f32[16, 16]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_21: "f32[16]" = torch.ops.prims.convert_element_type.default(convert_element_type_19, torch.float32);  convert_element_type_19 = None
        convert_element_type_22: "f32[128, 16]" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        convert_element_type_23: "f32[128, 16]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.float32);  arg8_1 = None
        mul_7: "f32[128, 16]" = torch.ops.aten.mul.Tensor(convert_element_type_23, convert_element_type_23);  convert_element_type_23 = None
        sub_3: "f32[128, 16]" = torch.ops.aten.sub.Tensor(1, mul_7);  mul_7 = None
        mul_8: "f32[128, 16]" = torch.ops.aten.mul.Tensor(convert_element_type_22, sub_3);  convert_element_type_22 = sub_3 = None
        convert_element_type_24: "bf16[128, 16]" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None
        permute_5: "bf16[16, 128]" = torch.ops.aten.permute.default(convert_element_type_24, [1, 0])
        mm_6: "bf16[16, 1]" = torch.ops.aten.mm.default(permute_5, arg9_1);  permute_5 = arg9_1 = None
        sum_4: "f32[1, 16]" = torch.ops.aten.sum.dim_IntList(convert_element_type_24, [0], True, dtype = torch.float32);  convert_element_type_24 = None
        view_3: "f32[16]" = torch.ops.aten.view.default(sum_4, _shape_param_3);  sum_4 = _shape_param_3 = None
        convert_element_type_25: "bf16[16]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        convert_element_type_26: "f32[16, 1]" = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        convert_element_type_27: "f32[16]" = torch.ops.prims.convert_element_type.default(convert_element_type_25, torch.float32);  convert_element_type_25 = None
        return (convert_element_type_8, convert_element_type_9, convert_element_type_14, convert_element_type_15, convert_element_type_20, convert_element_type_21, convert_element_type_26, convert_element_type_27)



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
