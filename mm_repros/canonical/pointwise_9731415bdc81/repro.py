"""
Standalone repro captured via capture_hook.
Label: lennard_jones
Pattern hash: 9731415bdc81
Shape hash: 3a4b8d8c
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
    def forward(self, arg0_1: "f32[16]", arg1_1: "f32[16, 1]", arg2_1: "f32[128, 1]", arg3_1: "f32[16]", arg4_1: "f32[16, 16]", arg5_1: "f32[16]", arg6_1: "f32[16, 16]", arg7_1: "f32[16]", arg8_1: "f32[16, 16]", arg9_1: "f32[1]", arg10_1: "f32[1, 16]"):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[16]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[16, 1]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convert_element_type_2: "bf16[128, 1]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        permute: "bf16[1, 16]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        convert_element_type_3: "f32[16]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        convert_element_type_4: "f32[128, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32)
        convert_element_type_5: "f32[1, 16]" = torch.ops.prims.convert_element_type.default(permute, torch.float32);  permute = None
        mul: "f32[128, 16]" = torch.ops.aten.mul.Tensor(convert_element_type_4, convert_element_type_5);  convert_element_type_4 = convert_element_type_5 = None
        mul_1: "f32[128, 16]" = torch.ops.aten.mul.Tensor(mul, 1);  mul = None
        mul_2: "f32[16]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1);  convert_element_type_3 = None
        add: "f32[128, 16]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        convert_element_type_6: "bf16[128, 16]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        tanh: "bf16[128, 16]" = torch.ops.aten.tanh.default(convert_element_type_6);  convert_element_type_6 = None
        convert_element_type_7: "bf16[16]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        convert_element_type_8: "bf16[16, 16]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        permute_1: "bf16[16, 16]" = torch.ops.aten.permute.default(convert_element_type_8, [1, 0]);  convert_element_type_8 = None
        addmm: "bf16[128, 16]" = torch.ops.aten.addmm.default(convert_element_type_7, tanh, permute_1);  convert_element_type_7 = None
        tanh_1: "bf16[128, 16]" = torch.ops.aten.tanh.default(addmm);  addmm = None
        convert_element_type_9: "bf16[16]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        convert_element_type_10: "bf16[16, 16]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
        permute_2: "bf16[16, 16]" = torch.ops.aten.permute.default(convert_element_type_10, [1, 0]);  convert_element_type_10 = None
        addmm_1: "bf16[128, 16]" = torch.ops.aten.addmm.default(convert_element_type_9, tanh_1, permute_2);  convert_element_type_9 = None
        tanh_2: "bf16[128, 16]" = torch.ops.aten.tanh.default(addmm_1);  addmm_1 = None
        convert_element_type_11: "bf16[16]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.bfloat16);  arg7_1 = None
        convert_element_type_12: "bf16[16, 16]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.bfloat16);  arg8_1 = None
        permute_3: "bf16[16, 16]" = torch.ops.aten.permute.default(convert_element_type_12, [1, 0]);  convert_element_type_12 = None
        addmm_2: "bf16[128, 16]" = torch.ops.aten.addmm.default(convert_element_type_11, tanh_2, permute_3);  convert_element_type_11 = None
        tanh_3: "bf16[128, 16]" = torch.ops.aten.tanh.default(addmm_2);  addmm_2 = None
        convert_element_type_13: "bf16[1]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.bfloat16);  arg9_1 = None
        convert_element_type_14: "bf16[1, 16]" = torch.ops.prims.convert_element_type.default(arg10_1, torch.bfloat16);  arg10_1 = None
        permute_4: "bf16[16, 1]" = torch.ops.aten.permute.default(convert_element_type_14, [1, 0]);  convert_element_type_14 = None
        addmm_3: "bf16[128, 1]" = torch.ops.aten.addmm.default(convert_element_type_13, tanh_3, permute_4);  convert_element_type_13 = permute_4 = None
        permute_5: "bf16[16, 16]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        permute_6: "bf16[16, 16]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_7: "bf16[16, 16]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        return (convert_element_type_2, tanh, tanh_1, tanh_2, tanh_3, addmm_3, permute_5, permute_6, permute_7)



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
