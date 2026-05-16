"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s4_g35
Pattern hash: ac50e462aa4e
Shape hash: b5d2a2ee
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_9: "f16[4096, 512]", mm_11: "f16[4096, 512]", mm_29: "f16[4096, 512]", mm_31: "f16[4096, 512]", mm_49: "f16[4096, 512]", mm_51: "f16[4096, 512]", mm_69: "f16[4096, 512]", mm_71: "f16[4096, 512]", mm_89: "f16[4096, 512]", mm_91: "f16[4096, 512]", mm_109: "f16[4096, 512]", mm_111: "f16[4096, 512]", arg121_1: "b8[4, 1024, 512]", arg13_1: "f32[512]", arg119_1: "f32[4, 1024, 512]", arg120_1: "f32[4, 1024, 1]", arg118_1: "b8[4, 1024, 512]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_9, [4, 1024, 512]);  mm_9 = None
        convert_element_type_default: "f32[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        reshape_default_1: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_11, [4, 1024, 512]);  mm_11 = None
        convert_element_type_default_1: "f32[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(reshape_default_1, torch.float32);  reshape_default_1 = None
        add_tensor: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(convert_element_type_default, convert_element_type_default_1);  convert_element_type_default = convert_element_type_default_1 = None
        reshape_default_2: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_29, [4, 1024, 512]);  mm_29 = None
        convert_element_type_default_2: "f32[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(reshape_default_2, torch.float32);  reshape_default_2 = None
        add_tensor_1: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default_2);  add_tensor = convert_element_type_default_2 = None
        reshape_default_3: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_31, [4, 1024, 512]);  mm_31 = None
        convert_element_type_default_3: "f32[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(reshape_default_3, torch.float32);  reshape_default_3 = None
        add_tensor_2: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_1, convert_element_type_default_3);  add_tensor_1 = convert_element_type_default_3 = None
        reshape_default_4: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_49, [4, 1024, 512]);  mm_49 = None
        convert_element_type_default_4: "f32[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(reshape_default_4, torch.float32);  reshape_default_4 = None
        add_tensor_3: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_2, convert_element_type_default_4);  add_tensor_2 = convert_element_type_default_4 = None
        reshape_default_5: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_51, [4, 1024, 512]);  mm_51 = None
        convert_element_type_default_5: "f32[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(reshape_default_5, torch.float32);  reshape_default_5 = None
        add_tensor_4: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_3, convert_element_type_default_5);  add_tensor_3 = convert_element_type_default_5 = None
        reshape_default_6: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_69, [4, 1024, 512]);  mm_69 = None
        convert_element_type_default_6: "f32[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(reshape_default_6, torch.float32);  reshape_default_6 = None
        add_tensor_5: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_4, convert_element_type_default_6);  add_tensor_4 = convert_element_type_default_6 = None
        reshape_default_7: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_71, [4, 1024, 512]);  mm_71 = None
        convert_element_type_default_7: "f32[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(reshape_default_7, torch.float32);  reshape_default_7 = None
        add_tensor_6: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_5, convert_element_type_default_7);  add_tensor_5 = convert_element_type_default_7 = None
        reshape_default_8: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_89, [4, 1024, 512]);  mm_89 = None
        convert_element_type_default_8: "f32[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(reshape_default_8, torch.float32);  reshape_default_8 = None
        add_tensor_7: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_6, convert_element_type_default_8);  add_tensor_6 = convert_element_type_default_8 = None
        reshape_default_9: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_91, [4, 1024, 512]);  mm_91 = None
        convert_element_type_default_9: "f32[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(reshape_default_9, torch.float32);  reshape_default_9 = None
        add_tensor_8: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_7, convert_element_type_default_9);  add_tensor_7 = convert_element_type_default_9 = None
        reshape_default_10: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_109, [4, 1024, 512]);  mm_109 = None
        convert_element_type_default_10: "f32[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(reshape_default_10, torch.float32);  reshape_default_10 = None
        add_tensor_9: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_8, convert_element_type_default_10);  add_tensor_8 = convert_element_type_default_10 = None
        reshape_default_11: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_111, [4, 1024, 512]);  mm_111 = None
        convert_element_type_default_11: "f32[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(reshape_default_11, torch.float32);  reshape_default_11 = None
        add_tensor_10: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_9, convert_element_type_default_11);  add_tensor_9 = convert_element_type_default_11 = None
        convert_element_type_default_12: "f32[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg121_1, torch.float32);  arg121_1 = None
        mul_tensor: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_12, 1.1111111111111112);  convert_element_type_default_12 = None
        mul_tensor_1: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_10, mul_tensor);  add_tensor_10 = mul_tensor = None
        mul_tensor_2: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg13_1);  mul_tensor_1 = arg13_1 = None
        mul_tensor_3: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, arg119_1)
        mul_tensor_4: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, arg120_1);  mul_tensor_2 = None
        sum_dim_int_list: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        pow_tensor_scalar: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg120_1, 3);  arg120_1 = None
        mul_scalar: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list, -0.5);  sum_dim_int_list = None
        mul_tensor_5: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None
        expand_default: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_tensor_5, [4, 1024, 512]);  mul_tensor_5 = None
        div_scalar: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_default, 512);  expand_default = None
        pow_tensor_scalar_1: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg119_1, 1.0);  arg119_1 = None
        mul_scalar_1: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_6: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_11: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_6);  mul_tensor_4 = mul_tensor_6 = None
        convert_element_type_default_13: "f16[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_tensor_11, torch.float16);  add_tensor_11 = None
        convert_element_type_default_14: "f16[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg118_1, torch.float16);  arg118_1 = None
        mul_tensor_7: "f16[4, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_14, 1.1111111111111112);  convert_element_type_default_14 = None
        mul_tensor_8: "f16[4, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_13, mul_tensor_7);  convert_element_type_default_13 = mul_tensor_7 = None
        reshape_default_12: "f16[4096, 512]" = torch.ops.aten.reshape.default(mul_tensor_8, [4096, 512]);  mul_tensor_8 = None
        return reshape_default_12


def _default_make_inputs():
    return [
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randint(0, 2, [4, 1024, 512], dtype=torch.bool, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [4, 1024, 512], dtype=torch.bool, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
