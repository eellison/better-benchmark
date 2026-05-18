"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g53
Pattern hash: cd6a22f6026d
Shape hash: e6ca600d
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_90: "f16[2464, 2048]", arg230_1: "f16[2464, 2048]", mm_190: "f16[1600, 768]", arg4_1: "f32[768]", arg70_1: "f32[32, 50, 768]", arg528_1: "f32[32, 50, 1]", add_117: "f32[32, 50, 768]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[77, 32, 2048]" = torch.ops.aten.reshape.default(mm_90, [77, 32, 2048]);  mm_90 = None
        reshape_default_1: "f16[77, 32, 2048]" = torch.ops.aten.reshape.default(arg230_1, [77, 32, 2048]);  arg230_1 = None
        mul_tensor: "f16[77, 32, 2048]" = torch.ops.aten.mul.Tensor(reshape_default_1, 1.702)
        sigmoid_default: "f16[77, 32, 2048]" = torch.ops.aten.sigmoid.default(mul_tensor);  mul_tensor = None
        mul_tensor_1: "f16[77, 32, 2048]" = torch.ops.aten.mul.Tensor(reshape_default, sigmoid_default)
        mul_tensor_2: "f16[77, 32, 2048]" = torch.ops.aten.mul.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None
        convert_element_type_default: "f32[77, 32, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.float32);  mul_tensor_2 = None
        convert_element_type_default_1: "f32[77, 32, 2048]" = torch.ops.prims.convert_element_type.default(sigmoid_default, torch.float32);  sigmoid_default = None
        sub_tensor: "f32[77, 32, 2048]" = torch.ops.aten.sub.Tensor(1, convert_element_type_default_1)
        mul_tensor_3: "f32[77, 32, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, sub_tensor);  convert_element_type_default_1 = sub_tensor = None
        mul_tensor_4: "f32[77, 32, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor_3);  convert_element_type_default = mul_tensor_3 = None
        convert_element_type_default_2: "f16[77, 32, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.float16);  mul_tensor_4 = None
        mul_tensor_5: "f16[77, 32, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 1.702);  convert_element_type_default_2 = None
        add_tensor: "f16[77, 32, 2048]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_5);  mul_tensor_1 = mul_tensor_5 = None
        reshape_default_2: "f16[2464, 2048]" = torch.ops.aten.reshape.default(add_tensor, [2464, 2048]);  add_tensor = None
        reshape_default_3: "f16[32, 50, 768]" = torch.ops.aten.reshape.default(mm_190, [32, 50, 768]);  mm_190 = None
        convert_element_type_default_3: "f32[32, 50, 768]" = torch.ops.prims.convert_element_type.default(reshape_default_3, torch.float32);  reshape_default_3 = None
        mul_tensor_6: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_3, arg4_1);  convert_element_type_default_3 = arg4_1 = None
        mul_tensor_7: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_6, 768)
        sum_dim_int_list: "f32[32, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [2], True)
        mul_tensor_8: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_6, arg70_1);  mul_tensor_6 = None
        sum_dim_int_list_1: "f32[32, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [2], True);  mul_tensor_8 = None
        mul_tensor_9: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(arg70_1, sum_dim_int_list_1);  arg70_1 = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[32, 50, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_7, sum_dim_int_list);  mul_tensor_7 = sum_dim_int_list = None
        sub_tensor_2: "f32[32, 50, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_9);  sub_tensor_1 = mul_tensor_9 = None
        mul_tensor_10: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(arg528_1, sub_tensor_2);  arg528_1 = sub_tensor_2 = None
        add_tensor_1: "f32[32, 50, 768]" = torch.ops.aten.add.Tensor(add_117, mul_tensor_10);  add_117 = mul_tensor_10 = None
        convert_element_type_default_4: "f16[32, 50, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None
        permute_default: "f16[50, 32, 768]" = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0, 2]);  convert_element_type_default_4 = None
        clone_default: "f16[50, 32, 768]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_4: "f16[1600, 768]" = torch.ops.aten.reshape.default(clone_default, [1600, 768]);  clone_default = None
        return (reshape_default_2, reshape_default_4)


def _default_make_inputs():
    return [
    torch.randn([2464, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
