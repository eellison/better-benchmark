"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g53
Pattern hash: 19c5443c1a41
Shape hash: b9d70ebc
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_88: "f16[2464, 512]", arg32_1: "f32[512]", arg448_1: "f32[77, 32, 512]", arg449_1: "f32[77, 32, 1]", add_53: "f32[77, 32, 512]", mm_188: "f16[1600, 3072]", arg72_1: "f16[1600, 3072]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[77, 32, 512]" = torch.ops.aten.reshape.default(mm_88, [77, 32, 512]);  mm_88 = None
        convert_element_type_default: "f32[77, 32, 512]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        mul_tensor: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, arg32_1);  convert_element_type_default = arg32_1 = None
        mul_tensor_1: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, 512)
        sum_dim_int_list: "f32[77, 32, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, arg448_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[77, 32, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(arg448_1, sum_dim_int_list_1);  arg448_1 = sum_dim_int_list_1 = None
        sub_tensor: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(arg449_1, sub_tensor_1);  arg449_1 = sub_tensor_1 = None
        add_tensor: "f32[77, 32, 512]" = torch.ops.aten.add.Tensor(add_53, mul_tensor_4);  add_53 = mul_tensor_4 = None
        convert_element_type_default_1: "f16[77, 32, 512]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float16);  add_tensor = None
        clone_default: "f16[77, 32, 512]" = torch.ops.aten.clone.default(convert_element_type_default_1, memory_format = torch.contiguous_format);  convert_element_type_default_1 = None
        reshape_default_1: "f16[2464, 512]" = torch.ops.aten.reshape.default(clone_default, [2464, 512]);  clone_default = None
        reshape_default_2: "f16[32, 50, 3072]" = torch.ops.aten.reshape.default(mm_188, [32, 50, 3072]);  mm_188 = None
        reshape_default_3: "f16[32, 50, 3072]" = torch.ops.aten.reshape.default(arg72_1, [32, 50, 3072]);  arg72_1 = None
        mul_tensor_5: "f16[32, 50, 3072]" = torch.ops.aten.mul.Tensor(reshape_default_3, 1.702)
        sigmoid_default: "f16[32, 50, 3072]" = torch.ops.aten.sigmoid.default(mul_tensor_5);  mul_tensor_5 = None
        mul_tensor_6: "f16[32, 50, 3072]" = torch.ops.aten.mul.Tensor(reshape_default_2, sigmoid_default)
        mul_tensor_7: "f16[32, 50, 3072]" = torch.ops.aten.mul.Tensor(reshape_default_2, reshape_default_3);  reshape_default_2 = reshape_default_3 = None
        convert_element_type_default_2: "f32[32, 50, 3072]" = torch.ops.prims.convert_element_type.default(mul_tensor_7, torch.float32);  mul_tensor_7 = None
        convert_element_type_default_3: "f32[32, 50, 3072]" = torch.ops.prims.convert_element_type.default(sigmoid_default, torch.float32);  sigmoid_default = None
        sub_tensor_2: "f32[32, 50, 3072]" = torch.ops.aten.sub.Tensor(1, convert_element_type_default_3)
        mul_tensor_8: "f32[32, 50, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_default_3, sub_tensor_2);  convert_element_type_default_3 = sub_tensor_2 = None
        mul_tensor_9: "f32[32, 50, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, mul_tensor_8);  convert_element_type_default_2 = mul_tensor_8 = None
        convert_element_type_default_4: "f16[32, 50, 3072]" = torch.ops.prims.convert_element_type.default(mul_tensor_9, torch.float16);  mul_tensor_9 = None
        mul_tensor_10: "f16[32, 50, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 1.702);  convert_element_type_default_4 = None
        add_tensor_1: "f16[32, 50, 3072]" = torch.ops.aten.add.Tensor(mul_tensor_6, mul_tensor_10);  mul_tensor_6 = mul_tensor_10 = None
        reshape_default_4: "f16[1600, 3072]" = torch.ops.aten.reshape.default(add_tensor_1, [1600, 3072]);  add_tensor_1 = None
        return (reshape_default_1, reshape_default_4)


def _default_make_inputs():
    return [
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # arg448_1
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # add_53
    torch.randn([1600, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 3072], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
