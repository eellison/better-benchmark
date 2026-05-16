"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g53
Pattern hash: a6ba7cd428c7
Shape hash: 6ca13f13
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_92: "f16[2464, 512]", arg31_1: "f32[512]", arg452_1: "f32[77, 32, 512]", arg453_1: "f32[77, 32, 1]", add_56: "f32[77, 32, 512]", mm_192: "f16[1600, 768]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[77, 32, 512]" = torch.ops.aten.reshape.default(mm_92, [77, 32, 512]);  mm_92 = None
        convert_element_type_default: "f32[77, 32, 512]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        mul_tensor: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, arg31_1);  convert_element_type_default = arg31_1 = None
        mul_tensor_1: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, 512)
        sum_dim_int_list: "f32[77, 32, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, arg452_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[77, 32, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(arg452_1, sum_dim_int_list_1);  arg452_1 = sum_dim_int_list_1 = None
        sub_tensor: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(arg453_1, sub_tensor_1);  arg453_1 = sub_tensor_1 = None
        add_tensor: "f32[77, 32, 512]" = torch.ops.aten.add.Tensor(add_56, mul_tensor_4);  add_56 = mul_tensor_4 = None
        convert_element_type_default_1: "f16[77, 32, 512]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float16);  add_tensor = None
        clone_default: "f16[77, 32, 512]" = torch.ops.aten.clone.default(convert_element_type_default_1, memory_format = torch.contiguous_format);  convert_element_type_default_1 = None
        reshape_default_1: "f16[2464, 512]" = torch.ops.aten.reshape.default(clone_default, [2464, 512]);  clone_default = None
        reshape_default_2: "f16[50, 32, 12, 64]" = torch.ops.aten.reshape.default(mm_192, [50, 32, 12, 64]);  mm_192 = None
        permute_default: "f16[32, 12, 50, 64]" = torch.ops.aten.permute.default(reshape_default_2, [1, 2, 0, 3]);  reshape_default_2 = None
        return (reshape_default_1, permute_default)


def _default_make_inputs():
    return [
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # arg452_1
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # add_56
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
