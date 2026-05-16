"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g53
Pattern hash: d162ff6db2c3
Shape hash: c75dff42
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_32: "f16[32, 8, 77, 64]", getitem_31: "f16[32, 8, 77, 64]", getitem_30: "f16[32, 8, 77, 64]", full_2: "f16[3, 77, 32, 512]", mm_187: "f16[1600, 768]", arg5_1: "f32[768]", arg74_1: "f32[32, 50, 768]", arg525_1: "f32[32, 50, 1]", add_114: "f32[32, 50, 768]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[256, 77, 64]" = torch.ops.aten.reshape.default(getitem_32, [256, 77, 64]);  getitem_32 = None
        reshape_default_1: "f16[256, 77, 64]" = torch.ops.aten.reshape.default(getitem_31, [256, 77, 64]);  getitem_31 = None
        reshape_default_2: "f16[256, 77, 64]" = torch.ops.aten.reshape.default(getitem_30, [256, 77, 64]);  getitem_30 = None
        permute_default: "f16[77, 256, 64]" = torch.ops.aten.permute.default(reshape_default, [1, 0, 2]);  reshape_default = None
        reshape_default_3: "f16[77, 32, 512]" = torch.ops.aten.reshape.default(permute_default, [77, 32, 512]);  permute_default = None
        permute_default_1: "f16[77, 256, 64]" = torch.ops.aten.permute.default(reshape_default_1, [1, 0, 2]);  reshape_default_1 = None
        reshape_default_4: "f16[77, 32, 512]" = torch.ops.aten.reshape.default(permute_default_1, [77, 32, 512]);  permute_default_1 = None
        permute_default_2: "f16[77, 256, 64]" = torch.ops.aten.permute.default(reshape_default_2, [1, 0, 2]);  reshape_default_2 = None
        reshape_default_5: "f16[77, 32, 512]" = torch.ops.aten.reshape.default(permute_default_2, [77, 32, 512]);  permute_default_2 = None
        select_scatter_default: "f16[3, 77, 32, 512]" = torch.ops.aten.select_scatter.default(full_2, reshape_default_3, 0, 2);  reshape_default_3 = None
        select_scatter_default_1: "f16[3, 77, 32, 512]" = torch.ops.aten.select_scatter.default(full_2, reshape_default_4, 0, 1);  reshape_default_4 = None
        add_tensor: "f16[3, 77, 32, 512]" = torch.ops.aten.add.Tensor(select_scatter_default, select_scatter_default_1);  select_scatter_default = select_scatter_default_1 = None
        select_scatter_default_2: "f16[3, 77, 32, 512]" = torch.ops.aten.select_scatter.default(full_2, reshape_default_5, 0, 0);  full_2 = reshape_default_5 = None
        add_tensor_1: "f16[3, 77, 32, 512]" = torch.ops.aten.add.Tensor(add_tensor, select_scatter_default_2);  add_tensor = select_scatter_default_2 = None
        unsqueeze_default: "f16[3, 77, 32, 1, 512]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 3);  add_tensor_1 = None
        permute_default_3: "f16[1, 77, 32, 3, 512]" = torch.ops.aten.permute.default(unsqueeze_default, [3, 1, 2, 0, 4]);  unsqueeze_default = None
        squeeze_dim: "f16[77, 32, 3, 512]" = torch.ops.aten.squeeze.dim(permute_default_3, 0);  permute_default_3 = None
        clone_default: "f16[77, 32, 3, 512]" = torch.ops.aten.clone.default(squeeze_dim, memory_format = torch.contiguous_format);  squeeze_dim = None
        reshape_default_6: "f16[77, 32, 1536]" = torch.ops.aten.reshape.default(clone_default, [77, 32, 1536]);  clone_default = None
        reshape_default_7: "f16[2464, 1536]" = torch.ops.aten.reshape.default(reshape_default_6, [2464, 1536]);  reshape_default_6 = None
        reshape_default_8: "f16[50, 32, 768]" = torch.ops.aten.reshape.default(mm_187, [50, 32, 768]);  mm_187 = None
        convert_element_type_default: "f32[50, 32, 768]" = torch.ops.prims.convert_element_type.default(reshape_default_8, torch.float32);  reshape_default_8 = None
        permute_default_4: "f32[32, 50, 768]" = torch.ops.aten.permute.default(convert_element_type_default, [1, 0, 2]);  convert_element_type_default = None
        mul_tensor: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(permute_default_4, arg5_1);  permute_default_4 = arg5_1 = None
        mul_tensor_1: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[32, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg74_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[32, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(arg74_1, sum_dim_int_list_1);  arg74_1 = sum_dim_int_list_1 = None
        sub_tensor: "f32[32, 50, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[32, 50, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(arg525_1, sub_tensor_1);  arg525_1 = sub_tensor_1 = None
        add_tensor_2: "f32[32, 50, 768]" = torch.ops.aten.add.Tensor(add_114, mul_tensor_4);  add_114 = mul_tensor_4 = None
        convert_element_type_default_1: "f16[32, 50, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float16);  add_tensor_2 = None
        reshape_default_9: "f16[1600, 768]" = torch.ops.aten.reshape.default(convert_element_type_default_1, [1600, 768]);  convert_element_type_default_1 = None
        return (reshape_default_7, reshape_default_9)


def _default_make_inputs():
    return [
    torch.randn(1261568, dtype=torch.float16, device='cuda').as_strided([32, 8, 77, 64], [512, 64, 16384, 1]),  # getitem_32
    torch.randn(1261568, dtype=torch.float16, device='cuda').as_strided([32, 8, 77, 64], [512, 64, 16384, 1]),  # getitem_31
    torch.randn(1261568, dtype=torch.float16, device='cuda').as_strided([32, 8, 77, 64], [512, 64, 16384, 1]),  # getitem_30
    torch.randn([3, 77, 32, 512], dtype=torch.float16, device='cuda'),
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
