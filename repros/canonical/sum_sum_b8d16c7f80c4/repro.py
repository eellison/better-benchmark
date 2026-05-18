"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g53
Pattern hash: b8d16c7f80c4
Shape hash: df24e3e5
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
    def forward(self, getitem_2: "f16[32, 8, 77, 64]", getitem_1: "f16[32, 8, 77, 64]", getitem: "f16[32, 8, 77, 64]", mm_107: "f16[1600, 768]", arg25_1: "f32[768]", arg204_1: "f32[32, 50, 768]", arg465_1: "f32[32, 50, 1]", add_64: "f32[32, 50, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # No stacktrace found for following nodes
        reshape_default: "f16[256, 77, 64]" = torch.ops.aten.reshape.default(getitem_2, _shape_param_0);  getitem_2 = _shape_param_0 = None
        reshape_default_1: "f16[256, 77, 64]" = torch.ops.aten.reshape.default(getitem_1, _shape_param_1);  getitem_1 = _shape_param_1 = None
        reshape_default_2: "f16[256, 77, 64]" = torch.ops.aten.reshape.default(getitem, _shape_param_2);  getitem = _shape_param_2 = None
        permute_default: "f16[77, 256, 64]" = torch.ops.aten.permute.default(reshape_default, [1, 0, 2]);  reshape_default = None
        reshape_default_3: "f16[77, 32, 512]" = torch.ops.aten.reshape.default(permute_default, _shape_param_3);  permute_default = _shape_param_3 = None
        permute_default_1: "f16[77, 256, 64]" = torch.ops.aten.permute.default(reshape_default_1, [1, 0, 2]);  reshape_default_1 = None
        reshape_default_4: "f16[77, 32, 512]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_4);  permute_default_1 = _shape_param_4 = None
        permute_default_2: "f16[77, 256, 64]" = torch.ops.aten.permute.default(reshape_default_2, [1, 0, 2]);  reshape_default_2 = None
        reshape_default_5: "f16[77, 32, 512]" = torch.ops.aten.reshape.default(permute_default_2, _shape_param_5);  permute_default_2 = _shape_param_5 = None
        full_default: "f16[3, 77, 32, 512]" = torch.ops.aten.full.default([3, 77, 32, 512], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_default: "f16[3, 77, 32, 512]" = torch.ops.aten.select_scatter.default(full_default, reshape_default_3, 0, 2);  reshape_default_3 = None
        select_scatter_default_1: "f16[3, 77, 32, 512]" = torch.ops.aten.select_scatter.default(full_default, reshape_default_4, 0, 1);  reshape_default_4 = None
        add_tensor: "f16[3, 77, 32, 512]" = torch.ops.aten.add.Tensor(select_scatter_default, select_scatter_default_1);  select_scatter_default = select_scatter_default_1 = None
        select_scatter_default_2: "f16[3, 77, 32, 512]" = torch.ops.aten.select_scatter.default(full_default, reshape_default_5, 0, 0);  full_default = reshape_default_5 = None
        add_tensor_1: "f16[3, 77, 32, 512]" = torch.ops.aten.add.Tensor(add_tensor, select_scatter_default_2);  add_tensor = select_scatter_default_2 = None
        unsqueeze_default: "f16[3, 77, 32, 1, 512]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 3);  add_tensor_1 = None
        permute_default_3: "f16[1, 77, 32, 3, 512]" = torch.ops.aten.permute.default(unsqueeze_default, [3, 1, 2, 0, 4]);  unsqueeze_default = None
        squeeze_dim: "f16[77, 32, 3, 512]" = torch.ops.aten.squeeze.dim(permute_default_3, 0);  permute_default_3 = None
        clone_default: "f16[77, 32, 3, 512]" = torch.ops.aten.clone.default(squeeze_dim, memory_format = torch.contiguous_format);  squeeze_dim = None
        reshape_default_6: "f16[77, 32, 1536]" = torch.ops.aten.reshape.default(clone_default, _shape_param_6);  clone_default = _shape_param_6 = None
        reshape_default_7: "f16[2464, 1536]" = torch.ops.aten.reshape.default(reshape_default_6, _shape_param_7);  reshape_default_6 = _shape_param_7 = None
        reshape_default_8: "f16[50, 32, 768]" = torch.ops.aten.reshape.default(mm_107, _shape_param_8);  mm_107 = _shape_param_8 = None
        convert_element_type_default: "f32[50, 32, 768]" = torch.ops.prims.convert_element_type.default(reshape_default_8, torch.float32);  reshape_default_8 = None
        permute_default_4: "f32[32, 50, 768]" = torch.ops.aten.permute.default(convert_element_type_default, [1, 0, 2]);  convert_element_type_default = None
        mul_tensor: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(permute_default_4, arg25_1);  permute_default_4 = arg25_1 = None
        mul_tensor_1: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[32, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg204_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[32, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(arg204_1, sum_dim_int_list_1);  arg204_1 = sum_dim_int_list_1 = None
        sub_tensor: "f32[32, 50, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[32, 50, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(arg465_1, sub_tensor_1);  arg465_1 = sub_tensor_1 = None
        add_tensor_2: "f32[32, 50, 768]" = torch.ops.aten.add.Tensor(add_64, mul_tensor_4);  add_64 = mul_tensor_4 = None
        convert_element_type_default_1: "f16[32, 50, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float16);  add_tensor_2 = None
        reshape_default_9: "f16[1600, 768]" = torch.ops.aten.reshape.default(convert_element_type_default_1, _shape_param_9);  convert_element_type_default_1 = _shape_param_9 = None
        return (reshape_default_7, reshape_default_9)


def _default_make_inputs():
    return [
    torch.randn(1261568, dtype=torch.float16, device='cuda').as_strided([32, 8, 77, 64], [512, 64, 16384, 1]),  # getitem_2
    torch.randn(1261568, dtype=torch.float16, device='cuda').as_strided([32, 8, 77, 64], [512, 64, 16384, 1]),  # getitem_1
    torch.randn(1261568, dtype=torch.float16, device='cuda').as_strided([32, 8, 77, 64], [512, 64, 16384, 1]),  # getitem
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    [256, 77, 64],  # _shape_param_0
    [256, 77, 64],  # _shape_param_1
    [256, 77, 64],  # _shape_param_2
    [77, 32, 512],  # _shape_param_3
    [77, 32, 512],  # _shape_param_4
    [77, 32, 512],  # _shape_param_5
    [77, 32, 1536],  # _shape_param_6
    [2464, 1536],  # _shape_param_7
    [50, 32, 768],  # _shape_param_8
    [1600, 768],  # _shape_param_9
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
