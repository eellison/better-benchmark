"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g53
Pattern hash: 99c11af030f9
Shape hash: 7f8304dc
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
    def forward(self, mm_1: "f16[32, 512]", arg354_1: "i64[32]", arg355_1: "i64[32]", arg54_1: "f32[512]", arg353_1: "f32[32, 77, 512]", arg361_1: "f32[32, 77, 1]", mm_100: "f16[1600, 3072]", arg215_1: "f16[1600, 3072]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[32, 512]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        full_default: "f32[32, 77, 512]" = torch.ops.aten.full.default([32, 77, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[32, 77, 512]" = torch.ops.aten.index_put.default(full_default, [arg354_1, arg355_1], convert_element_type_default, True);  full_default = arg354_1 = arg355_1 = convert_element_type_default = None
        mul_tensor: "f32[32, 77, 512]" = torch.ops.aten.mul.Tensor(index_put_default, arg54_1);  index_put_default = arg54_1 = None
        mul_tensor_1: "f32[32, 77, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, 512)
        sum_dim_int_list: "f32[32, 77, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[32, 77, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, arg353_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[32, 77, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[32, 77, 512]" = torch.ops.aten.mul.Tensor(arg353_1, sum_dim_int_list_1);  arg353_1 = sum_dim_int_list_1 = None
        sub_tensor: "f32[32, 77, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[32, 77, 512]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[32, 77, 512]" = torch.ops.aten.mul.Tensor(arg361_1, sub_tensor_1);  arg361_1 = sub_tensor_1 = None
        permute_default: "f32[77, 32, 512]" = torch.ops.aten.permute.default(mul_tensor_4, [1, 0, 2]);  mul_tensor_4 = None
        convert_element_type_default_1: "f16[77, 32, 512]" = torch.ops.prims.convert_element_type.default(permute_default, torch.float16);  permute_default = None
        clone_default: "f16[77, 32, 512]" = torch.ops.aten.clone.default(convert_element_type_default_1, memory_format = torch.contiguous_format);  convert_element_type_default_1 = None
        reshape_default: "f16[2464, 512]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None
        reshape_default_1: "f16[32, 50, 3072]" = torch.ops.aten.reshape.default(mm_100, _shape_param_1);  mm_100 = _shape_param_1 = None
        reshape_default_2: "f16[32, 50, 3072]" = torch.ops.aten.reshape.default(arg215_1, _shape_param_2);  arg215_1 = _shape_param_2 = None
        mul_tensor_5: "f16[32, 50, 3072]" = torch.ops.aten.mul.Tensor(reshape_default_2, 1.702)
        sigmoid_default: "f16[32, 50, 3072]" = torch.ops.aten.sigmoid.default(mul_tensor_5);  mul_tensor_5 = None
        mul_tensor_6: "f16[32, 50, 3072]" = torch.ops.aten.mul.Tensor(reshape_default_1, sigmoid_default)
        mul_tensor_7: "f16[32, 50, 3072]" = torch.ops.aten.mul.Tensor(reshape_default_1, reshape_default_2);  reshape_default_1 = reshape_default_2 = None
        convert_element_type_default_2: "f32[32, 50, 3072]" = torch.ops.prims.convert_element_type.default(mul_tensor_7, torch.float32);  mul_tensor_7 = None
        convert_element_type_default_3: "f32[32, 50, 3072]" = torch.ops.prims.convert_element_type.default(sigmoid_default, torch.float32);  sigmoid_default = None
        sub_tensor_2: "f32[32, 50, 3072]" = torch.ops.aten.sub.Tensor(1, convert_element_type_default_3)
        mul_tensor_8: "f32[32, 50, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_default_3, sub_tensor_2);  convert_element_type_default_3 = sub_tensor_2 = None
        mul_tensor_9: "f32[32, 50, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, mul_tensor_8);  convert_element_type_default_2 = mul_tensor_8 = None
        convert_element_type_default_4: "f16[32, 50, 3072]" = torch.ops.prims.convert_element_type.default(mul_tensor_9, torch.float16);  mul_tensor_9 = None
        mul_tensor_10: "f16[32, 50, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 1.702);  convert_element_type_default_4 = None
        add_tensor: "f16[32, 50, 3072]" = torch.ops.aten.add.Tensor(mul_tensor_6, mul_tensor_10);  mul_tensor_6 = mul_tensor_10 = None
        reshape_default_3: "f16[1600, 3072]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_3);  add_tensor = _shape_param_3 = None
        return (reshape_default, reshape_default_3)


def _default_make_inputs():
    return [
    torch.randn([32, 512], dtype=torch.float16, device='cuda'),
    torch.randint(0, 2, [32], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [32], dtype=torch.int64, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 77, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 77, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1600, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 3072], dtype=torch.float16, device='cuda'),
    [2464, 512],  # _shape_param_0
    [32, 50, 3072],  # _shape_param_1
    [32, 50, 3072],  # _shape_param_2
    [1600, 3072],  # _shape_param_3
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
