"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g20
Pattern hash: 5be548710b92
Shape hash: f250c46d
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_145: "f32[32, 50, 1]", add_79: "f32[32, 50, 768]", getitem_146: "f32[32, 50, 1]", arg138_1: "f32[768]", arg139_1: "f32[768]", arg141_1: "f32[2304, 768]", getitem_303: "f32[77, 32, 1]", clone_58: "f32[77, 32, 512]", getitem_304: "f32[77, 32, 1]", arg289_1: "f32[512]", arg290_1: "f32[512]", arg291_1: "f32[1536]", arg292_1: "f32[1536, 512]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[32, 50, 1]" = torch.ops.aten.add.Tensor(getitem_145, 1e-05);  getitem_145 = None
        rsqrt_default: "f32[32, 50, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[32, 50, 768]" = torch.ops.aten.sub.Tensor(add_79, getitem_146);  add_79 = getitem_146 = None
        mul_tensor: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg138_1);  mul_tensor = arg138_1 = None
        add_tensor_1: "f32[32, 50, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg139_1);  mul_tensor_1 = arg139_1 = None
        permute_default: "f32[50, 32, 768]" = torch.ops.aten.permute.default(add_tensor_1, [1, 0, 2]);  add_tensor_1 = None
        convert_element_type_default: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(arg141_1, torch.float16);  arg141_1 = None
        convert_element_type_default_1: "f16[50, 32, 768]" = torch.ops.prims.convert_element_type.default(permute_default, torch.float16);  permute_default = None
        permute_default_1: "f16[768, 2304]" = torch.ops.aten.permute.default(convert_element_type_default, [1, 0]);  convert_element_type_default = None
        clone_default: "f16[50, 32, 768]" = torch.ops.aten.clone.default(convert_element_type_default_1, memory_format = torch.contiguous_format);  convert_element_type_default_1 = None
        reshape_default: "f16[1600, 768]" = torch.ops.aten.reshape.default(clone_default, [1600, 768]);  clone_default = None
        add_tensor_2: "f32[77, 32, 1]" = torch.ops.aten.add.Tensor(getitem_303, 1e-05);  getitem_303 = None
        rsqrt_default_1: "f32[77, 32, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor_1: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(clone_58, getitem_304);  clone_58 = getitem_304 = None
        mul_tensor_2: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        mul_tensor_3: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, arg289_1);  mul_tensor_2 = arg289_1 = None
        add_tensor_3: "f32[77, 32, 512]" = torch.ops.aten.add.Tensor(mul_tensor_3, arg290_1);  mul_tensor_3 = arg290_1 = None
        convert_element_type_default_2: "f16[1536]" = torch.ops.prims.convert_element_type.default(arg291_1, torch.float16);  arg291_1 = None
        convert_element_type_default_3: "f16[1536, 512]" = torch.ops.prims.convert_element_type.default(arg292_1, torch.float16);  arg292_1 = None
        convert_element_type_default_4: "f16[77, 32, 512]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float16);  add_tensor_3 = None
        reshape_default_1: "f16[2464, 512]" = torch.ops.aten.reshape.default(convert_element_type_default_4, [2464, 512]);  convert_element_type_default_4 = None
        permute_default_2: "f16[512, 1536]" = torch.ops.aten.permute.default(convert_element_type_default_3, [1, 0]);  convert_element_type_default_3 = None
        return (permute_default_1, reshape_default, convert_element_type_default_2, reshape_default_1, permute_default_2)


def _default_make_inputs():
    return [
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 512], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
