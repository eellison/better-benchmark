"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g20
Pattern hash: 1c8e7e56f84d
Shape hash: 2711d792
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_158: "f32[32, 1]", clone_24: "f32[32, 768]", getitem_159: "f32[32, 1]", arg150_1: "f32[768]", arg151_1: "f32[768]", arg152_1: "f32[768, 512]", getitem_316: "f32[32, 77, 1]", permute_242: "f32[32, 77, 512]", getitem_317: "f32[32, 77, 1]", arg301_1: "f32[512]", arg302_1: "f32[512]", arg153_1: "i64[32, 77]", arg303_1: "f32[512, 512]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[32, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-05);  getitem_158 = None
        rsqrt_default: "f32[32, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[32, 768]" = torch.ops.aten.sub.Tensor(clone_24, getitem_159);  clone_24 = getitem_159 = None
        mul_tensor: "f32[32, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg150_1);  mul_tensor = arg150_1 = None
        add_tensor_1: "f32[32, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg151_1);  mul_tensor_1 = arg151_1 = None
        convert_element_type_default: "f16[768, 512]" = torch.ops.prims.convert_element_type.default(arg152_1, torch.float16);  arg152_1 = None
        convert_element_type_default_1: "f16[32, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None
        add_tensor_2: "f32[32, 77, 1]" = torch.ops.aten.add.Tensor(getitem_316, 1e-05);  getitem_316 = None
        rsqrt_default_1: "f32[32, 77, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor_1: "f32[32, 77, 512]" = torch.ops.aten.sub.Tensor(permute_242, getitem_317);  permute_242 = getitem_317 = None
        mul_tensor_2: "f32[32, 77, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        mul_tensor_3: "f32[32, 77, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, arg301_1);  mul_tensor_2 = arg301_1 = None
        add_tensor_3: "f32[32, 77, 512]" = torch.ops.aten.add.Tensor(mul_tensor_3, arg302_1);  mul_tensor_3 = arg302_1 = None
        iota_default: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        argmax_default: "i64[32]" = torch.ops.aten.argmax.default(arg153_1, -1);  arg153_1 = None
        index_tensor: "f32[32, 512]" = torch.ops.aten.index.Tensor(add_tensor_3, [iota_default, argmax_default]);  add_tensor_3 = iota_default = argmax_default = None
        convert_element_type_default_2: "f16[512, 512]" = torch.ops.prims.convert_element_type.default(arg303_1, torch.float16);  arg303_1 = None
        convert_element_type_default_3: "f16[32, 512]" = torch.ops.prims.convert_element_type.default(index_tensor, torch.float16);  index_tensor = None
        permute_default: "f16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_default_2, [1, 0]);  convert_element_type_default_2 = None
        return (convert_element_type_default, convert_element_type_default_1, convert_element_type_default_3, permute_default)


def _default_make_inputs():
    return [
    torch.randn([32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 77, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 77, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 77, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [32, 77], dtype=torch.int64, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
