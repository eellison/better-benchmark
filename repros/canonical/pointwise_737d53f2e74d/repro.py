"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s2_g21
Pattern hash: 737d53f2e74d
Shape hash: f5e67d4b
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
    def forward(self, getitem_130: "f32[4, 1024, 1]", add_92: "f32[4, 1024, 768]", getitem_131: "f32[4, 1024, 1]", arg142_1: "f32[768]", arg143_1: "f32[768]", arg145_1: "f32[768, 3072]", arg144_1: "f32[3072]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_default: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_92, getitem_131);  add_92 = getitem_131 = None
        mul_tensor: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg142_1);  mul_tensor = arg142_1 = None
        add_tensor_1: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg143_1);  mul_tensor_1 = arg143_1 = None
        reshape_default: "f32[4096, 768]" = torch.ops.aten.reshape.default(add_tensor_1, [-1, 768]);  add_tensor_1 = None
        convert_element_type_default: "f16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg145_1, torch.float16);  arg145_1 = None
        convert_element_type_default_1: "f16[4096, 768]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float16);  reshape_default = None
        convert_element_type_default_2: "f16[3072]" = torch.ops.prims.convert_element_type.default(arg144_1, torch.float16);  arg144_1 = None
        return (convert_element_type_default, convert_element_type_default_1, convert_element_type_default_2)


def _default_make_inputs():
    return [
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
