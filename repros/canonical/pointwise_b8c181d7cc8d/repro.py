"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s1_g5
Pattern hash: b8c181d7cc8d
Shape hash: 201f75e8
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
    def forward(self, getitem_96: "f32[1, 1024, 1]", convert_element_type_72: "f32[1, 1024, 1024]", getitem_97: "f32[1, 1024, 1]", arg195_1: "bf16[1024]", arg196_1: "bf16[1024]", arg1_1: "bf16[50265, 1024]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_default: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_72, getitem_97);  convert_element_type_72 = getitem_97 = None
        mul_tensor: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, arg195_1);  mul_tensor = arg195_1 = None
        add_tensor_1: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg196_1);  mul_tensor_1 = arg196_1 = None
        convert_element_type_default: "bf16[1, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        permute_default: "bf16[1024, 50265]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        reshape_default: "bf16[1024, 1024]" = torch.ops.aten.reshape.default(convert_element_type_default, [1024, 1024]);  convert_element_type_default = None
        constant_pad_nd_default: "bf16[1024, 50272]" = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 7, 0, 0]);  permute_default = None
        return (reshape_default, constant_pad_nd_default)


def _default_make_inputs():
    return [
    torch.randn([1, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([50265, 1024], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
