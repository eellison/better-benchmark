"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s2_g21
Pattern hash: 62f6aa643fe7
Shape hash: d31a42cb
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
    def forward(self, getitem_132: "f32[4, 1024, 1]", add_97: "f32[4, 1024, 768]", getitem_133: "f32[4, 1024, 1]", arg148_1: "f32[768]", arg149_1: "f32[768]", arg150_1: "f32[2, 768]", _shape_param_0):
        # No stacktrace found for following nodes
        add_tensor: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
        rsqrt_default: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_97, getitem_133);  add_97 = getitem_133 = None
        mul_tensor: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg148_1);  mul_tensor = arg148_1 = None
        add_tensor_1: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg149_1);  mul_tensor_1 = arg149_1 = None
        convert_element_type_default: "f16[2, 768]" = torch.ops.prims.convert_element_type.default(arg150_1, torch.float16);  arg150_1 = None
        convert_element_type_default_1: "f16[4, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None
        permute_default: "f16[768, 2]" = torch.ops.aten.permute.default(convert_element_type_default, [1, 0]);  convert_element_type_default = None
        reshape_default: "f16[4096, 768]" = torch.ops.aten.reshape.default(convert_element_type_default_1, _shape_param_0);  convert_element_type_default_1 = _shape_param_0 = None
        return (permute_default, reshape_default)


def _default_make_inputs():
    return [
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2, 768], dtype=torch.float32, device='cuda'),
    [4096, 768],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
