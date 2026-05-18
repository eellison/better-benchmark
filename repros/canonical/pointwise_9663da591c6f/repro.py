"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s3_g62
Pattern hash: 9663da591c6f
Shape hash: 5872be08
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
    def forward(self, getitem_48: "f32[4, 476, 1]", add_97: "f32[4, 476, 768]", getitem_49: "f32[4, 476, 1]", arg198_1: "f32[768]", arg199_1: "f32[768]", arg201_1: "f32[768]", arg200_1: "f32[768, 768]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[4, 476, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_default: "f32[4, 476, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[4, 476, 768]" = torch.ops.aten.sub.Tensor(add_97, getitem_49);  add_97 = getitem_49 = None
        mul_tensor: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg198_1);  mul_tensor = arg198_1 = None
        add_tensor_1: "f32[4, 476, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg199_1);  mul_tensor_1 = arg199_1 = None
        select_int: "f32[4, 768]" = torch.ops.aten.select.int(add_tensor_1, 1, 0);  add_tensor_1 = None
        convert_element_type_default: "f16[768]" = torch.ops.prims.convert_element_type.default(arg201_1, torch.float16);  arg201_1 = None
        convert_element_type_default_1: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(arg200_1, torch.float16);  arg200_1 = None
        convert_element_type_default_2: "f16[4, 768]" = torch.ops.prims.convert_element_type.default(select_int, torch.float16);  select_int = None
        permute_default: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        return (convert_element_type_default, convert_element_type_default_2, permute_default)


def _default_make_inputs():
    return [
    torch.randn([4, 476, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
