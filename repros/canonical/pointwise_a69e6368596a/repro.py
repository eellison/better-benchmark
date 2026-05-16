"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s2_g20
Pattern hash: a69e6368596a
Shape hash: 68b1e224
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_192: "f32[8, 198, 1]", add_84: "f32[8, 198, 768]", getitem_193: "f32[8, 198, 1]", arg150_1: "f32[768]", arg151_1: "f32[768]", arg153_1: "f32[1000]", arg152_1: "f32[1000, 768]", arg155_1: "f32[1000]", arg154_1: "f32[1000, 768]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_192, 1e-06);  getitem_192 = None
        rsqrt_default: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_84, getitem_193);  add_84 = getitem_193 = None
        mul_tensor: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg150_1);  mul_tensor = arg150_1 = None
        add_tensor_1: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg151_1);  mul_tensor_1 = arg151_1 = None
        select_int: "f32[8, 768]" = torch.ops.aten.select.int(add_tensor_1, 1, 0)
        select_int_1: "f32[8, 768]" = torch.ops.aten.select.int(add_tensor_1, 1, 1);  add_tensor_1 = None
        convert_element_type_default: "f16[1000]" = torch.ops.prims.convert_element_type.default(arg153_1, torch.float16);  arg153_1 = None
        convert_element_type_default_1: "f16[1000, 768]" = torch.ops.prims.convert_element_type.default(arg152_1, torch.float16);  arg152_1 = None
        convert_element_type_default_2: "f16[8, 768]" = torch.ops.prims.convert_element_type.default(select_int, torch.float16);  select_int = None
        permute_default: "f16[768, 1000]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        convert_element_type_default_3: "f16[1000]" = torch.ops.prims.convert_element_type.default(arg155_1, torch.float16);  arg155_1 = None
        convert_element_type_default_4: "f16[1000, 768]" = torch.ops.prims.convert_element_type.default(arg154_1, torch.float16);  arg154_1 = None
        convert_element_type_default_5: "f16[8, 768]" = torch.ops.prims.convert_element_type.default(select_int_1, torch.float16);  select_int_1 = None
        permute_default_1: "f16[768, 1000]" = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0]);  convert_element_type_default_4 = None
        return (convert_element_type_default, convert_element_type_default_2, permute_default, convert_element_type_default_3, convert_element_type_default_5, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([8, 198, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([1000], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1000], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
