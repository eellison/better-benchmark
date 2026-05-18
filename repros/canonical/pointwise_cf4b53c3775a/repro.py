"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s6_g10
Pattern hash: cf4b53c3775a
Shape hash: 54e4ec62
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
    def forward(self, getitem_148: "f32[256, 16, 1]", convert_element_type_167: "f32[256, 16, 240]", getitem_149: "f32[256, 16, 1]", arg261_1: "bf16[240]", arg262_1: "bf16[240]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[256, 16, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-05);  getitem_148 = None
        rsqrt_default: "f32[256, 16, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[256, 16, 240]" = torch.ops.aten.sub.Tensor(convert_element_type_167, getitem_149);  convert_element_type_167 = getitem_149 = None
        mul_tensor: "f32[256, 16, 240]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[256, 16, 240]" = torch.ops.aten.mul.Tensor(mul_tensor, arg261_1);  mul_tensor = arg261_1 = None
        add_tensor_1: "f32[256, 16, 240]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg262_1);  mul_tensor_1 = arg262_1 = None
        convert_element_type_default: "bf16[256, 16, 240]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        reshape_default: "bf16[64, 4, 16, 240]" = torch.ops.aten.reshape.default(convert_element_type_default, [64, 4, 16, -1]);  convert_element_type_default = None
        permute_default: "bf16[64, 240, 16, 4]" = torch.ops.aten.permute.default(reshape_default, [0, 3, 2, 1]);  reshape_default = None
        clone_default: "bf16[64, 240, 16, 4]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "bf16[61440, 4, 2, 2]" = torch.ops.aten.reshape.default(clone_default, [61440, 4, 2, 2]);  clone_default = None
        permute_default_1: "bf16[61440, 2, 4, 2]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None
        clone_default_1: "bf16[61440, 2, 4, 2]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_2: "bf16[64, 240, 8, 8]" = torch.ops.aten.reshape.default(clone_default_1, [64, 240, 8, 8]);  clone_default_1 = None
        return reshape_default_2


def _default_make_inputs():
    return [
    torch.randn([256, 16, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256, 16, 240], dtype=torch.float32, device='cuda'),
    torch.randn([256, 16, 1], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.bfloat16, device='cuda'),
    torch.randn([240], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
