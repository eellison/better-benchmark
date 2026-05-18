"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s1_g10
Pattern hash: 6dd8dcfeb24e
Shape hash: 30377639
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
    def forward(self, arg142_1: "bf16[96]", arg143_1: "bf16[96]", convolution_28: "bf16[128, 96, 35, 35]", arg144_1: "bf16[96]", arg145_1: "bf16[96]", cat_2: "bf16[128, 288, 35, 35]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[96]" = torch.ops.prims.convert_element_type.default(arg142_1, torch.float32);  arg142_1 = None
        convert_element_type_default_1: "f32[96]" = torch.ops.prims.convert_element_type.default(arg143_1, torch.float32);  arg143_1 = None
        add_tensor: "f32[96]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 0.001);  convert_element_type_default_1 = None
        sqrt_default: "f32[96]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, -1);  convert_element_type_default = None
        unsqueeze_default_1: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        unsqueeze_default_2: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        sub_tensor: "f32[128, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_default_1);  convolution_28 = unsqueeze_default_1 = None
        mul_tensor_1: "f32[128, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "bf16[96, 1]" = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_default_5: "bf16[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 96, 35, 35]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "bf16[96, 1]" = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
        unsqueeze_default_7: "bf16[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 96, 35, 35]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        convert_element_type_default_2: "bf16[128, 96, 35, 35]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        relu_default: "bf16[128, 96, 35, 35]" = torch.ops.aten.relu.default(convert_element_type_default_2);  convert_element_type_default_2 = None
        _low_memory_max_pool_with_offsets_default = torch.ops.prims._low_memory_max_pool_with_offsets.default(cat_2, [3, 3], [2, 2], [0, 0], [1, 1], False);  cat_2 = None
        getitem: "bf16[128, 288, 17, 17]" = _low_memory_max_pool_with_offsets_default[0]
        getitem_1: "i8[128, 288, 17, 17]" = _low_memory_max_pool_with_offsets_default[1];  _low_memory_max_pool_with_offsets_default = None
        return (relu_default, getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([96], dtype=torch.bfloat16, device='cuda'),
    torch.randn([96], dtype=torch.bfloat16, device='cuda'),
    torch.randn([128, 96, 35, 35], dtype=torch.bfloat16, device='cuda'),
    torch.randn([96], dtype=torch.bfloat16, device='cuda'),
    torch.randn([96], dtype=torch.bfloat16, device='cuda'),
    torch.randn([128, 288, 35, 35], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
