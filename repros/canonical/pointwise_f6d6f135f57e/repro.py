"""
Standalone repro captured via capture_hook.
Label: vllm_Qwen_Qwen3-30B-A3B_001
Pattern hash: f6d6f135f57e
Shape hash: abcc6a60
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([16384, 1536], bf16), T([16384, 768], bf16))"

class Repro(torch.nn.Module):
    def forward(self, arg77_1: "bf16[16384, 1536]", _grouped_mm_13: "bf16[16384, 768]"):
        # No stacktrace found for following nodes
        split_tensor = torch.ops.aten.split.Tensor(arg77_1, 768, -1);  arg77_1 = None
        getitem: "bf16[16384, 768]" = split_tensor[0]
        getitem_1: "bf16[16384, 768]" = split_tensor[1];  split_tensor = None
        convert_element_type_default: "f32[16384, 768]" = torch.ops.prims.convert_element_type.default(getitem, torch.float32);  getitem = None
        neg_default: "f32[16384, 768]" = torch.ops.aten.neg.default(convert_element_type_default)
        exp_default: "f32[16384, 768]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[16384, 768]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[16384, 768]" = torch.ops.aten.div.Tensor(convert_element_type_default, add_tensor)
        convert_element_type_default_1: "bf16[16384, 768]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        mul_tensor: "bf16[16384, 768]" = torch.ops.aten.mul.Tensor(_grouped_mm_13, convert_element_type_default_1);  convert_element_type_default_1 = None
        mul_tensor_1: "bf16[16384, 768]" = torch.ops.aten.mul.Tensor(_grouped_mm_13, getitem_1);  _grouped_mm_13 = getitem_1 = None
        convert_element_type_default_2: "f32[16384, 768]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.float32);  mul_tensor_1 = None
        reciprocal_default: "f32[16384, 768]" = torch.ops.aten.reciprocal.default(add_tensor);  add_tensor = None
        mul_tensor_2: "f32[16384, 768]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_3: "f32[16384, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, mul_tensor_2);  convert_element_type_default_2 = None
        sub_tensor: "f32[16384, 768]" = torch.ops.aten.sub.Tensor(1, mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_4: "f32[16384, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, sub_tensor);  convert_element_type_default = sub_tensor = None
        add_tensor_1: "f32[16384, 768]" = torch.ops.aten.add.Tensor(mul_tensor_4, 1);  mul_tensor_4 = None
        mul_tensor_5: "f32[16384, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_3, add_tensor_1);  mul_tensor_3 = add_tensor_1 = None
        convert_element_type_default_3: "bf16[16384, 768]" = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.bfloat16);  mul_tensor_5 = None
        cat_default: "bf16[16384, 1536]" = torch.ops.aten.cat.default([convert_element_type_default_3, mul_tensor], 1);  convert_element_type_default_3 = mul_tensor = None
        permute_default: "bf16[1536, 16384]" = torch.ops.aten.permute.default(cat_default, [1, 0]);  cat_default = None
        return permute_default

def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)

def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
