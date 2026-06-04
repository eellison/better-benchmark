"""
Standalone repro captured via capture_hook.
Label: vllm_Qwen_Qwen3-30B-A3B_000
Pattern hash: eae6c9133626
Shape hash: 2c61ad94
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([16384, 1536], bf16))"

class Repro(torch.nn.Module):
    def forward(self, _grouped_mm_6: "bf16[16384, 1536]"):
        # No stacktrace found for following nodes
        split_tensor = torch.ops.aten.split.Tensor(_grouped_mm_6, 768, -1);  _grouped_mm_6 = None
        getitem: "bf16[16384, 768]" = split_tensor[0]
        getitem_1: "bf16[16384, 768]" = split_tensor[1];  split_tensor = None
        convert_element_type_default: "f32[16384, 768]" = torch.ops.prims.convert_element_type.default(getitem, torch.float32);  getitem = None
        neg_default: "f32[16384, 768]" = torch.ops.aten.neg.default(convert_element_type_default)
        exp_default: "f32[16384, 768]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[16384, 768]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[16384, 768]" = torch.ops.aten.div.Tensor(convert_element_type_default, add_tensor);  convert_element_type_default = add_tensor = None
        convert_element_type_default_1: "bf16[16384, 768]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        mul_tensor: "bf16[16384, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, getitem_1);  convert_element_type_default_1 = getitem_1 = None
        return mul_tensor

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
