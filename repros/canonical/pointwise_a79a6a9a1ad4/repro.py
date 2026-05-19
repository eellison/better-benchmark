"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import glob
import os
import torch
from math import inf
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_58: "bf16[16384, 768]", getitem_59: "bf16[16384, 768]", arg46_1: "bf16[128, 2048, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_default: "f32[16384, 768]" = torch.ops.prims.convert_element_type.default(getitem_58, torch.float32);  getitem_58 = None
        neg_default: "f32[16384, 768]" = torch.ops.aten.neg.default(convert_element_type_default)
        exp_default: "f32[16384, 768]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[16384, 768]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[16384, 768]" = torch.ops.aten.div.Tensor(convert_element_type_default, add_tensor);  convert_element_type_default = add_tensor = None
        convert_element_type_default_1: "bf16[16384, 768]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:515 in _default_apply_gate, code: return self.act_fn(gate) * up  # (S, intermediate_dim)
        mul_tensor: "bf16[16384, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, getitem_59);  convert_element_type_default_1 = getitem_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:364 in _grouped_linear, code: out = _grouped_mm(input, weight.transpose(-2, -1), offs=offs)
        permute_default: "bf16[128, 768, 2048]" = torch.ops.aten.permute.default(arg46_1, [0, 2, 1]);  arg46_1 = None
        return (mul_tensor, permute_default)


def _default_make_inputs():
    return [
    torch.randn(25165056, dtype=torch.bfloat16, device='cuda').as_strided([16384, 768], [1536, 1]),  # getitem_58
    torch.randn(25165056, dtype=torch.bfloat16, device='cuda').as_strided([16384, 768], [1536, 1]),  # getitem_59
    torch.randn([128, 2048, 768], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
