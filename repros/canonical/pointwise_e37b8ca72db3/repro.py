"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm: "bf16[85, 768]", mm_1: "bf16[85, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:99 in forward, code: return nn.functional.silu(input)
        convert_element_type_default: "f32[85, 768]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        neg_default: "f32[85, 768]" = torch.ops.aten.neg.default(convert_element_type_default)
        exp_default: "f32[85, 768]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[85, 768]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[85, 768]" = torch.ops.aten.div.Tensor(convert_element_type_default, add_tensor);  convert_element_type_default = add_tensor = None
        convert_element_type_default_1: "bf16[85, 768]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:209 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_tensor: "bf16[85, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, mm_1);  convert_element_type_default_1 = mm_1 = None
        return mul_tensor


def _default_make_inputs():
    return [
    torch.randn([85, 768], dtype=torch.bfloat16, device='cuda'),
    torch.randn([85, 768], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
