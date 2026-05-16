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
    def forward(self, addmm_66: "f32[8192, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:160 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        reshape_default: "f32[4, 2048, 768]" = torch.ops.aten.reshape.default(addmm_66, [4, 2048, 768]);  addmm_66 = None
        mul_tensor: "f32[4, 2048, 768]" = torch.ops.aten.mul.Tensor(reshape_default, 0.125);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:161 in forward, code: query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        reshape_default_1: "f32[4, 2048, 12, 64]" = torch.ops.aten.reshape.default(mul_tensor, [4, -1, 12, 64]);  mul_tensor = None
        permute_default: "f32[4, 12, 2048, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None
        return permute_default


def _default_make_inputs():
    return [
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
