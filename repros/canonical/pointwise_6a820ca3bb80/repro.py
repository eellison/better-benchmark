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
    def forward(self, mm_3: "bf16[2048, 32768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:403 in forward, code: k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        reshape_default: "bf16[4, 512, 32768]" = torch.ops.aten.reshape.default(mm_3, [4, 512, 32768]);  mm_3 = None
        reshape_default_1: "bf16[4, 512, 128, 256]" = torch.ops.aten.reshape.default(reshape_default, [4, 512, -1, 256]);  reshape_default = None
        permute_default: "bf16[4, 128, 512, 256]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:404 in forward, code: k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        split_with_sizes_default = torch.ops.aten.split_with_sizes.default(permute_default, [128, 128], -1);  permute_default = None
        getitem: "bf16[4, 128, 512, 128]" = split_with_sizes_default[0]
        getitem_1: "bf16[4, 128, 512, 128]" = split_with_sizes_default[1];  split_with_sizes_default = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([2048, 32768], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
