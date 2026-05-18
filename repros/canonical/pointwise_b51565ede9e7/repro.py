"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_2: "bf16[2048, 576]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:400 in forward, code: compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        reshape_default: "bf16[4, 512, 576]" = torch.ops.aten.reshape.default(mm_2, [4, 512, 576]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:401 in forward, code: k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        split_with_sizes_default = torch.ops.aten.split_with_sizes.default(reshape_default, [512, 64], -1);  reshape_default = None
        getitem: "bf16[4, 512, 512]" = split_with_sizes_default[0]
        getitem_1: "bf16[4, 512, 64]" = split_with_sizes_default[1];  split_with_sizes_default = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([2048, 576], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
