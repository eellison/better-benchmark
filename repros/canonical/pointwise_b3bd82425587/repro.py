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
    def forward(self, mm_70: "f32[4096, 2048]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:243 in forward, code: key = self.k_proj(hidden_states)
        reshape_default: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_70, [32, 128, 2048]);  mm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189 in _split_heads, code: tensor = tensor.view(new_shape)
        reshape_default_1: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(reshape_default, [32, 128, 16, 128]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:190 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_default: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_1: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(permute_default, [0, 1, 3, 2]);  permute_default = None
        expand_default: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_default_1, [32, 16, 128, 128]);  permute_default_1 = None
        clone_default: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_2: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_default, [512, 128, 128]);  clone_default = None
        return reshape_default_2


def _default_make_inputs():
    return [
    torch.randn([4096, 2048], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
