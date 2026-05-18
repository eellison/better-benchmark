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
    def forward(self, bmm_142: "f32[512, 128, 128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:205 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        reshape_default: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_142, [32, 16, 128, 128]);  bmm_142 = None
        permute_default: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(reshape_default, [0, 1, 3, 2]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:190 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_default_1: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189 in _split_heads, code: tensor = tensor.view(new_shape)
        reshape_default_1: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(permute_default_1, [32, 128, 2048]);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:243 in forward, code: key = self.k_proj(hidden_states)
        clone_default: "f32[32, 128, 2048]" = torch.ops.aten.clone.default(reshape_default_1, memory_format = torch.contiguous_format);  reshape_default_1 = None
        reshape_default_2: "f32[4096, 2048]" = torch.ops.aten.reshape.default(clone_default, [4096, 2048]);  clone_default = None
        permute_default_2: "f32[2048, 4096]" = torch.ops.aten.permute.default(reshape_default_2, [1, 0]);  reshape_default_2 = None
        return permute_default_2


def _default_make_inputs():
    return [
    torch.randn([512, 128, 128], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
