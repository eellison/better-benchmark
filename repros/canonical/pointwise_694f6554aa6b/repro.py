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
    def forward(self, bmm_1: "f32[6144, 64, 64]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1307 in forward, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        reshape_default: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(bmm_1, [8, 12, 64, 64, 64]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1314 in forward, code: out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)
        reshape_default_1: "f32[8, 12, 4096, 64]" = torch.ops.aten.reshape.default(reshape_default, [8, 12, 4096, 64]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:413 in _merge_hidden_size_dims, code: x = x.permute(0, 2, 1, 3)
        permute_default: "f32[8, 4096, 12, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:414 in _merge_hidden_size_dims, code: return torch.reshape(x, (x.size()[0], -1, num_attn_heads * attn_head_size))
        clone_default: "f32[8, 4096, 12, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_2: "f32[8, 4096, 768]" = torch.ops.aten.reshape.default(clone_default, [8, 4096, 768]);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1370 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_3: "f32[32768, 768]" = torch.ops.aten.reshape.default(reshape_default_2, [32768, 768]);  reshape_default_2 = None
        return reshape_default_3


def _default_make_inputs():
    return [
    torch.randn([6144, 64, 64], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
