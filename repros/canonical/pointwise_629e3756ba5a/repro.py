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
    def forward(self, mm: "f32[32768, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1197 in forward, code: query_vectors = self.query(hidden_states)
        reshape_default: "f32[8, 4096, 768]" = torch.ops.aten.reshape.default(mm, [8, 4096, 768]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:406 in _split_hidden_size_dim, code: x = x.view(*new_x_shape)
        reshape_default_1: "f32[8, 4096, 12, 64]" = torch.ops.aten.reshape.default(reshape_default, [8, 4096, 12, 64]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:407 in _split_hidden_size_dim, code: return x.transpose(2, 1)
        permute_default: "f32[8, 12, 4096, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:424 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        reshape_default_2: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(permute_default, [8, 12, 64, 64, 64]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1271 in forward, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        expand_default: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.expand.default(reshape_default_2, [8, 12, 64, 64, 64]);  reshape_default_2 = None
        clone_default: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_3: "f32[6144, 64, 64]" = torch.ops.aten.reshape.default(clone_default, [6144, 64, 64]);  clone_default = None
        return reshape_default_3


def _default_make_inputs():
    return [
    torch.randn([32768, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
