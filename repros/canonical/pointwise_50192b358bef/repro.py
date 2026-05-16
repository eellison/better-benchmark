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
    def forward(self, mm_2: "f32[32768, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1199 in forward, code: value_vectors = self.value(hidden_states)
        reshape_default: "f32[8, 4096, 768]" = torch.ops.aten.reshape.default(mm_2, [8, 4096, 768]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:406 in _split_hidden_size_dim, code: x = x.view(*new_x_shape)
        reshape_default_1: "f32[8, 4096, 12, 64]" = torch.ops.aten.reshape.default(reshape_default, [8, 4096, 12, 64]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:407 in _split_hidden_size_dim, code: return x.transpose(2, 1)
        permute_default: "f32[8, 12, 4096, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:424 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        reshape_default_2: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(permute_default, [8, 12, 64, 64, 64]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_tensor: "f32[8, 12, 1, 64, 64]" = torch.ops.aten.slice.Tensor(reshape_default_2, 2, -1, 9223372036854775807)
        slice_tensor_1: "f32[8, 12, 63, 64, 64]" = torch.ops.aten.slice.Tensor(reshape_default_2, 2, 0, -1)
        cat_default: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.cat.default([slice_tensor, slice_tensor_1], 2);  slice_tensor = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:399 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_default_1: "f32[8, 12, 64, 128, 64]" = torch.ops.aten.cat.default([cat_default, reshape_default_2], 3);  cat_default = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1307 in forward, code: out_vectors = torch.matmul(attention_probs, value_vectors)
        expand_default: "f32[8, 12, 64, 128, 64]" = torch.ops.aten.expand.default(cat_default_1, [8, 12, 64, 128, 64]);  cat_default_1 = None
        reshape_default_3: "f32[6144, 128, 64]" = torch.ops.aten.reshape.default(expand_default, [6144, 128, 64]);  expand_default = None
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
