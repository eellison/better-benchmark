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
    def forward(self, mm_45: "f32[8192, 768]", arg182_1: "f32[768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:509 in forward, code: key_vectors = self.key(hidden_states)
        reshape_default: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_45, [1024, 8, 768]);  mm_45 = None
        add_tensor: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(reshape_default, arg182_1);  reshape_default = arg182_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:521 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        reshape_default_1: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(add_tensor, [1024, 8, 12, 64]);  add_tensor = None
        permute_default: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(reshape_default_1, [1, 0, 2, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:786 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_default_1: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        reshape_default_2: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_default_1, [96, 1024, 64]);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: hidden_states = hidden_states.view(
        reshape_default_3: "f32[96, 2, 512, 64]" = torch.ops.aten.reshape.default(reshape_default_2, [96, 2, 512, 64]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:730 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_default: "f32[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(reshape_default_3, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_default: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_default, 4);  as_strided_default = None
        permute_default_2: "f32[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_default, [0, 1, 4, 2, 3]);  unsqueeze_default = None
        permute_default_3: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_default_2, [0, 1, 4, 3, 2]);  permute_default_2 = None
        clone_default: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_default_3, memory_format = torch.contiguous_format);  permute_default_3 = None
        reshape_default_4: "f32[288, 64, 512]" = torch.ops.aten.reshape.default(clone_default, [288, 64, 512]);  clone_default = None
        return reshape_default_4


def _default_make_inputs():
    return [
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
