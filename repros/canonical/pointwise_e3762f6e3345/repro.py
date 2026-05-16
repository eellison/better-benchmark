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
    def forward(self, mm_44: "f32[8192, 768]", arg180_1: "f32[768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:508 in forward, code: query_vectors = self.query(hidden_states)
        reshape_default: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_44, [1024, 8, 768]);  mm_44 = None
        add_tensor: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(reshape_default, arg180_1);  reshape_default = arg180_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:518 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_tensor: "f32[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_tensor, 8.0);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        reshape_default_1: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(div_tensor, [1024, 8, 12, 64]);  div_tensor = None
        permute_default: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(reshape_default_1, [1, 0, 2, 3]);  reshape_default_1 = None
        permute_default_1: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        reshape_default_2: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_default_1, [96, 1024, 64]);  permute_default_1 = None
        reshape_default_3: "f32[96, 2, 512, 64]" = torch.ops.aten.reshape.default(reshape_default_2, [96, 2, 512, 64]);  reshape_default_2 = None
        as_strided_default: "f32[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(reshape_default_3, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  reshape_default_3 = None
        unsqueeze_default: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_default, 4);  as_strided_default = None
        clone_default: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_default, memory_format = torch.contiguous_format);  unsqueeze_default = None
        reshape_default_4: "f32[288, 512, 64]" = torch.ops.aten.reshape.default(clone_default, [288, 512, 64]);  clone_default = None
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
