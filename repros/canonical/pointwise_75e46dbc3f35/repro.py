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
    def forward(self, bmm_23: "f32[384, 256, 64]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:877 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        reshape_default: "f32[96, 4, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_23, [96, 4, 256, 1, 64]);  bmm_23 = None
        permute_default: "f32[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(reshape_default, [0, 1, 2, 4, 3]);  reshape_default = None
        reshape_default_1: "f32[96, 4, 256, 64]" = torch.ops.aten.reshape.default(permute_default, [96, 4, 256, 64]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:878 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        reshape_default_2: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(reshape_default_1, [8, 12, 1024, 64]);  reshape_default_1 = None
        permute_default_1: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:617 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_default_2: "f32[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_default_1, [1, 0, 2, 3]);  permute_default_1 = None
        clone_default: "f32[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        reshape_default_3: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(clone_default, [1024, 8, 768]);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:646 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_default_3: "f32[8, 1024, 768]" = torch.ops.aten.permute.default(reshape_default_3, [1, 0, 2]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1093 in forward, code: hidden_states = self.dense(hidden_states)
        clone_default_1: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(permute_default_3, memory_format = torch.contiguous_format);  permute_default_3 = None
        reshape_default_4: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_default_1, [8192, 768]);  clone_default_1 = None
        return reshape_default_4


def _default_make_inputs():
    return [
    torch.randn([384, 256, 64], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
