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
    def forward(self, mm_46: "f32[8192, 768]", arg184_1: "f32[768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:510 in forward, code: value_vectors = self.value(hidden_states)
        reshape_default: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_46, [1024, 8, 768]);  mm_46 = None
        add_tensor: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(reshape_default, arg184_1);  reshape_default = arg184_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:598 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        reshape_default_1: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(add_tensor, [1024, 8, 12, 64]);  add_tensor = None
        permute_default: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(reshape_default_1, [1, 0, 2, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:859 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_default_1: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        reshape_default_2: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_default_1, [96, 1024, 64]);  permute_default_1 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default: "f32[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(reshape_default_2, [0, 0, 256, 256], -1.0);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:873 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_default: "f32[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_default, [96, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:877 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_default: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_default, 4);  as_strided_default = None
        clone_default: "f32[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_default, memory_format = torch.contiguous_format);  unsqueeze_default = None
        reshape_default_3: "f32[384, 768, 64]" = torch.ops.aten.reshape.default(clone_default, [384, 768, 64]);  clone_default = None
        return reshape_default_3


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
