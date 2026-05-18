"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_training
Pattern hash: 6fd2e7e81344
Shape hash: 5d81550e
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_44: "f32[8192, 768]", _shape_param_0, primals_181: "f32[768]", mm_45: "f32[8192, 768]", _shape_param_1, primals_183: "f32[768]", primals_184: "f32[768, 768]", _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        reshape_default: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_44, _shape_param_0);  mm_44 = _shape_param_0 = None
        add_tensor: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(reshape_default, primals_181);  reshape_default = primals_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        reshape_default_1: "f32[1024, 8, 768]" = torch.ops.aten.reshape.default(mm_45, _shape_param_1);  mm_45 = _shape_param_1 = None
        add_tensor_1: "f32[1024, 8, 768]" = torch.ops.aten.add.Tensor(reshape_default_1, primals_183);  reshape_default_1 = primals_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        permute_default: "f32[768, 768]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_tensor: "f32[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_tensor, 8.0);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        reshape_default_2: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_2);  add_tensor_1 = _shape_param_2 = None
        permute_default_1: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(reshape_default_2, [1, 0, 2, 3]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:774 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_default_2: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_default_1, [0, 2, 1, 3]);  permute_default_1 = None
        reshape_default_3: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_default_2, _shape_param_3);  permute_default_2 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:706 in _chunk, code: hidden_states = hidden_states.view(
        reshape_default_4: "f32[96, 2, 512, 64]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_4);  reshape_default_3 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_default: "f32[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(reshape_default_4, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_default: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_default, 4);  as_strided_default = None
        permute_default_3: "f32[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_default, [0, 1, 4, 2, 3]);  unsqueeze_default = None
        reshape_default_5: "f32[1024, 8, 12, 64]" = torch.ops.aten.reshape.default(div_tensor, _shape_param_5);  div_tensor = _shape_param_5 = None
        permute_default_4: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(reshape_default_5, [1, 0, 2, 3]);  reshape_default_5 = None
        permute_default_5: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_default_4, [0, 2, 1, 3]);  permute_default_4 = None
        reshape_default_6: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(permute_default_5, _shape_param_6);  permute_default_5 = _shape_param_6 = None
        reshape_default_7: "f32[96, 2, 512, 64]" = torch.ops.aten.reshape.default(reshape_default_6, _shape_param_7);  reshape_default_6 = _shape_param_7 = None
        as_strided_default_1: "f32[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(reshape_default_7, [96, 3, 512, 64], [64, 1572864, 6144, 1]);  reshape_default_7 = None
        unsqueeze_default_1: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_default_1, 4);  as_strided_default_1 = None
        clone_default: "f32[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_default_1, memory_format = torch.contiguous_format);  unsqueeze_default_1 = None
        reshape_default_8: "f32[288, 512, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_8);  clone_default = _shape_param_8 = None
        permute_default_6: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_default_3, [0, 1, 4, 3, 2]);  permute_default_3 = None
        clone_default_1: "f32[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_default_6, memory_format = torch.contiguous_format);  permute_default_6 = None
        reshape_default_9: "f32[288, 64, 512]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_9);  clone_default_1 = _shape_param_9 = None
        return (permute_default, reshape_default_8, reshape_default_9)


def _default_make_inputs():
    return [
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [1024, 8, 768],  # _shape_param_0
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [1024, 8, 768],  # _shape_param_1
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [1024, 8, 12, 64],  # _shape_param_2
    [96, 1024, 64],  # _shape_param_3
    [96, 2, 512, 64],  # _shape_param_4
    [1024, 8, 12, 64],  # _shape_param_5
    [96, 1024, 64],  # _shape_param_6
    [96, 2, 512, 64],  # _shape_param_7
    [288, 512, 64],  # _shape_param_8
    [288, 64, 512],  # _shape_param_9
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
