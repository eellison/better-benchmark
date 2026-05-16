"""
Standalone repro captured via capture_hook.
Label: hf_Reformer_inference
Pattern hash: b2d5672f28f2
Shape hash: 426cedf5
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[32768, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, mm_1: "f32[32768, 768]", _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, add_1: "f32[8, 4096, 256]", _shape_param_10, arg5_1: "f32[768, 256]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1162 in forward, code: query_vectors = self.query(hidden_states)
        reshape_default: "f32[8, 4096, 768]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:380 in _split_hidden_size_dim, code: x = x.view(*new_x_shape)
        reshape_default_1: "f32[8, 4096, 12, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:381 in _split_hidden_size_dim, code: return x.transpose(2, 1)
        permute_default: "f32[8, 12, 4096, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        reshape_default_2: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(permute_default, _shape_param_2);  permute_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1236 in forward, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        expand_default: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.expand.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        clone_default: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_3: "f32[6144, 64, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1163 in forward, code: key_vectors = self.key(hidden_states)
        reshape_default_4: "f32[8, 4096, 768]" = torch.ops.aten.reshape.default(mm_1, _shape_param_5);  mm_1 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:380 in _split_hidden_size_dim, code: x = x.view(*new_x_shape)
        reshape_default_5: "f32[8, 4096, 12, 64]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_6);  reshape_default_4 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:381 in _split_hidden_size_dim, code: return x.transpose(2, 1)
        permute_default_1: "f32[8, 12, 4096, 64]" = torch.ops.aten.permute.default(reshape_default_5, [0, 2, 1, 3]);  reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1188 in forward, code: key_vectors = key_vectors / np.sqrt(self.attention_head_size)
        full_default: "f64[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_tensor: "f32[8, 12, 4096, 64]" = torch.ops.aten.div.Tensor(permute_default_1, full_default);  permute_default_1 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:398 in _split_seq_length_dim_to, code: return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
        reshape_default_6: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.reshape.default(div_tensor, _shape_param_7);  div_tensor = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:372 in _look_adjacent, code: slices.append(torch.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], dim=2))
        slice_tensor: "f32[8, 12, 1, 64, 64]" = torch.ops.aten.slice.Tensor(reshape_default_6, 2, -1, 9223372036854775807)
        slice_tensor_1: "f32[8, 12, 63, 64, 64]" = torch.ops.aten.slice.Tensor(reshape_default_6, 2, 0, -1)
        cat_default: "f32[8, 12, 64, 64, 64]" = torch.ops.aten.cat.default([slice_tensor, slice_tensor_1], 2);  slice_tensor = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:373 in _look_adjacent, code: return torch.cat(slices, dim=3)
        cat_default_1: "f32[8, 12, 64, 128, 64]" = torch.ops.aten.cat.default([cat_default, reshape_default_6], 3);  cat_default = reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1236 in forward, code: query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
        permute_default_2: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.permute.default(cat_default_1, [0, 1, 2, 4, 3]);  cat_default_1 = None
        expand_default_1: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.expand.default(permute_default_2, _shape_param_8);  permute_default_2 = _shape_param_8 = None
        reshape_default_7: "f32[6144, 64, 128]" = torch.ops.aten.reshape.default(expand_default_1, _shape_param_9);  expand_default_1 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1164 in forward, code: value_vectors = self.value(hidden_states)
        reshape_default_8: "f32[32768, 256]" = torch.ops.aten.reshape.default(add_1, _shape_param_10);  add_1 = _shape_param_10 = None
        permute_default_3: "f32[256, 768]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        return (reshape_default_3, reshape_default_7, reshape_default_8, permute_default_3)


def _default_make_inputs():
    return [
    torch.randn([32768, 768], dtype=torch.float32, device='cuda'),
    [8, 4096, 768],  # _shape_param_0
    [8, 4096, 12, 64],  # _shape_param_1
    [8, 12, 64, 64, 64],  # _shape_param_2
    [8, 12, 64, 64, 64],  # _shape_param_3
    [6144, 64, 64],  # _shape_param_4
    torch.randn([32768, 768], dtype=torch.float32, device='cuda'),
    [8, 4096, 768],  # _shape_param_5
    [8, 4096, 12, 64],  # _shape_param_6
    [8, 12, 64, 64, 64],  # _shape_param_7
    [8, 12, 64, 64, 128],  # _shape_param_8
    [6144, 64, 128],  # _shape_param_9
    torch.randn([8, 4096, 256], dtype=torch.float32, device='cuda'),
    [32768, 256],  # _shape_param_10
    torch.randn([768, 256], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
