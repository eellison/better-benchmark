"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForCausalLM_inference
Pattern hash: ee3140d4b76d
Shape hash: 249ceb70
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[1024, 2048]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, mm_1: "f32[1024, 2048]", _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, unsqueeze: "i64[1, 128]", _shape_param_8, add_5: "f32[8, 128, 2048]", _shape_param_9, arg7_1: "f32[2048, 2048]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        reshape_default: "f32[8, 128, 2048]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        reshape_default_1: "f32[8, 128, 16, 128]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_default: "f32[8, 16, 128, 128]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_default: "f32[8, 16, 128, 128]" = torch.ops.aten.expand.default(permute_default, _shape_param_2);  permute_default = _shape_param_2 = None
        clone_default: "f32[8, 16, 128, 128]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_2: "f32[128, 128, 128]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        reshape_default_3: "f32[8, 128, 2048]" = torch.ops.aten.reshape.default(mm_1, _shape_param_4);  mm_1 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        reshape_default_4: "f32[8, 128, 16, 128]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_5);  reshape_default_3 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_default_1: "f32[8, 16, 128, 128]" = torch.ops.aten.permute.default(reshape_default_4, [0, 2, 1, 3]);  reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_2: "f32[8, 16, 128, 128]" = torch.ops.aten.permute.default(permute_default_1, [0, 1, 3, 2]);  permute_default_1 = None
        expand_default_1: "f32[8, 16, 128, 128]" = torch.ops.aten.expand.default(permute_default_2, _shape_param_6);  permute_default_2 = _shape_param_6 = None
        clone_default_1: "f32[8, 16, 128, 128]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_5: "f32[128, 128, 128]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_7);  clone_default_1 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:875 in _preprocess_mask_arguments, code: position_ids = position_ids.expand(batch_size, -1)
        expand_default_2: "i64[8, 128]" = torch.ops.aten.expand.default(unsqueeze, _shape_param_8);  unsqueeze = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:765 in find_packed_sequence_indices, code: first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
        slice_tensor: "i64[8, 1]" = torch.ops.aten.slice.Tensor(expand_default_2, 1, 0, 1)
        sub_tensor: "i64[8, 1]" = torch.ops.aten.sub.Tensor(slice_tensor, 1);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:766 in find_packed_sequence_indices, code: position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
        cat_default: "i64[8, 129]" = torch.ops.aten.cat.default([sub_tensor, expand_default_2], -1);  sub_tensor = expand_default_2 = None
        slice_tensor_1: "i64[8, 128]" = torch.ops.aten.slice.Tensor(cat_default, -1, 1, 129)
        slice_tensor_2: "i64[8, 128]" = torch.ops.aten.slice.Tensor(cat_default, -1, 0, 128);  cat_default = None
        sub_tensor_1: "i64[8, 128]" = torch.ops.aten.sub.Tensor(slice_tensor_1, slice_tensor_2);  slice_tensor_1 = slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:767 in find_packed_sequence_indices, code: packed_sequence_mask = (position_diff != 1).cumsum(-1)
        ne_scalar: "b8[8, 128]" = torch.ops.aten.ne.Scalar(sub_tensor_1, 1);  sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        reshape_default_6: "f32[1024, 2048]" = torch.ops.aten.reshape.default(add_5, _shape_param_9);  add_5 = _shape_param_9 = None
        permute_default_3: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        return (reshape_default_2, reshape_default_5, ne_scalar, reshape_default_6, permute_default_3)


def _default_make_inputs():
    return [
    torch.randn([1024, 2048], dtype=torch.float32, device='cuda'),
    [8, 128, 2048],  # _shape_param_0
    [8, 128, 16, 128],  # _shape_param_1
    [8, 16, 128, 128],  # _shape_param_2
    [128, 128, 128],  # _shape_param_3
    torch.randn([1024, 2048], dtype=torch.float32, device='cuda'),
    [8, 128, 2048],  # _shape_param_4
    [8, 128, 16, 128],  # _shape_param_5
    [8, 16, 128, 128],  # _shape_param_6
    [128, 128, 128],  # _shape_param_7
    torch.randint(0, 2, [1, 128], dtype=torch.int64, device='cuda'),
    [8, -1],  # _shape_param_8
    torch.randn([8, 128, 2048], dtype=torch.float32, device='cuda'),
    [1024, 2048],  # _shape_param_9
    torch.randn([2048, 2048], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
