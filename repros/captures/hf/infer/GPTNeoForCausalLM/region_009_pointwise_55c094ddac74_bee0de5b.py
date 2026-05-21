"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForCausalLM_infer
Pattern hash: 55c094ddac74
Shape hash: bee0de5b
"""
_shapes_config = "(T([4096, 2048], f32), T([4096, 2048], f32), T([1, 128], i64), T([32, 128, 2048], f32), T([2048, 2048], f32), S([32, 128, 2048]), S([32, 128, 16, 128]), S([32, 16, 128, 128]), S([512, 128, 128]), S([32, 128, 2048]), S([32, 128, 16, 128]), S([32, 16, 128, 128]), S([512, 128, 128]), S([32, -1]), S([4096, 2048]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[4096, 2048]", mm_1: "f32[4096, 2048]", unsqueeze: "i64[1, 128]", add_5: "f32[32, 128, 2048]", arg7_1: "f32[2048, 2048]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        reshape_default: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        reshape_default_1: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_default: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        expand_default: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_default, _shape_param_2);  permute_default = _shape_param_2 = None
        clone_default: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_2: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        reshape_default_3: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_1, _shape_param_4);  mm_1 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        reshape_default_4: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_5);  reshape_default_3 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_default_1: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(reshape_default_4, [0, 2, 1, 3]);  reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        permute_default_2: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(permute_default_1, [0, 1, 3, 2]);  permute_default_1 = None
        expand_default_1: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_default_2, _shape_param_6);  permute_default_2 = _shape_param_6 = None
        clone_default_1: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_5: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_7);  clone_default_1 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:875 in _preprocess_mask_arguments, code: position_ids = position_ids.expand(batch_size, -1)
        expand_default_2: "i64[32, 128]" = torch.ops.aten.expand.default(unsqueeze, _shape_param_8);  unsqueeze = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:765 in find_packed_sequence_indices, code: first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
        slice_tensor: "i64[32, 1]" = torch.ops.aten.slice.Tensor(expand_default_2, 1, 0, 1)
        sub_tensor: "i64[32, 1]" = torch.ops.aten.sub.Tensor(slice_tensor, 1);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:766 in find_packed_sequence_indices, code: position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
        cat_default: "i64[32, 129]" = torch.ops.aten.cat.default([sub_tensor, expand_default_2], -1);  sub_tensor = expand_default_2 = None
        slice_tensor_1: "i64[32, 128]" = torch.ops.aten.slice.Tensor(cat_default, -1, 1, 129)
        slice_tensor_2: "i64[32, 128]" = torch.ops.aten.slice.Tensor(cat_default, -1, 0, 128);  cat_default = None
        sub_tensor_1: "i64[32, 128]" = torch.ops.aten.sub.Tensor(slice_tensor_1, slice_tensor_2);  slice_tensor_1 = slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:767 in find_packed_sequence_indices, code: packed_sequence_mask = (position_diff != 1).cumsum(-1)
        ne_scalar: "b8[32, 128]" = torch.ops.aten.ne.Scalar(sub_tensor_1, 1);  sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        reshape_default_6: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_5, _shape_param_9);  add_5 = _shape_param_9 = None
        permute_default_3: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        return (reshape_default_2, reshape_default_5, ne_scalar, reshape_default_6, permute_default_3)



def make_inputs():
    return [
    torch.randn([4096, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 2048], dtype=torch.float32, device='cuda'),
    torch.randint(0, 128, [1, 128], dtype=torch.int64, device='cuda'),
    torch.randn([32, 128, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 2048], dtype=torch.float32, device='cuda'),
    [32, 128, 2048],  # _shape_param_0
    [32, 128, 16, 128],  # _shape_param_1
    [32, 16, 128, 128],  # _shape_param_2
    [512, 128, 128],  # _shape_param_3
    [32, 128, 2048],  # _shape_param_4
    [32, 128, 16, 128],  # _shape_param_5
    [32, 16, 128, 128],  # _shape_param_6
    [512, 128, 128],  # _shape_param_7
    [32, -1],  # _shape_param_8
    [4096, 2048],  # _shape_param_9
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
