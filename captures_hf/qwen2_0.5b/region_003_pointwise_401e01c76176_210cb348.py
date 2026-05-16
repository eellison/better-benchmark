"""
Standalone repro captured via capture_hook.
Label: qwen2_0.5b
Pattern hash: 401e01c76176
Shape hash: 210cb348
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_69: "f16[2048, 896]", _shape_param_0, _shape_param_1, convert_element_type_2: "f16[1, 512, 64]", convert_element_type_3: "f16[1, 512, 64]", addmm_70: "f16[2048, 128]", _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, addmm_71: "f16[2048, 128]", _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, expand: "b8[4, 1, 512, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f16[4, 512, 896]" = torch.ops.aten.reshape.default(addmm_69, _shape_param_0);  addmm_69 = _shape_param_0 = None
        reshape_default_1: "f16[4, 512, 14, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f16[4, 14, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:142 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze_default: "f16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1);  convert_element_type_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:144 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor: "f16[4, 14, 512, 64]" = torch.ops.aten.mul.Tensor(permute_default, unsqueeze_default)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:119 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_tensor: "f16[4, 14, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:120 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_default: "f16[4, 14, 512, 32]" = torch.ops.aten.neg.default(slice_tensor);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:118 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_tensor_1: "f16[4, 14, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default, 3, 0, 32);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:120 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_default: "f16[4, 14, 512, 64]" = torch.ops.aten.cat.default([neg_default, slice_tensor_1], -1);  neg_default = slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:143 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_default_1: "f16[1, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1);  convert_element_type_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:144 in apply_rotary_pos_emb, code: q_embed = (q * cos) + (rotate_half(q) * sin)
        mul_tensor_1: "f16[4, 14, 512, 64]" = torch.ops.aten.mul.Tensor(cat_default, unsqueeze_default_1);  cat_default = None
        add_tensor: "f16[4, 14, 512, 64]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:218 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_2: "f16[4, 512, 128]" = torch.ops.aten.reshape.default(addmm_70, _shape_param_2);  addmm_70 = _shape_param_2 = None
        reshape_default_3: "f16[4, 512, 2, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_1: "f16[4, 2, 512, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:145 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor_2: "f16[4, 2, 512, 64]" = torch.ops.aten.mul.Tensor(permute_default_1, unsqueeze_default);  unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:119 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_tensor_2: "f16[4, 2, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 32, 9223372036854775807)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:120 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_default_1: "f16[4, 2, 512, 32]" = torch.ops.aten.neg.default(slice_tensor_2);  slice_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:118 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_tensor_3: "f16[4, 2, 512, 32]" = torch.ops.aten.slice.Tensor(permute_default_1, 3, 0, 32);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:120 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        cat_default_1: "f16[4, 2, 512, 64]" = torch.ops.aten.cat.default([neg_default_1, slice_tensor_3], -1);  neg_default_1 = slice_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:145 in apply_rotary_pos_emb, code: k_embed = (k * cos) + (rotate_half(k) * sin)
        mul_tensor_3: "f16[4, 2, 512, 64]" = torch.ops.aten.mul.Tensor(cat_default_1, unsqueeze_default_1);  cat_default_1 = unsqueeze_default_1 = None
        add_tensor_1: "f16[4, 2, 512, 64]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_default_2: "f16[4, 2, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 2);  add_tensor_1 = None
        expand_default: "f16[4, 2, 7, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_default_2, _shape_param_4);  unsqueeze_default_2 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_default: "f16[4, 2, 7, 512, 64]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_4: "f16[4, 14, 512, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_5);  clone_default = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:219 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_5: "f16[4, 512, 128]" = torch.ops.aten.reshape.default(addmm_71, _shape_param_6);  addmm_71 = _shape_param_6 = None
        reshape_default_6: "f16[4, 512, 2, 64]" = torch.ops.aten.reshape.default(reshape_default_5, _shape_param_7);  reshape_default_5 = _shape_param_7 = None
        permute_default_2: "f16[4, 2, 512, 64]" = torch.ops.aten.permute.default(reshape_default_6, [0, 2, 1, 3]);  reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:24 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_default_3: "f16[4, 2, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(permute_default_2, 2);  permute_default_2 = None
        expand_default_1: "f16[4, 2, 7, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_default_3, _shape_param_8);  unsqueeze_default_3 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:25 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_default_1: "f16[4, 2, 7, 512, 64]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_7: "f16[4, 14, 512, 64]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_9);  clone_default_1 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default, full_default_1);  expand = full_default = full_default_1 = None
        return (add_tensor, reshape_default_4, reshape_default_7, where_self)



def make_inputs():
    return [
    torch.randn([2048, 896], dtype=torch.float16, device='cuda'),
    [4, 512, 896],  # _shape_param_0
    [4, 512, -1, 64],  # _shape_param_1
    torch.randn([1, 512, 64], dtype=torch.float16, device='cuda'),
    torch.randn([1, 512, 64], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 128], dtype=torch.float16, device='cuda'),
    [4, 512, 128],  # _shape_param_2
    [4, 512, -1, 64],  # _shape_param_3
    [4, 2, 7, 512, 64],  # _shape_param_4
    [4, 14, 512, 64],  # _shape_param_5
    torch.randn([2048, 128], dtype=torch.float16, device='cuda'),
    [4, 512, 128],  # _shape_param_6
    [4, 512, -1, 64],  # _shape_param_7
    [4, 2, 7, 512, 64],  # _shape_param_8
    [4, 14, 512, 64],  # _shape_param_9
    torch.randint(0, 2, [4, 1, 512, 512], dtype=torch.bool, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
