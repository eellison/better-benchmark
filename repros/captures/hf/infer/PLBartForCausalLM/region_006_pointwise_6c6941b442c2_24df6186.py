"""
Standalone repro captured via capture_hook.
Label: hf_PLBartForCausalLM_infer
Pattern hash: 6c6941b442c2
Shape hash: 24df6186
"""
_shapes_config = "(T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), S([16, 1024, 768]), S([16, 1024, -1, 64]), S([16, 1024, 768]), S([16, 1024, -1, 64]), S([16, 1024, 768]), S([16, 1024, -1, 64]), S([16, -1, 1024, 1024]), S([16, 12, 1024, 1024]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm: "f32[16384, 768]", addmm_1: "f32[16384, 768]", addmm_2: "f32[16384, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:209 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f32[16, 1024, 768]" = torch.ops.aten.reshape.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        reshape_default_1: "f32[16, 1024, 12, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[16, 12, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:229 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_2: "f32[16, 1024, 768]" = torch.ops.aten.reshape.default(addmm_1, _shape_param_2);  addmm_1 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:232 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        reshape_default_3: "f32[16, 1024, 12, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_1: "f32[16, 12, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:230 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_4: "f32[16, 1024, 768]" = torch.ops.aten.reshape.default(addmm_2, _shape_param_4);  addmm_2 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/plbart/modeling_plbart.py:233 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        reshape_default_5: "f32[16, 1024, 12, 64]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_5);  reshape_default_4 = _shape_param_5 = None
        permute_default_2: "f32[16, 12, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_5, [0, 2, 1, 3]);  reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_default: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[1024]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_default: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None
        unsqueeze_default_1: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 1);  unsqueeze_default = None
        unsqueeze_default_2: "i64[1, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 2);  unsqueeze_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_default_1: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_1: "i64[1024]" = torch.ops.aten.add.Tensor(iota_default_1, 0);  iota_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_default_3: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add_tensor_1, 0);  add_tensor_1 = None
        unsqueeze_default_4: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 1);  unsqueeze_default_3 = None
        unsqueeze_default_5: "i64[1, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le_tensor: "b8[1, 1, 1024, 1024]" = torch.ops.aten.le.Tensor(unsqueeze_default_2, unsqueeze_default_5);  unsqueeze_default_2 = unsqueeze_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_default: "b8[16, 1, 1024, 1024]" = torch.ops.aten.expand.default(le_tensor, _shape_param_6);  le_tensor = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[16, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_default, full_default, full_default_1);  expand_default = full_default = full_default_1 = None
        expand_default_1: "f32[16, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_self, _shape_param_7);  where_self = _shape_param_7 = None
        return (permute_default, permute_default_1, permute_default_2, expand_default_1)



def make_inputs():
    return [
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    [16, 1024, 768],  # _shape_param_0
    [16, 1024, -1, 64],  # _shape_param_1
    [16, 1024, 768],  # _shape_param_2
    [16, 1024, -1, 64],  # _shape_param_3
    [16, 1024, 768],  # _shape_param_4
    [16, 1024, -1, 64],  # _shape_param_5
    [16, -1, 1024, 1024],  # _shape_param_6
    [16, 12, 1024, 1024],  # _shape_param_7
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
