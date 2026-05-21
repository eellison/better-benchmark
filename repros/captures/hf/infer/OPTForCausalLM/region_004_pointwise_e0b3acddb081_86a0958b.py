"""
Standalone repro captured via capture_hook.
Label: hf_OPTForCausalLM_infer
Pattern hash: e0b3acddb081
Shape hash: 86a0958b
"""
_shapes_config = "(T([8192, 768], f32), T([8192, 768], f32), T([8192, 768], f32), T([4, 1, 2048, 2048], b8), S([4, 2048, 768]), S([4, -1, 12, 64]), S([4, 2048, 768]), S([4, -1, 12, 64]), S([4, 2048, 768]), S([4, -1, 12, 64]), S([4, 12, 2048, 2048]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_66: "f32[8192, 768]", addmm_67: "f32[8192, 768]", addmm_68: "f32[8192, 768]", expand: "b8[4, 1, 2048, 2048]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:151 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        reshape_default: "f32[4, 2048, 768]" = torch.ops.aten.reshape.default(addmm_66, _shape_param_0);  addmm_66 = _shape_param_0 = None
        mul_tensor: "f32[4, 2048, 768]" = torch.ops.aten.mul.Tensor(reshape_default, 0.125);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:152 in forward, code: query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        reshape_default_1: "f32[4, 2048, 12, 64]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_1);  mul_tensor = _shape_param_1 = None
        permute_default: "f32[4, 12, 2048, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:154 in forward, code: key_states = self.k_proj(hidden_states)
        reshape_default_2: "f32[4, 2048, 768]" = torch.ops.aten.reshape.default(addmm_67, _shape_param_2);  addmm_67 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:156 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        reshape_default_3: "f32[4, 2048, 12, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_1: "f32[4, 12, 2048, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:155 in forward, code: value_states = self.v_proj(hidden_states)
        reshape_default_4: "f32[4, 2048, 768]" = torch.ops.aten.reshape.default(addmm_68, _shape_param_4);  addmm_68 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:157 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        reshape_default_5: "f32[4, 2048, 12, 64]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_5);  reshape_default_4 = _shape_param_5 = None
        permute_default_2: "f32[4, 12, 2048, 64]" = torch.ops.aten.permute.default(reshape_default_5, [0, 2, 1, 3]);  reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[4, 1, 2048, 2048]" = torch.ops.aten.where.self(expand, full_default, full_default_1);  expand = full_default = full_default_1 = None
        expand_default: "f32[4, 12, 2048, 2048]" = torch.ops.aten.expand.default(where_self, _shape_param_6);  where_self = _shape_param_6 = None
        return (permute_default, permute_default_1, permute_default_2, expand_default)



def make_inputs():
    return [
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [4, 1, 2048, 2048], dtype=torch.bool, device='cuda'),
    [4, 2048, 768],  # _shape_param_0
    [4, -1, 12, 64],  # _shape_param_1
    [4, 2048, 768],  # _shape_param_2
    [4, -1, 12, 64],  # _shape_param_3
    [4, 2048, 768],  # _shape_param_4
    [4, -1, 12, 64],  # _shape_param_5
    [4, 12, 2048, 2048],  # _shape_param_6
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
