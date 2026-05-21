"""
Standalone repro captured via capture_hook.
Label: hf_BlenderbotForCausalLM_train
Pattern hash: 7b93b154dd3c
Shape hash: 030f3ff7
"""
_shapes_config = "(T([4096, 2560], f32), T([4096, 2560], f32), T([4096, 2560], f32), T([32, 1, 128, 128], b8), S([32, 128, 2560]), S([32, 128, -1, 80]), S([32, 128, 2560]), S([32, 128, 2560]), S([32, 128, -1, 80]), S([32, 128, -1, 80]), S([32, 32, 128, 128]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm: "f32[4096, 2560]", addmm_1: "f32[4096, 2560]", addmm_2: "f32[4096, 2560]", primals_10: "b8[32, 1, 128, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f32[32, 128, 2560]" = torch.ops.aten.reshape.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        reshape_default_1: "f32[32, 128, 32, 80]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[32, 32, 128, 80]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_2: "f32[32, 128, 2560]" = torch.ops.aten.reshape.default(addmm_1, _shape_param_2);  addmm_1 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_3: "f32[32, 128, 2560]" = torch.ops.aten.reshape.default(addmm_2, _shape_param_3);  addmm_2 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        reshape_default_4: "f32[32, 128, 32, 80]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_4);  reshape_default_2 = _shape_param_4 = None
        permute_default_1: "f32[32, 32, 128, 80]" = torch.ops.aten.permute.default(reshape_default_4, [0, 2, 1, 3]);  reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        reshape_default_5: "f32[32, 128, 32, 80]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_5);  reshape_default_3 = _shape_param_5 = None
        permute_default_2: "f32[32, 32, 128, 80]" = torch.ops.aten.permute.default(reshape_default_5, [0, 2, 1, 3]);  reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(primals_10, full_default_1, full_default);  primals_10 = full_default_1 = full_default = None
        expand_default: "f32[32, 32, 128, 128]" = torch.ops.aten.expand.default(where_self, _shape_param_6);  where_self = _shape_param_6 = None
        return (permute_default, permute_default_1, permute_default_2, expand_default)



def make_inputs():
    return [
    torch.randn([4096, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 2560], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [32, 1, 128, 128], dtype=torch.bool, device='cuda'),
    [32, 128, 2560],  # _shape_param_0
    [32, 128, -1, 80],  # _shape_param_1
    [32, 128, 2560],  # _shape_param_2
    [32, 128, 2560],  # _shape_param_3
    [32, 128, -1, 80],  # _shape_param_4
    [32, 128, -1, 80],  # _shape_param_5
    [32, 32, 128, 128],  # _shape_param_6
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
