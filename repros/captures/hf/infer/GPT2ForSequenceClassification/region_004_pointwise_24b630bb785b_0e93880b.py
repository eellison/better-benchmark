"""
Standalone repro captured via capture_hook.
Label: hf_GPT2ForSequenceClassification_infer
Pattern hash: 24b630bb785b
Shape hash: 0e93880b
"""
_shapes_config = "(T([8192, 2304], f32), T([8, 1, 1024, 1024], b8), S([8, 1024, 2304]), S([8, 1024, -1, 64]), S([8, 1024, -1, 64]), S([8, 1024, -1, 64]), S([8, 12, 1024, 1024]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_44: "f32[8192, 2304]", expand_1: "b8[8, 1, 1024, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        reshape_default: "f32[8, 1024, 2304]" = torch.ops.aten.reshape.default(addmm_44, _shape_param_0);  addmm_44 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_tensor = torch.ops.aten.split.Tensor(reshape_default, 768, 2);  reshape_default = None
        getitem: "f32[8, 1024, 768]" = split_tensor[0]
        getitem_1: "f32[8, 1024, 768]" = split_tensor[1]
        getitem_2: "f32[8, 1024, 768]" = split_tensor[2];  split_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        reshape_default_1: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem, _shape_param_1);  getitem = _shape_param_1 = None
        permute_default: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        reshape_default_2: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_1, _shape_param_2);  getitem_1 = _shape_param_2 = None
        permute_default_1: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        reshape_default_3: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(getitem_2, _shape_param_3);  getitem_2 = _shape_param_3 = None
        permute_default_2: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_1, full_default, full_default_1);  expand_1 = full_default = full_default_1 = None
        expand_default: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where_self, _shape_param_4);  where_self = _shape_param_4 = None
        return (permute_default, permute_default_1, permute_default_2, expand_default)



def make_inputs():
    return [
    torch.randn([8192, 2304], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 1, 1024, 1024], dtype=torch.bool, device='cuda'),
    [8, 1024, 2304],  # _shape_param_0
    [8, 1024, -1, 64],  # _shape_param_1
    [8, 1024, -1, 64],  # _shape_param_2
    [8, 1024, -1, 64],  # _shape_param_3
    [8, 12, 1024, 1024],  # _shape_param_4
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
