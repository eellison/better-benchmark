"""
Standalone repro captured via capture_hook.
Label: bert_large
Pattern hash: d195a5865427
Shape hash: e40f32c1
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_138: "f16[2048, 1024]", _shape_param_0, _shape_param_1, addmm_139: "f16[2048, 1024]", _shape_param_2, _shape_param_3, addmm_140: "f16[2048, 1024]", _shape_param_4, _shape_param_5, expand: "b8[4, 1, 512, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default: "f16[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_138, _shape_param_0);  addmm_138 = _shape_param_0 = None
        reshape_default_1: "f16[4, 512, 16, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f16[4, 16, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_2: "f16[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_139, _shape_param_2);  addmm_139 = _shape_param_2 = None
        reshape_default_3: "f16[4, 512, 16, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_1: "f16[4, 16, 512, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_4: "f16[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_140, _shape_param_4);  addmm_140 = _shape_param_4 = None
        reshape_default_5: "f16[4, 512, 16, 64]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_5);  reshape_default_4 = _shape_param_5 = None
        permute_default_2: "f16[4, 16, 512, 64]" = torch.ops.aten.permute.default(reshape_default_5, [0, 2, 1, 3]);  reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f16[4, 1, 512, 512]" = torch.ops.aten.where.self(expand, full_default, full_default_1);  expand = full_default = full_default_1 = None
        return (permute_default, permute_default_1, permute_default_2, where_self)



def make_inputs():
    return [
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [4, 512, 1024],  # _shape_param_0
    [4, 512, -1, 64],  # _shape_param_1
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [4, 512, 1024],  # _shape_param_2
    [4, 512, -1, 64],  # _shape_param_3
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [4, 512, 1024],  # _shape_param_4
    [4, 512, -1, 64],  # _shape_param_5
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
