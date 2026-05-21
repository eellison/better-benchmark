"""
Standalone repro captured via capture_hook.
Label: hf_DistilBertForMaskedLM_infer
Pattern hash: 8e9a93fe0b5e
Shape hash: e0035305
"""
_shapes_config = "(T([32768, 768], f32), T([32768, 768], f32), T([256, 128, 768], f32), T([768, 768], f32), S([256, 128, 768]), S([256, 128, -1, 64]), S([256, 12, 128, 64]), S([3072, 128, 64]), S([256, 128, 768]), S([256, 128, -1, 64]), S([256, 12, 64, 128]), S([3072, 64, 128]), S([32768, 768]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_30: "f32[32768, 768]", addmm_31: "f32[32768, 768]", add_44: "f32[256, 128, 768]", arg90_1: "f32[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:187 in forward, code: query_layer = self.q_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(addmm_30, _shape_param_0);  addmm_30 = _shape_param_0 = None
        reshape_default_1: "f32[256, 128, 12, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[256, 12, 128, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_scalar: "f32[256, 12, 128, 64]" = torch.ops.aten.mul.Scalar(permute_default, 0.3535533905932738);  permute_default = None
        expand_default: "f32[256, 12, 128, 64]" = torch.ops.aten.expand.default(mul_scalar, _shape_param_2);  mul_scalar = _shape_param_2 = None
        clone_default: "f32[256, 12, 128, 64]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_2: "f32[3072, 128, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:188 in forward, code: key_layer = self.k_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_3: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(addmm_31, _shape_param_4);  addmm_31 = _shape_param_4 = None
        reshape_default_4: "f32[256, 128, 12, 64]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_5);  reshape_default_3 = _shape_param_5 = None
        permute_default_1: "f32[256, 12, 128, 64]" = torch.ops.aten.permute.default(reshape_default_4, [0, 2, 1, 3]);  reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_default_2: "f32[256, 12, 64, 128]" = torch.ops.aten.permute.default(permute_default_1, [0, 1, 3, 2]);  permute_default_1 = None
        mul_scalar_1: "f32[256, 12, 64, 128]" = torch.ops.aten.mul.Scalar(permute_default_2, 0.3535533905932738);  permute_default_2 = None
        expand_default_1: "f32[256, 12, 64, 128]" = torch.ops.aten.expand.default(mul_scalar_1, _shape_param_6);  mul_scalar_1 = _shape_param_6 = None
        clone_default_1: "f32[256, 12, 64, 128]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_5: "f32[3072, 64, 128]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_7);  clone_default_1 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:189 in forward, code: value_layer = self.v_lin(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_6: "f32[32768, 768]" = torch.ops.aten.reshape.default(add_44, _shape_param_8);  add_44 = _shape_param_8 = None
        permute_default_3: "f32[768, 768]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        return (reshape_default_2, reshape_default_5, reshape_default_6, permute_default_3)



def make_inputs():
    return [
    torch.randn([32768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([256, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [256, 128, 768],  # _shape_param_0
    [256, 128, -1, 64],  # _shape_param_1
    [256, 12, 128, 64],  # _shape_param_2
    [3072, 128, 64],  # _shape_param_3
    [256, 128, 768],  # _shape_param_4
    [256, 128, -1, 64],  # _shape_param_5
    [256, 12, 64, 128],  # _shape_param_6
    [3072, 64, 128],  # _shape_param_7
    [32768, 768],  # _shape_param_8
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
