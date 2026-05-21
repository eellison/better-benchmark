"""
Standalone repro captured via capture_hook.
Label: hf_RobertaForCausalLM_train
Pattern hash: 56eb1ded572c
Shape hash: 6ee8d283
"""
_shapes_config = "(T([384, 512, 64], f32), T([384, 64, 512], f32), T([384, 512, 64], f32), T([768, 768], f32), T([768, 768], f32), T([768, 768], f32), S([32, 12, 512, 64]), S([32, 12, 64, 512]), S([32, 12, 512, 64]), S([32, 512, 768]), S([16384, 768]), S([32, 512, 768]), S([16384, 768]), S([32, 512, 768]), S([16384, 768]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, bmm_64: "f32[384, 512, 64]", bmm_66: "f32[384, 64, 512]", bmm_67: "f32[384, 512, 64]", primals_29: "f32[768, 768]", primals_27: "f32[768, 768]", primals_25: "f32[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_64, _shape_param_0);  bmm_64 = _shape_param_0 = None
        reshape_default_1: "f32[32, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_66, _shape_param_1);  bmm_66 = _shape_param_1 = None
        reshape_default_2: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_67, _shape_param_2);  bmm_67 = _shape_param_2 = None
        mul_scalar: "f32[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(reshape_default_1, 0.3535533905932738);  reshape_default_1 = None
        permute_default: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(mul_scalar, [0, 1, 3, 2]);  mul_scalar = None
        mul_scalar_1: "f32[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(reshape_default_2, 0.3535533905932738);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:228 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_1: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_3: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        reshape_default_4: "f32[16384, 768]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_4);  reshape_default_3 = _shape_param_4 = None
        permute_default_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
        permute_default_3: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:227 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_4: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        reshape_default_5: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(permute_default_4, _shape_param_5);  permute_default_4 = _shape_param_5 = None
        clone_default_1: "f32[32, 512, 768]" = torch.ops.aten.clone.default(reshape_default_5, memory_format = torch.contiguous_format);  reshape_default_5 = None
        reshape_default_6: "f32[16384, 768]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_6);  clone_default_1 = _shape_param_6 = None
        permute_default_5: "f32[768, 768]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
        permute_default_6: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_5, [1, 0]);  permute_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:226 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_7: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(mul_scalar_1, [0, 2, 1, 3]);  mul_scalar_1 = None
        clone_default_2: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default_7, memory_format = torch.contiguous_format);  permute_default_7 = None
        reshape_default_7: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_7);  clone_default_2 = _shape_param_7 = None
        reshape_default_8: "f32[16384, 768]" = torch.ops.aten.reshape.default(reshape_default_7, _shape_param_8);  reshape_default_7 = _shape_param_8 = None
        permute_default_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
        permute_default_9: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_8, [1, 0]);  permute_default_8 = None
        return (reshape_default_4, permute_default_3, reshape_default_6, permute_default_6, reshape_default_8, permute_default_9)



def make_inputs():
    return [
    torch.randn([384, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([384, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [32, 12, 512, 64],  # _shape_param_0
    [32, 12, 64, 512],  # _shape_param_1
    [32, 12, 512, 64],  # _shape_param_2
    [32, 512, 768],  # _shape_param_3
    [16384, 768],  # _shape_param_4
    [32, 512, 768],  # _shape_param_5
    [16384, 768],  # _shape_param_6
    [32, 512, 768],  # _shape_param_7
    [16384, 768],  # _shape_param_8
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
