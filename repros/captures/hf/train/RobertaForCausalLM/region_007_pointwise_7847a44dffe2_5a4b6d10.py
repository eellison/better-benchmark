"""
Standalone repro captured via capture_hook.
Label: hf_RobertaForCausalLM_train
Pattern hash: 7847a44dffe2
Shape hash: 5a4b6d10
"""
_shapes_config = "(T([16384, 768], f32), T([16384, 768], f32), T([768, 768], f32), S([32, 512, 768]), S([32, 512, -1, 64]), S([32, 512, 768]), S([32, 512, -1, 64]), S([32, 12, 512, 64]), S([384, 512, 64]), S([32, 12, 64, 512]), S([384, 64, 512]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_66: "f32[16384, 768]", addmm_67: "f32[16384, 768]", primals_189: "f32[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:226 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_66, _shape_param_0);  addmm_66 = _shape_param_0 = None
        reshape_default_1: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:227 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_2: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(addmm_67, _shape_param_2);  addmm_67 = _shape_param_2 = None
        reshape_default_3: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_1: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/roberta/modeling_roberta.py:228 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_189, [1, 0]);  primals_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_scalar: "f32[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_default, 0.3535533905932738);  permute_default = None
        permute_default_3: "f32[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_default_1, [0, 1, 3, 2]);  permute_default_1 = None
        mul_scalar_1: "f32[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_3, 0.3535533905932738);  permute_default_3 = None
        expand_default: "f32[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_scalar, _shape_param_4);  mul_scalar = _shape_param_4 = None
        clone_default: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_4: "f32[384, 512, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_5);  clone_default = _shape_param_5 = None
        expand_default_1: "f32[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_1, _shape_param_6);  mul_scalar_1 = _shape_param_6 = None
        clone_default_1: "f32[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_5: "f32[384, 64, 512]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_7);  clone_default_1 = _shape_param_7 = None
        return (permute_default_2, reshape_default_4, reshape_default_5)



def make_inputs():
    return [
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [32, 512, 768],  # _shape_param_0
    [32, 512, -1, 64],  # _shape_param_1
    [32, 512, 768],  # _shape_param_2
    [32, 512, -1, 64],  # _shape_param_3
    [32, 12, 512, 64],  # _shape_param_4
    [384, 512, 64],  # _shape_param_5
    [32, 12, 64, 512],  # _shape_param_6
    [384, 64, 512],  # _shape_param_7
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
