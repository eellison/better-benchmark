"""
Standalone repro captured via capture_hook.
Label: t5_base
Pattern hash: 14337b35d74b
Shape hash: b96385a3
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_185: "f16[512, 768]", _shape_param_0, add_164: "f32[4, 128, 768]", arg251_1: "f16[768]", _shape_param_1, arg252_1: "f16[768, 768]", mul_51: "f16[4, 512, 768]", _shape_param_2, arg253_1: "f16[768, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        reshape_default: "f16[4, 128, 768]" = torch.ops.aten.reshape.default(mm_185, _shape_param_0);  mm_185 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_tensor: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_164, reshape_default);  add_164 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[4, 128, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor, 2)
        mean_dim: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor_1: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[4, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_tensor, rsqrt_default);  add_tensor = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_default: "f16[4, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float16);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_1: "f16[4, 128, 768]" = torch.ops.aten.mul.Tensor(arg251_1, convert_element_type_default);  arg251_1 = convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_1: "f16[512, 768]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_1);  mul_tensor_1 = _shape_param_1 = None
        permute_default: "f16[768, 768]" = torch.ops.aten.permute.default(arg252_1, [1, 0]);  arg252_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_2: "f16[2048, 768]" = torch.ops.aten.reshape.default(mul_51, _shape_param_2);  mul_51 = _shape_param_2 = None
        permute_default_1: "f16[768, 768]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        return (reshape_default_1, permute_default, reshape_default_2, permute_default_1)



def make_inputs():
    return [
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    [4, 128, 768],  # _shape_param_0
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float16, device='cuda'),
    [512, 768],  # _shape_param_1
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 512, 768], dtype=torch.float16, device='cuda'),
    [2048, 768],  # _shape_param_2
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
