"""
Standalone repro captured via capture_hook.
Label: mistral_7b
Pattern hash: a3a7fba6c952
Shape hash: 7e37e3bf
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_216: "f16[1024, 4096]", _shape_param_0, add_216: "f16[4, 256, 4096]", arg283_1: "f16[4096]", _shape_param_1, arg284_1: "f16[4096, 4096]", _shape_param_2, arg285_1: "f16[1024, 4096]", _shape_param_3, arg286_1: "f16[1024, 4096]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        reshape_default: "f16[4, 256, 4096]" = torch.ops.aten.reshape.default(mm_216, _shape_param_0);  mm_216 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:239 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "f16[4, 256, 4096]" = torch.ops.aten.add.Tensor(add_216, reshape_default);  add_216 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:193 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default: "f32[4, 256, 4096]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:194 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[4, 256, 4096]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2)
        mean_dim: "f32[4, 256, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:195 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor_1: "f32[4, 256, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-05);  mean_dim = None
        rsqrt_default: "f32[4, 256, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[4, 256, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_default, rsqrt_default);  convert_element_type_default = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:196 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_1: "f16[4, 256, 4096]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float16);  mul_tensor = None
        mul_tensor_1: "f16[4, 256, 4096]" = torch.ops.aten.mul.Tensor(arg283_1, convert_element_type_default_1);  arg283_1 = convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:150 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_1: "f16[1024, 4096]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_1);  _shape_param_1 = None
        permute_default: "f16[4096, 4096]" = torch.ops.aten.permute.default(arg284_1, [1, 0]);  arg284_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:151 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_2: "f16[1024, 4096]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_2);  _shape_param_2 = None
        permute_default_1: "f16[4096, 1024]" = torch.ops.aten.permute.default(arg285_1, [1, 0]);  arg285_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mistral/modeling_mistral.py:152 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_3: "f16[1024, 4096]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_3);  mul_tensor_1 = _shape_param_3 = None
        permute_default_2: "f16[4096, 1024]" = torch.ops.aten.permute.default(arg286_1, [1, 0]);  arg286_1 = None
        return (reshape_default_1, permute_default, reshape_default_2, permute_default_1, reshape_default_3, permute_default_2)



def make_inputs():
    return [
    torch.randn([1024, 4096], dtype=torch.float16, device='cuda'),
    [4, 256, 4096],  # _shape_param_0
    torch.randn([4, 256, 4096], dtype=torch.float16, device='cuda'),
    torch.randn([4096], dtype=torch.float16, device='cuda'),
    [1024, 4096],  # _shape_param_1
    torch.randn([4096, 4096], dtype=torch.float16, device='cuda'),
    [1024, 4096],  # _shape_param_2
    torch.randn([1024, 4096], dtype=torch.float16, device='cuda'),
    [1024, 4096],  # _shape_param_3
    torch.randn([1024, 4096], dtype=torch.float16, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
