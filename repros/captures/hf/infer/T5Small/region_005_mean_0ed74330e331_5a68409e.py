"""
Standalone repro captured via capture_hook.
Label: hf_T5Small_infer
Pattern hash: 0ed74330e331
Shape hash: 5a68409e
"""
_shapes_config = "(T([8192, 512], f32), T([8, 1024, 512], f32), T([512], f32), T([512, 512], f32), T([8, 1024, 512], f32), T([512, 512], f32), T([512, 512], f32), S([8, 1024, 512]), S([8192, 512]), S([8192, 512]), S([8192, 512]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_89: "f32[8192, 512]", add_86: "f32[8, 1024, 512]", arg124_1: "f32[512]", arg125_1: "f32[512, 512]", mul_27: "f32[8, 1024, 512]", arg126_1: "f32[512, 512]", arg127_1: "f32[512, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        reshape_default: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_89, _shape_param_0);  mm_89 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_tensor: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_86, reshape_default);  add_86 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor, 2)
        mean_dim: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor_1: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor, rsqrt_default);  add_tensor = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg124_1, mul_tensor);  arg124_1 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_1: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_1);  mul_tensor_1 = _shape_param_1 = None
        permute_default: "f32[512, 512]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_2: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_27, _shape_param_2);  _shape_param_2 = None
        permute_default_1: "f32[512, 512]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_3: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_27, _shape_param_3);  mul_27 = _shape_param_3 = None
        permute_default_2: "f32[512, 512]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        return (reshape_default_1, permute_default, reshape_default_2, permute_default_1, reshape_default_3, permute_default_2)



def make_inputs():
    return [
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    [8, 1024, 512],  # _shape_param_0
    [8192, 512],  # _shape_param_1
    [8192, 512],  # _shape_param_2
    [8192, 512],  # _shape_param_3
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
