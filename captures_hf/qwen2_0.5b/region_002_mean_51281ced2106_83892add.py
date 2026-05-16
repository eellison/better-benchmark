"""
Standalone repro captured via capture_hook.
Label: qwen2_0.5b
Pattern hash: 51281ced2106
Shape hash: 83892add
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_92: "f16[2048, 896]", _shape_param_0, add_163: "f16[4, 512, 896]", arg288_1: "f16[896]", _shape_param_1, arg289_1: "f16[4864, 896]", _shape_param_2, arg290_1: "f16[4864, 896]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default: "f16[4, 512, 896]" = torch.ops.aten.reshape.default(mm_92, _shape_param_0);  mm_92 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:302 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "f16[4, 512, 896]" = torch.ops.aten.add.Tensor(add_163, reshape_default);  add_163 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:260 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default: "f32[4, 512, 896]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:261 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[4, 512, 896]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2)
        mean_dim: "f32[4, 512, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor_1: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_default, rsqrt_default);  convert_element_type_default = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_1: "f16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float16);  mul_tensor = None
        mul_tensor_1: "f16[4, 512, 896]" = torch.ops.aten.mul.Tensor(arg288_1, convert_element_type_default_1);  arg288_1 = convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        reshape_default_1: "f16[2048, 896]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_1);  _shape_param_1 = None
        permute_default: "f16[896, 4864]" = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        reshape_default_2: "f16[2048, 896]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_2);  mul_tensor_1 = _shape_param_2 = None
        permute_default_1: "f16[896, 4864]" = torch.ops.aten.permute.default(arg290_1, [1, 0]);  arg290_1 = None
        return (reshape_default_1, permute_default, reshape_default_2, permute_default_1)



def make_inputs():
    return [
    torch.randn([2048, 896], dtype=torch.float16, device='cuda'),
    [4, 512, 896],  # _shape_param_0
    torch.randn([4, 512, 896], dtype=torch.float16, device='cuda'),
    torch.randn([896], dtype=torch.float16, device='cuda'),
    [2048, 896],  # _shape_param_1
    torch.randn([4864, 896], dtype=torch.float16, device='cuda'),
    [2048, 896],  # _shape_param_2
    torch.randn([4864, 896], dtype=torch.float16, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
