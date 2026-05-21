"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_infer
Pattern hash: bb5163e87aa4
Shape hash: 246b40a7
"""
_shapes_config = "(T([250112, 512], f32), T([32, 128], i64), T([512], f32), T([384, 512], f32), T([384, 512], f32), S([4096, 512]), S([4096, 512]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[250112, 512]", arg0_1: "i64[32, 128]", arg2_1: "f32[512]", arg3_1: "f32[384, 512]", arg4_1: "f32[384, 512]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:680 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding_default: "f32[32, 128, 512]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(embedding_default, 2)
        mean_dim: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(embedding_default, rsqrt_default);  embedding_default = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg2_1, mul_tensor);  arg2_1 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_0);  _shape_param_0 = None
        permute_default: "f32[512, 384]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_1: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_1);  mul_tensor_1 = _shape_param_1 = None
        permute_default_1: "f32[512, 384]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        return (reshape_default, permute_default, reshape_default_1, permute_default_1)



def make_inputs():
    return [
    torch.randn([250112, 512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 250112, [32, 128], dtype=torch.int64, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512], dtype=torch.float32, device='cuda'),
    [4096, 512],  # _shape_param_0
    [4096, 512],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
