"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_train
Pattern hash: 1cdd17891b1e
Shape hash: 03453655
"""
_shapes_config = "(T([4096, 512], f32), T([84], i64), T([32, 128, 512], f32), T([512], f32), T([384, 512], f32), T([384, 512], f32), T([32, 128, 512], f32), S([32, 128, 512]), S([4096, 512]), S([4096, 512]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_136: "f32[4096, 512]", inductor_seeds_default: "i64[84]", add_142: "f32[32, 128, 512]", primals_182: "f32[512]", primals_183: "f32[384, 512]", primals_184: "f32[384, 512]", mul_143: "f32[32, 128, 512]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        reshape_default: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_136, _shape_param_0);  mm_136 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 78);  inductor_seeds_default = None
        inductor_random_default: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default([32, 128, 512], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default);  gt_scalar = reshape_default = None
        mul_tensor_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        add_tensor: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_142, mul_tensor_1);  add_142 = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor, 2)
        mean_dim: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor_1: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_2: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor, rsqrt_default);  add_tensor = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_3: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(primals_182, mul_tensor_2);  primals_182 = mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default: "f32[512, 384]" = torch.ops.aten.permute.default(primals_183, [1, 0]);  primals_183 = None
        reshape_default_1: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_tensor_3, _shape_param_1);  mul_tensor_3 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_default_1: "f32[512, 384]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
        reshape_default_2: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_143, _shape_param_2);  mul_143 = _shape_param_2 = None
        return (permute_default, reshape_default_1, permute_default_1, reshape_default_2)



def make_inputs():
    return [
    torch.randn([4096, 512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 84, [84], dtype=torch.int64, device='cuda'),
    torch.randn([32, 128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512], dtype=torch.float32, device='cuda'),
    torch.randn([384, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 512], dtype=torch.float32, device='cuda'),
    [32, 128, 512],  # _shape_param_0
    [4096, 512],  # _shape_param_1
    [4096, 512],  # _shape_param_2
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
