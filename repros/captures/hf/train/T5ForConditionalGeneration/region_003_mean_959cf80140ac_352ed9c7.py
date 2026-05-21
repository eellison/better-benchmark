"""
Standalone repro captured via capture_hook.
Label: hf_T5ForConditionalGeneration_train
Pattern hash: 959cf80140ac
Shape hash: 352ed9c7
"""
_shapes_config = "(T([8192, 512], f32), T([64], i64), T([8, 1024, 512], f32), T([512], f32), T([2048, 512], f32), S([8, 1024, 512]), S([8192, 512]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_93: "f32[8192, 512]", inductor_seeds_default: "i64[64]", add_89: "f32[8, 1024, 512]", primals_130: "f32[512]", primals_131: "f32[2048, 512]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        reshape_default: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_93, _shape_param_0);  mm_93 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 60);  inductor_seeds_default = None
        inductor_random_default: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default);  gt_scalar = reshape_default = None
        mul_tensor_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        add_tensor: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_89, mul_tensor_1);  add_89 = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor, 2)
        mean_dim: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor_1: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_2: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor, rsqrt_default);  add_tensor = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_3: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(primals_130, mul_tensor_2);  primals_130 = mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        permute_default: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
        reshape_default_1: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_tensor_3, _shape_param_1);  mul_tensor_3 = _shape_param_1 = None
        return (permute_default, reshape_default_1)



def make_inputs():
    return [
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 64, [64], dtype=torch.int64, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    [8, 1024, 512],  # _shape_param_0
    [8192, 512],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
