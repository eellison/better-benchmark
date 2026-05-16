"""
Standalone repro captured via capture_hook.
Label: vit_b_16_training
Pattern hash: e47bf40e6193
Shape hash: da2407bb
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_12: "f32[4, 768]", primals_150: "f32[768]", mul_86: "f32[4, 197, 768]", div: "f32[4, 197, 1]", _shape_param_0, primals_148: "f32[768, 3072]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:301 in forward, code: x = x[:, 0]
        full_default: "f32[4, 197, 768]" = torch.ops.aten.full.default([4, 197, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_default: "f32[4, 197, 768]" = torch.ops.aten.select_scatter.default(full_default, mm_12, 1, 0);  full_default = mm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:157 in forward, code: return self.ln(self.layers(self.dropout(input)))
        mul_tensor: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(select_scatter_default, primals_150);  select_scatter_default = primals_150 = None
        mul_tensor_1: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[4, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_86);  mul_tensor = None
        sum_dim_int_list_1: "f32[4, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(mul_86, sum_dim_int_list_1);  mul_86 = sum_dim_int_list_1 = None
        sub_tensor: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(div, sub_tensor_1);  div = sub_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:118 in forward, code: y = self.mlp(y)
        reshape_default: "f32[788, 768]" = torch.ops.aten.reshape.default(mul_tensor_4, _shape_param_0);  mul_tensor_4 = _shape_param_0 = None
        permute_default: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
        permute_default_1: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default, permute_default_1)



def make_inputs():
    return [
    torch.randn([4, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn(605184, dtype=torch.float32, device='cuda').as_strided([4, 197, 768], [768, 3072, 1]),  # mul_86
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    [788, 768],  # _shape_param_0
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
