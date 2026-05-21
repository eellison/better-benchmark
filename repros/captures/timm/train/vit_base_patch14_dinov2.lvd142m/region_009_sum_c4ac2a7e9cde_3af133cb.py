"""
Standalone repro captured via capture_hook.
Label: timm_vit_base_patch14_dinov2.lvd142m_train
Pattern hash: c4ac2a7e9cde
Shape hash: 3af133cb
"""
_shapes_config = "(T([175360, 768], f32), T([768], f32), T([128, 1370, 768], f32), T([128, 1370, 1], f32, stride=(1376, 1, 1)), T([128, 1370, 768], f32), T([768], f32), T([768, 768], f32), S([128, 1370, 768]), S([175360, 768]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_90: "f32[175360, 768]", primals_13: "f32[768]", mul_3: "f32[128, 1370, 768]", div_23: "f32[128, 1370, 1]", add_130: "f32[128, 1370, 768]", primals_12: "f32[768]", primals_10: "f32[768, 768]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default: "f32[128, 1370, 768]" = torch.ops.aten.reshape.default(mm_90, _shape_param_0);  mm_90 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(reshape_default, primals_13);  reshape_default = primals_13 = None
        mul_tensor_1: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[128, 1370, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_3);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 1370, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(mul_3, sum_dim_int_list_1);  mul_3 = sum_dim_int_list_1 = None
        sub_tensor: "f32[128, 1370, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[128, 1370, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(div_23, sub_tensor_1);  div_23 = sub_tensor_1 = None
        add_tensor: "f32[128, 1370, 768]" = torch.ops.aten.add.Tensor(add_130, mul_tensor_4);  add_130 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_tensor_5: "f32[128, 1370, 768]" = torch.ops.aten.mul.Tensor(add_tensor, primals_12);  add_tensor = primals_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_1: "f32[175360, 768]" = torch.ops.aten.reshape.default(mul_tensor_5, _shape_param_1);  mul_tensor_5 = _shape_param_1 = None
        permute_default: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_1, permute_default_1)



def make_inputs():
    return [
    torch.randn([175360, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([128, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn(176122, dtype=torch.float32, device='cuda').as_strided([128, 1370, 1], [1376, 1, 1]),  # div_23
    torch.randn([128, 1370, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [128, 1370, 768],  # _shape_param_0
    [175360, 768],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
