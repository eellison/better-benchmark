"""
Standalone repro captured via capture_hook.
Label: timm_vit_base_patch16_siglip_256_train
Pattern hash: 36804d150a9a
Shape hash: 80de50c2
"""
_shapes_config = "(T([32768, 768], f32), T([768], f32), T([128, 256, 768], f32), T([128, 256, 1], f32), T([128, 256, 768], f32), T([768, 768], f32), S([128, 256, 768]), S([32768, 768]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_101: "f32[32768, 768]", primals_11: "f32[768]", mul_2: "f32[128, 256, 768]", div_24: "f32[128, 256, 1]", add_138: "f32[128, 256, 768]", primals_9: "f32[768, 768]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default: "f32[128, 256, 768]" = torch.ops.aten.reshape.default(mm_101, _shape_param_0);  mm_101 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(reshape_default, primals_11);  reshape_default = primals_11 = None
        mul_tensor_1: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[128, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(mul_2, sum_dim_int_list_1);  mul_2 = sum_dim_int_list_1 = None
        sub_tensor: "f32[128, 256, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[128, 256, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[128, 256, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_tensor_1);  div_24 = sub_tensor_1 = None
        add_tensor: "f32[128, 256, 768]" = torch.ops.aten.add.Tensor(add_138, mul_tensor_4);  add_138 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_1: "f32[32768, 768]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None
        permute_default: "f32[768, 768]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_1, permute_default_1)



def make_inputs():
    return [
    torch.randn([32768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([128, 256, 768], dtype=torch.float32, device='cuda'),
    torch.randn([128, 256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 256, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [128, 256, 768],  # _shape_param_0
    [32768, 768],  # _shape_param_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
