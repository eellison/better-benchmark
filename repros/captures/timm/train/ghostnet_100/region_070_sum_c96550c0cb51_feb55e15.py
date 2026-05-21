"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: c96550c0cb51
Shape hash: feb55e15
"""
_shapes_config = "(T([512, 48, 56, 56], f32, stride=(150528, 1, 2688, 48)), T([512, 48, 56, 56], f32, stride=(150528, 1, 2688, 48)), T([1, 48, 1, 1], f32), T([48], f32), T([48], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_418: "f32[512, 48, 56, 56]", convolution_7: "f32[512, 48, 56, 56]", unsqueeze_1186: "f32[1, 48, 1, 1]", squeeze_22: "f32[48]", primals_48: "f32[48]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:436 in forward, code: x = self.bn_dw(x)
        sum_dim_int_list: "f32[48]" = torch.ops.aten.sum.dim_IntList(getitem_418, [0, 2, 3])
        sub_tensor: "f32[512, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_1186);  convolution_7 = unsqueeze_1186 = None
        mul_tensor: "f32[512, 48, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_418, sub_tensor)
        sum_dim_int_list_1: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[48]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 6.228077168367346e-07);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[48]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 6.228077168367346e-07);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_tensor_4: "f32[48]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_48);  squeeze_22 = primals_48 = None
        unsqueeze_default_6: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[512, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[512, 48, 56, 56]" = torch.ops.aten.sub.Tensor(getitem_418, mul_tensor_6);  getitem_418 = mul_tensor_6 = None
        sub_tensor_2: "f32[512, 48, 56, 56]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[512, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        return mul_tensor_7



def make_inputs():
    return [
    torch.randn(77070336, dtype=torch.float32, device='cuda').as_strided([512, 48, 56, 56], [150528, 1, 2688, 48]),  # getitem_418
    torch.randn(77070336, dtype=torch.float32, device='cuda').as_strided([512, 48, 56, 56], [150528, 1, 2688, 48]),  # convolution_7
    torch.randn([1, 48, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([48], dtype=torch.float32, device='cuda'),
    torch.randn([48], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
