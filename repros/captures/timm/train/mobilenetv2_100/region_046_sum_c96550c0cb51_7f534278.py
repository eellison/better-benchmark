"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv2_100_train
Pattern hash: c96550c0cb51
Shape hash: 7f534278
"""
_shapes_config = "(T([128, 160, 7, 7], f32, stride=(7840, 1, 1120, 160)), T([128, 160, 7, 7], f32, stride=(7840, 1, 1120, 160)), T([1, 160, 1, 1], f32), T([160], f32), T([160], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_113: "f32[128, 160, 7, 7]", convolution_47: "f32[128, 160, 7, 7]", unsqueeze_258: "f32[1, 160, 1, 1]", squeeze_142: "f32[160]", primals_288: "f32[160]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_dim_int_list: "f32[160]" = torch.ops.aten.sum.dim_IntList(getitem_113, [0, 2, 3])
        sub_tensor: "f32[128, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_258);  convolution_47 = unsqueeze_258 = None
        mul_tensor: "f32[128, 160, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_113, sub_tensor)
        sum_dim_int_list_1: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[160]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.00015943877551020407);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[160]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.00015943877551020407);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
        mul_tensor_4: "f32[160]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_288);  squeeze_142 = primals_288 = None
        unsqueeze_default_6: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[128, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[128, 160, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_113, mul_tensor_6);  getitem_113 = mul_tensor_6 = None
        sub_tensor_2: "f32[128, 160, 7, 7]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[128, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        return mul_tensor_7



def make_inputs():
    return [
    torch.randn(1003520, dtype=torch.float32, device='cuda').as_strided([128, 160, 7, 7], [7840, 1, 1120, 160]),  # getitem_113
    torch.randn(1003520, dtype=torch.float32, device='cuda').as_strided([128, 160, 7, 7], [7840, 1, 1120, 160]),  # convolution_47
    torch.randn([1, 160, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
