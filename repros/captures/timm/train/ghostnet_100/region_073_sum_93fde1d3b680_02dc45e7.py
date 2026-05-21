"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 93fde1d3b680
Shape hash: 02dc45e7
"""
_shapes_config = "(T([512, 72, 56, 56], f32, stride=(225792, 1, 4032, 72)), T([512, 36, 56, 56], f32, stride=(112896, 1, 2016, 36)), T([512, 36, 56, 56], f32, stride=(112896, 1, 2016, 36)), T([], f32), T([512, 36, 56, 56], f32, stride=(112896, 1, 2016, 36)), T([1, 36, 1, 1], f32), T([36], f32), T([36], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_400: "f32[512, 72, 56, 56]", getitem_403: "f32[512, 36, 56, 56]", relu_5: "f32[512, 36, 56, 56]", full_default: "f32[]", convolution_12: "f32[512, 36, 56, 56]", unsqueeze_1126: "f32[1, 36, 1, 1]", squeeze_37: "f32[36]", primals_78: "f32[36]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_tensor: "f32[512, 36, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_400, 1, 0, 36);  getitem_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor: "f32[512, 36, 56, 56]" = torch.ops.aten.add.Tensor(slice_tensor, getitem_403);  slice_tensor = getitem_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        le_scalar: "b8[512, 36, 56, 56]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_self: "f32[512, 36, 56, 56]" = torch.ops.aten.where.self(le_scalar, full_default, add_tensor);  le_scalar = full_default = add_tensor = None
        sum_dim_int_list: "f32[36]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[512, 36, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_1126);  convolution_12 = unsqueeze_1126 = None
        mul_tensor: "f32[512, 36, 56, 56]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[36]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[36]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 6.228077168367346e-07);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[36]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 6.228077168367346e-07);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_tensor_4: "f32[36]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_78);  squeeze_37 = primals_78 = None
        unsqueeze_default_6: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[512, 36, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[512, 36, 56, 56]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[512, 36, 56, 56]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[512, 36, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        return mul_tensor_7



def make_inputs():
    return [
    torch.randn(115605504, dtype=torch.float32, device='cuda').as_strided([512, 72, 56, 56], [225792, 1, 4032, 72]),  # getitem_400
    torch.randn(57802752, dtype=torch.float32, device='cuda').as_strided([512, 36, 56, 56], [112896, 1, 2016, 36]),  # getitem_403
    torch.randn(57802752, dtype=torch.float32, device='cuda').as_strided([512, 36, 56, 56], [112896, 1, 2016, 36]),  # relu_5
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(57802752, dtype=torch.float32, device='cuda').as_strided([512, 36, 56, 56], [112896, 1, 2016, 36]),  # convolution_12
    torch.randn([1, 36, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([36], dtype=torch.float32, device='cuda'),
    torch.randn([36], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
