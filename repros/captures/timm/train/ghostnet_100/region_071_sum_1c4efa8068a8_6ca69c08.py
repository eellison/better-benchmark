"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 1c4efa8068a8
Shape hash: 6ca69c08
"""
_shapes_config = "(T([512, 24, 56, 56], f32, stride=(75264, 1, 1344, 24)), T([512, 12, 56, 56], f32, stride=(37632, 1, 672, 12)), T([512, 12, 56, 56], f32, stride=(37632, 1, 672, 12)), T([1, 12, 1, 1], f32), T([12], f32), T([12], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, copy_30: "f32[512, 24, 56, 56]", getitem_415: "f32[512, 12, 56, 56]", convolution_8: "f32[512, 12, 56, 56]", unsqueeze_1174: "f32[1, 12, 1, 1]", squeeze_25: "f32[12]", primals_54: "f32[12]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_tensor: "f32[512, 12, 56, 56]" = torch.ops.aten.slice.Tensor(copy_30, 1, 0, 12);  copy_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor: "f32[512, 12, 56, 56]" = torch.ops.aten.add.Tensor(slice_tensor, getitem_415);  slice_tensor = getitem_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        sum_dim_int_list: "f32[12]" = torch.ops.aten.sum.dim_IntList(add_tensor, [0, 2, 3])
        sub_tensor: "f32[512, 12, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_1174);  convolution_8 = unsqueeze_1174 = None
        mul_tensor: "f32[512, 12, 56, 56]" = torch.ops.aten.mul.Tensor(add_tensor, sub_tensor)
        sum_dim_int_list_1: "f32[12]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[12]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 6.228077168367346e-07);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 12]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 12, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[12]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 6.228077168367346e-07);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_tensor_4: "f32[12]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 12]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 12, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_54);  squeeze_25 = primals_54 = None
        unsqueeze_default_6: "f32[1, 12]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 12, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[512, 12, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[512, 12, 56, 56]" = torch.ops.aten.sub.Tensor(add_tensor, mul_tensor_6);  add_tensor = mul_tensor_6 = None
        sub_tensor_2: "f32[512, 12, 56, 56]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[512, 12, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        return mul_tensor_7



def make_inputs():
    return [
    torch.randn(38535168, dtype=torch.float32, device='cuda').as_strided([512, 24, 56, 56], [75264, 1, 1344, 24]),  # copy_30
    torch.randn(19267584, dtype=torch.float32, device='cuda').as_strided([512, 12, 56, 56], [37632, 1, 672, 12]),  # getitem_415
    torch.randn(19267584, dtype=torch.float32, device='cuda').as_strided([512, 12, 56, 56], [37632, 1, 672, 12]),  # convolution_8
    torch.randn([1, 12, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([12], dtype=torch.float32, device='cuda'),
    torch.randn([12], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
