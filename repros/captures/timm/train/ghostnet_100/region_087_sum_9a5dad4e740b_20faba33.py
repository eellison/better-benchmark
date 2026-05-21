"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 9a5dad4e740b
Shape hash: 20faba33
"""
_shapes_config = "(T([512, 40, 28, 28], f32, stride=(31360, 1, 1120, 40)), T([512, 40, 28, 28], f32, stride=(31360, 1, 1120, 40)), T([512, 20, 28, 28], f32, stride=(15680, 1, 560, 20)), T([1, 20, 1, 1], f32), T([20], f32), T([20], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_334: "f32[512, 40, 28, 28]", getitem_349: "f32[512, 40, 28, 28]", convolution_30: "f32[512, 20, 28, 28]", unsqueeze_958: "f32[1, 20, 1, 1]", squeeze_79: "f32[20]", primals_170: "f32[20]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor: "f32[512, 40, 28, 28]" = torch.ops.aten.add.Tensor(getitem_334, getitem_349);  getitem_334 = getitem_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        clone_default: "f32[512, 40, 28, 28]" = torch.ops.aten.clone.default(add_tensor, memory_format = torch.contiguous_format)
        copy_default: "f32[512, 40, 28, 28]" = torch.ops.aten.copy.default(add_tensor, clone_default);  add_tensor = clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_tensor: "f32[512, 20, 28, 28]" = torch.ops.aten.slice.Tensor(copy_default, 1, 20, 40);  copy_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sum_dim_int_list: "f32[20]" = torch.ops.aten.sum.dim_IntList(slice_tensor, [0, 2, 3])
        sub_tensor: "f32[512, 20, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_958);  convolution_30 = unsqueeze_958 = None
        mul_tensor: "f32[512, 20, 28, 28]" = torch.ops.aten.mul.Tensor(slice_tensor, sub_tensor)
        sum_dim_int_list_1: "f32[20]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[20]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 2.4912308673469386e-06);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[20]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 2.4912308673469386e-06);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_tensor_4: "f32[20]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_170);  squeeze_79 = primals_170 = None
        unsqueeze_default_6: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[512, 20, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[512, 20, 28, 28]" = torch.ops.aten.sub.Tensor(slice_tensor, mul_tensor_6);  slice_tensor = mul_tensor_6 = None
        sub_tensor_2: "f32[512, 20, 28, 28]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[512, 20, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        return mul_tensor_7



def make_inputs():
    return [
    torch.randn(16056320, dtype=torch.float32, device='cuda').as_strided([512, 40, 28, 28], [31360, 1, 1120, 40]),  # getitem_334
    torch.randn(16056320, dtype=torch.float32, device='cuda').as_strided([512, 40, 28, 28], [31360, 1, 1120, 40]),  # getitem_349
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([512, 20, 28, 28], [15680, 1, 560, 20]),  # convolution_30
    torch.randn([1, 20, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([20], dtype=torch.float32, device='cuda'),
    torch.randn([20], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
