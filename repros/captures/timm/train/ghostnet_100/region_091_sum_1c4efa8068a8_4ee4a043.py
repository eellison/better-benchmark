"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 1c4efa8068a8
Shape hash: 4ee4a043
"""
_shapes_config = "(T([512, 80, 14, 14], f32, stride=(15680, 1, 1120, 80)), T([512, 40, 14, 14], f32, stride=(7840, 1, 560, 40)), T([512, 40, 14, 14], f32, stride=(7840, 1, 560, 40)), T([1, 40, 1, 1], f32), T([40], f32), T([40], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, copy_22: "f32[512, 80, 14, 14]", getitem_337: "f32[512, 40, 14, 14]", convolution_34: "f32[512, 40, 14, 14]", unsqueeze_910: "f32[1, 40, 1, 1]", squeeze_91: "f32[40]", primals_194: "f32[40]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_tensor: "f32[512, 40, 14, 14]" = torch.ops.aten.slice.Tensor(copy_22, 1, 0, 40);  copy_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor: "f32[512, 40, 14, 14]" = torch.ops.aten.add.Tensor(slice_tensor, getitem_337);  slice_tensor = getitem_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        sum_dim_int_list: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_tensor, [0, 2, 3])
        sub_tensor: "f32[512, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_910);  convolution_34 = unsqueeze_910 = None
        mul_tensor: "f32[512, 40, 14, 14]" = torch.ops.aten.mul.Tensor(add_tensor, sub_tensor)
        sum_dim_int_list_1: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[40]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 9.964923469387754e-06);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[40]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 9.964923469387754e-06);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_tensor_4: "f32[40]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_194);  squeeze_91 = primals_194 = None
        unsqueeze_default_6: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[512, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[512, 40, 14, 14]" = torch.ops.aten.sub.Tensor(add_tensor, mul_tensor_6);  add_tensor = mul_tensor_6 = None
        sub_tensor_2: "f32[512, 40, 14, 14]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[512, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        return mul_tensor_7



def make_inputs():
    return [
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([512, 80, 14, 14], [15680, 1, 1120, 80]),  # copy_22
    torch.randn(4014080, dtype=torch.float32, device='cuda').as_strided([512, 40, 14, 14], [7840, 1, 560, 40]),  # getitem_337
    torch.randn(4014080, dtype=torch.float32, device='cuda').as_strided([512, 40, 14, 14], [7840, 1, 560, 40]),  # convolution_34
    torch.randn([1, 40, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
