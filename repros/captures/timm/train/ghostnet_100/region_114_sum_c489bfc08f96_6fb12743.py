"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: c489bfc08f96
Shape hash: 6fb12743
"""
_shapes_config = "(T([512, 160, 7, 7], f32), T([512, 160, 7, 7], f32, stride=(7840, 1, 1120, 160)), T([512, 80, 7, 7], f32, stride=(3920, 1, 560, 80)), T([1, 80, 1, 1], f32), T([80], f32), T([80], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, clone_4: "f32[512, 160, 7, 7]", getitem_223: "f32[512, 160, 7, 7]", convolution_70: "f32[512, 80, 7, 7]", unsqueeze_550: "f32[1, 80, 1, 1]", squeeze_181: "f32[80]", primals_386: "f32[80]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor: "f32[512, 160, 7, 7]" = torch.ops.aten.add.Tensor(clone_4, getitem_223);  clone_4 = getitem_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        new_empty_strided_default: "f32[512, 160, 7, 7]" = torch.ops.aten.new_empty_strided.default(add_tensor, [512, 160, 7, 7], [7840, 1, 1120, 160])
        copy_default: "f32[512, 160, 7, 7]" = torch.ops.aten.copy.default(new_empty_strided_default, add_tensor);  new_empty_strided_default = add_tensor = None
        clone_default: "f32[512, 160, 7, 7]" = torch.ops.aten.clone.default(copy_default, memory_format = torch.contiguous_format)
        copy_default_1: "f32[512, 160, 7, 7]" = torch.ops.aten.copy.default(copy_default, clone_default);  copy_default = clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_tensor: "f32[512, 80, 7, 7]" = torch.ops.aten.slice.Tensor(copy_default_1, 1, 80, 160);  copy_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sum_dim_int_list: "f32[80]" = torch.ops.aten.sum.dim_IntList(slice_tensor, [0, 2, 3])
        sub_tensor: "f32[512, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_550);  convolution_70 = unsqueeze_550 = None
        mul_tensor: "f32[512, 80, 7, 7]" = torch.ops.aten.mul.Tensor(slice_tensor, sub_tensor)
        sum_dim_int_list_1: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[80]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 3.985969387755102e-05);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[80]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 3.985969387755102e-05);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
        mul_tensor_4: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_181, primals_386);  squeeze_181 = primals_386 = None
        unsqueeze_default_6: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[512, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[512, 80, 7, 7]" = torch.ops.aten.sub.Tensor(slice_tensor, mul_tensor_6);  slice_tensor = mul_tensor_6 = None
        sub_tensor_2: "f32[512, 80, 7, 7]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[512, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        return mul_tensor_7



def make_inputs():
    return [
    torch.randn([512, 160, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn(4014080, dtype=torch.float32, device='cuda').as_strided([512, 160, 7, 7], [7840, 1, 1120, 160]),  # getitem_223
    torch.randn(2007040, dtype=torch.float32, device='cuda').as_strided([512, 80, 7, 7], [3920, 1, 560, 80]),  # convolution_70
    torch.randn([1, 80, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
