"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 1c4efa8068a8
Shape hash: 3b285a26
"""
_shapes_config = "(T([512, 16, 112, 112], f32, stride=(200704, 1, 1792, 16)), T([512, 8, 112, 112], f32, stride=(100352, 1, 896, 8)), T([512, 8, 112, 112], f32, stride=(100352, 1, 896, 8)), T([1, 8, 1, 1], f32), T([8], f32), T([8], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, copy_32: "f32[512, 16, 112, 112]", getitem_430: "f32[512, 8, 112, 112]", convolution_3: "f32[512, 8, 112, 112]", unsqueeze_1234: "f32[1, 8, 1, 1]", squeeze_10: "f32[8]", primals_24: "f32[8]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_tensor: "f32[512, 8, 112, 112]" = torch.ops.aten.slice.Tensor(copy_32, 1, 0, 8);  copy_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor: "f32[512, 8, 112, 112]" = torch.ops.aten.add.Tensor(slice_tensor, getitem_430);  slice_tensor = getitem_430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        sum_dim_int_list: "f32[8]" = torch.ops.aten.sum.dim_IntList(add_tensor, [0, 2, 3])
        sub_tensor: "f32[512, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_1234);  convolution_3 = unsqueeze_1234 = None
        mul_tensor: "f32[512, 8, 112, 112]" = torch.ops.aten.mul.Tensor(add_tensor, sub_tensor)
        sum_dim_int_list_1: "f32[8]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[8]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 1.5570192920918366e-07);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[8]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 1.5570192920918366e-07);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_tensor_4: "f32[8]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_24);  squeeze_10 = primals_24 = None
        unsqueeze_default_6: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[512, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[512, 8, 112, 112]" = torch.ops.aten.sub.Tensor(add_tensor, mul_tensor_6);  add_tensor = mul_tensor_6 = None
        sub_tensor_2: "f32[512, 8, 112, 112]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[512, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        return mul_tensor_7



def make_inputs():
    return [
    torch.randn(102760448, dtype=torch.float32, device='cuda').as_strided([512, 16, 112, 112], [200704, 1, 1792, 16]),  # copy_32
    torch.randn(51380224, dtype=torch.float32, device='cuda').as_strided([512, 8, 112, 112], [100352, 1, 896, 8]),  # getitem_430
    torch.randn(51380224, dtype=torch.float32, device='cuda').as_strided([512, 8, 112, 112], [100352, 1, 896, 8]),  # convolution_3
    torch.randn([1, 8, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8], dtype=torch.float32, device='cuda'),
    torch.randn([8], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
