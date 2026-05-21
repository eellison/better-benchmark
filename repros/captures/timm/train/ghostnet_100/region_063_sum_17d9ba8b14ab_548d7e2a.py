"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 17d9ba8b14ab
Shape hash: 548d7e2a
"""
_shapes_config = "(T([512, 16, 112, 112], f32), T([512, 16, 112, 112], f32, stride=(200704, 1, 1792, 16)), T([512, 16, 112, 112], f32, stride=(200704, 1, 1792, 16)), T([], f32), T([512, 16, 112, 112], f32, stride=(200704, 1, 1792, 16)), T([1, 16, 1, 1], f32), T([16], f32), T([16], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, clone_16: "f32[512, 16, 112, 112]", getitem_439: "f32[512, 16, 112, 112]", relu: "f32[512, 16, 112, 112]", full_default: "f32[]", convolution: "f32[512, 16, 112, 112]", unsqueeze_1270: "f32[1, 16, 1, 1]", squeeze_1: "f32[16]", primals_6: "f32[16]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor: "f32[512, 16, 112, 112]" = torch.ops.aten.add.Tensor(clone_16, getitem_439);  clone_16 = getitem_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:823 in forward_features, code: x = self.act1(x)
        le_scalar: "b8[512, 16, 112, 112]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_self: "f32[512, 16, 112, 112]" = torch.ops.aten.where.self(le_scalar, full_default, add_tensor);  le_scalar = full_default = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:822 in forward_features, code: x = self.bn1(x)
        sum_dim_int_list: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[512, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1270);  convolution = unsqueeze_1270 = None
        mul_tensor: "f32[512, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[16]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 1.5570192920918366e-07);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[16]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 1.5570192920918366e-07);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_tensor_4: "f32[16]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_6);  squeeze_1 = primals_6 = None
        unsqueeze_default_6: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[512, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[512, 16, 112, 112]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[512, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[512, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        return mul_tensor_7



def make_inputs():
    return [
    torch.randn([512, 16, 112, 112], dtype=torch.float32, device='cuda'),
    torch.randn(102760448, dtype=torch.float32, device='cuda').as_strided([512, 16, 112, 112], [200704, 1, 1792, 16]),  # getitem_439
    torch.randn(102760448, dtype=torch.float32, device='cuda').as_strided([512, 16, 112, 112], [200704, 1, 1792, 16]),  # relu
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(102760448, dtype=torch.float32, device='cuda').as_strided([512, 16, 112, 112], [200704, 1, 1792, 16]),  # convolution
    torch.randn([1, 16, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
