"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 9c6bfbc4476f
Shape hash: 78b4082d
"""
_shapes_config = "(T([512, 40, 28, 28], f32), T([512, 40, 28, 28], f32, stride=(31360, 1, 1120, 40)), T([1, 40, 1, 1], f32), T([40], f32), T([40], f32), T([512, 72, 56, 56], f32, stride=(225792, 1, 4032, 72)), T([512, 36, 56, 56], f32, stride=(112896, 1, 2016, 36)), T([1, 36, 1, 1], f32), T([1, 36, 1, 1], f32), T([36], f32), T([36], f32), T([], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, clone_13: "f32[512, 40, 28, 28]", convolution_24: "f32[512, 40, 28, 28]", unsqueeze_1006: "f32[1, 40, 1, 1]", squeeze_67: "f32[40]", primals_142: "f32[40]", getitem_388: "f32[512, 72, 56, 56]", convolution_17: "f32[512, 36, 56, 56]", getitem_35: "f32[1, 36, 1, 1]", rsqrt_17: "f32[1, 36, 1, 1]", primals_108: "f32[36]", primals_109: "f32[36]", full_default: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        sum_dim_int_list: "f32[40]" = torch.ops.aten.sum.dim_IntList(clone_13, [0, 2, 3])
        sub_tensor: "f32[512, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_1006);  convolution_24 = unsqueeze_1006 = None
        mul_tensor: "f32[512, 40, 28, 28]" = torch.ops.aten.mul.Tensor(clone_13, sub_tensor)
        sum_dim_int_list_1: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[40]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 2.4912308673469386e-06);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[40]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 2.4912308673469386e-06);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_tensor_4: "f32[40]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_142);  squeeze_67 = primals_142 = None
        unsqueeze_default_6: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[512, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[512, 40, 28, 28]" = torch.ops.aten.sub.Tensor(clone_13, mul_tensor_6);  clone_13 = mul_tensor_6 = None
        sub_tensor_2: "f32[512, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[512, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_tensor: "f32[512, 36, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_388, 1, 36, 72);  getitem_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sub_tensor_3: "f32[512, 36, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_35)
        mul_tensor_8: "f32[512, 36, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_3, rsqrt_17);  sub_tensor_3 = None
        unsqueeze_default_9: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1)
        unsqueeze_default_10: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, -1);  unsqueeze_default_9 = None
        mul_tensor_9: "f32[512, 36, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor_8, unsqueeze_default_10);  mul_tensor_8 = unsqueeze_default_10 = None
        unsqueeze_default_11: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(primals_109, -1);  primals_109 = None
        unsqueeze_default_12: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, -1);  unsqueeze_default_11 = None
        add_tensor: "f32[512, 36, 56, 56]" = torch.ops.aten.add.Tensor(mul_tensor_9, unsqueeze_default_12);  mul_tensor_9 = unsqueeze_default_12 = None
        relu_default: "f32[512, 36, 56, 56]" = torch.ops.aten.relu.default(add_tensor);  add_tensor = None
        le_scalar: "b8[512, 36, 56, 56]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self: "f32[512, 36, 56, 56]" = torch.ops.aten.where.self(le_scalar, full_default, slice_tensor);  le_scalar = full_default = slice_tensor = None
        squeeze_dims: "f32[36]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        unsqueeze_default_13: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_14: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        sum_dim_int_list_2: "f32[36]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_4: "f32[512, 36, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_default_15);  convolution_17 = unsqueeze_default_15 = None
        mul_tensor_10: "f32[512, 36, 56, 56]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_4)
        sum_dim_int_list_3: "f32[36]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 2, 3]);  mul_tensor_10 = None
        mul_tensor_11: "f32[36]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 6.228077168367346e-07);  sum_dim_int_list_2 = None
        unsqueeze_default_16: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(mul_tensor_11, 0);  mul_tensor_11 = None
        unsqueeze_default_17: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 2);  unsqueeze_default_16 = None
        unsqueeze_default_18: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_17, 3);  unsqueeze_default_17 = None
        mul_tensor_12: "f32[36]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 6.228077168367346e-07);  sum_dim_int_list_3 = None
        squeeze_dims_1: "f32[36]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_tensor_13: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_14: "f32[36]" = torch.ops.aten.mul.Tensor(mul_tensor_12, mul_tensor_13);  mul_tensor_12 = mul_tensor_13 = None
        unsqueeze_default_19: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(mul_tensor_14, 0);  mul_tensor_14 = None
        unsqueeze_default_20: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_19, 2);  unsqueeze_default_19 = None
        unsqueeze_default_21: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 3);  unsqueeze_default_20 = None
        mul_tensor_15: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_108);  squeeze_dims_1 = primals_108 = None
        unsqueeze_default_22: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(mul_tensor_15, 0);  mul_tensor_15 = None
        unsqueeze_default_23: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, 2);  unsqueeze_default_22 = None
        unsqueeze_default_24: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 3);  unsqueeze_default_23 = None
        mul_tensor_16: "f32[512, 36, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_4, unsqueeze_default_21);  sub_tensor_4 = unsqueeze_default_21 = None
        sub_tensor_5: "f32[512, 36, 56, 56]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_16);  where_self = mul_tensor_16 = None
        sub_tensor_6: "f32[512, 36, 56, 56]" = torch.ops.aten.sub.Tensor(sub_tensor_5, unsqueeze_default_18);  sub_tensor_5 = unsqueeze_default_18 = None
        mul_tensor_17: "f32[512, 36, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_6, unsqueeze_default_24);  sub_tensor_6 = unsqueeze_default_24 = None
        return (mul_tensor_7, mul_tensor_17)



def make_inputs():
    return [
    torch.randn([512, 40, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn(16056320, dtype=torch.float32, device='cuda').as_strided([512, 40, 28, 28], [31360, 1, 1120, 40]),  # convolution_24
    torch.randn([1, 40, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn(115605504, dtype=torch.float32, device='cuda').as_strided([512, 72, 56, 56], [225792, 1, 4032, 72]),  # getitem_388
    torch.randn(57802752, dtype=torch.float32, device='cuda').as_strided([512, 36, 56, 56], [112896, 1, 2016, 36]),  # convolution_17
    torch.randn([1, 36, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 36, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([36], dtype=torch.float32, device='cuda'),
    torch.randn([36], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
