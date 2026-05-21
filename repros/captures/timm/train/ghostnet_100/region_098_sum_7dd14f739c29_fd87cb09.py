"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 7dd14f739c29
Shape hash: fd87cb09
"""
_shapes_config = "(T([512, 80, 14, 14], f32, stride=(15680, 1, 1120, 80)), T([512, 80, 14, 14], f32, stride=(15680, 1, 1120, 80)), T([1, 80, 1, 1], f32), T([80], f32), T([80], f32), T([512, 480, 14, 14], f32, stride=(94080, 1, 6720, 480)), T([512, 240, 14, 14], f32, stride=(47040, 1, 3360, 240)), T([512, 240, 14, 14], f32, stride=(47040, 1, 3360, 240)), T([], f32), T([512, 240, 14, 14], f32, stride=(47040, 1, 3360, 240)), T([1, 240, 1, 1], f32), T([240], f32), T([240], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_271: "f32[512, 80, 14, 14]", convolution_56: "f32[512, 80, 14, 14]", unsqueeze_670: "f32[1, 80, 1, 1]", squeeze_151: "f32[80]", primals_318: "f32[80]", add_446: "f32[512, 480, 14, 14]", getitem_289: "f32[512, 240, 14, 14]", relu_21: "f32[512, 240, 14, 14]", full_default: "f32[]", convolution_50: "f32[512, 240, 14, 14]", unsqueeze_718: "f32[1, 240, 1, 1]", squeeze_139: "f32[240]", primals_290: "f32[240]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        sum_dim_int_list: "f32[80]" = torch.ops.aten.sum.dim_IntList(getitem_271, [0, 2, 3])
        sub_tensor: "f32[512, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_670);  convolution_56 = unsqueeze_670 = None
        mul_tensor: "f32[512, 80, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_271, sub_tensor)
        sum_dim_int_list_1: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[80]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 9.964923469387754e-06);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[80]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 9.964923469387754e-06);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
        mul_tensor_4: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_318);  squeeze_151 = primals_318 = None
        unsqueeze_default_6: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[512, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[512, 80, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_271, mul_tensor_6);  getitem_271 = mul_tensor_6 = None
        sub_tensor_2: "f32[512, 80, 14, 14]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[512, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_tensor: "f32[512, 240, 14, 14]" = torch.ops.aten.slice.Tensor(add_446, 1, 0, 240);  add_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor: "f32[512, 240, 14, 14]" = torch.ops.aten.add.Tensor(slice_tensor, getitem_289);  slice_tensor = getitem_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        le_scalar: "b8[512, 240, 14, 14]" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
        where_self: "f32[512, 240, 14, 14]" = torch.ops.aten.where.self(le_scalar, full_default, add_tensor);  le_scalar = full_default = add_tensor = None
        sum_dim_int_list_2: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_3: "f32[512, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_718);  convolution_50 = unsqueeze_718 = None
        mul_tensor_8: "f32[512, 240, 14, 14]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_3)
        sum_dim_int_list_3: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 2, 3]);  mul_tensor_8 = None
        mul_tensor_9: "f32[240]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 9.964923469387754e-06);  sum_dim_int_list_2 = None
        unsqueeze_default_9: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, 0);  mul_tensor_9 = None
        unsqueeze_default_10: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None
        mul_tensor_10: "f32[240]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 9.964923469387754e-06);  sum_dim_int_list_3 = None
        mul_tensor_11: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
        mul_tensor_12: "f32[240]" = torch.ops.aten.mul.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
        unsqueeze_default_12: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_tensor_12, 0);  mul_tensor_12 = None
        unsqueeze_default_13: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 2);  unsqueeze_default_12 = None
        unsqueeze_default_14: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 3);  unsqueeze_default_13 = None
        mul_tensor_13: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_290);  squeeze_139 = primals_290 = None
        unsqueeze_default_15: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_tensor_13, 0);  mul_tensor_13 = None
        unsqueeze_default_16: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 2);  unsqueeze_default_15 = None
        unsqueeze_default_17: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 3);  unsqueeze_default_16 = None
        mul_tensor_14: "f32[512, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_14);  sub_tensor_3 = unsqueeze_default_14 = None
        sub_tensor_4: "f32[512, 240, 14, 14]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_14);  where_self = mul_tensor_14 = None
        sub_tensor_5: "f32[512, 240, 14, 14]" = torch.ops.aten.sub.Tensor(sub_tensor_4, unsqueeze_default_11);  sub_tensor_4 = unsqueeze_default_11 = None
        mul_tensor_15: "f32[512, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_17);  sub_tensor_5 = unsqueeze_default_17 = None
        return (mul_tensor_7, mul_tensor_15)



def make_inputs():
    return [
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([512, 80, 14, 14], [15680, 1, 1120, 80]),  # getitem_271
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([512, 80, 14, 14], [15680, 1, 1120, 80]),  # convolution_56
    torch.randn([1, 80, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn(48168960, dtype=torch.float32, device='cuda').as_strided([512, 480, 14, 14], [94080, 1, 6720, 480]),  # add_446
    torch.randn(24084480, dtype=torch.float32, device='cuda').as_strided([512, 240, 14, 14], [47040, 1, 3360, 240]),  # getitem_289
    torch.randn(24084480, dtype=torch.float32, device='cuda').as_strided([512, 240, 14, 14], [47040, 1, 3360, 240]),  # relu_21
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(24084480, dtype=torch.float32, device='cuda').as_strided([512, 240, 14, 14], [47040, 1, 3360, 240]),  # convolution_50
    torch.randn([1, 240, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
