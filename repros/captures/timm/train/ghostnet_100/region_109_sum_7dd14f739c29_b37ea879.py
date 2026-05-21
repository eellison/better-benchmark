"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 7dd14f739c29
Shape hash: b37ea879
"""
_shapes_config = "(T([512, 112, 7, 7], f32, stride=(5488, 1, 784, 112)), T([512, 112, 7, 7], f32, stride=(5488, 1, 784, 112)), T([1, 112, 1, 1], f32), T([112], f32), T([112], f32), T([512, 672, 14, 14], f32, stride=(131712, 1, 9408, 672)), T([512, 336, 14, 14], f32, stride=(65856, 1, 4704, 336)), T([512, 336, 14, 14], f32, stride=(65856, 1, 4704, 336)), T([], f32), T([512, 336, 14, 14], f32, stride=(65856, 1, 4704, 336)), T([1, 336, 1, 1], f32), T([336], f32), T([336], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_226: "f32[512, 112, 7, 7]", convolution_71: "f32[512, 112, 7, 7]", unsqueeze_538: "f32[1, 112, 1, 1]", squeeze_184: "f32[112]", primals_392: "f32[112]", getitem_244: "f32[512, 672, 14, 14]", getitem_247: "f32[512, 336, 14, 14]", relu_27: "f32[512, 336, 14, 14]", full_default: "f32[]", convolution_64: "f32[512, 336, 14, 14]", unsqueeze_598: "f32[1, 336, 1, 1]", squeeze_169: "f32[336]", primals_358: "f32[336]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        sum_dim_int_list: "f32[112]" = torch.ops.aten.sum.dim_IntList(getitem_226, [0, 2, 3])
        sub_tensor: "f32[512, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_538);  convolution_71 = unsqueeze_538 = None
        mul_tensor: "f32[512, 112, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_226, sub_tensor)
        sum_dim_int_list_1: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[112]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 3.985969387755102e-05);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[112]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 3.985969387755102e-05);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
        mul_tensor_4: "f32[112]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_184, primals_392);  squeeze_184 = primals_392 = None
        unsqueeze_default_6: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[512, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[512, 112, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_226, mul_tensor_6);  getitem_226 = mul_tensor_6 = None
        sub_tensor_2: "f32[512, 112, 7, 7]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[512, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_tensor: "f32[512, 336, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_244, 1, 0, 336);  getitem_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor: "f32[512, 336, 14, 14]" = torch.ops.aten.add.Tensor(slice_tensor, getitem_247);  slice_tensor = getitem_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        le_scalar: "b8[512, 336, 14, 14]" = torch.ops.aten.le.Scalar(relu_27, 0);  relu_27 = None
        where_self: "f32[512, 336, 14, 14]" = torch.ops.aten.where.self(le_scalar, full_default, add_tensor);  le_scalar = full_default = add_tensor = None
        sum_dim_int_list_2: "f32[336]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_3: "f32[512, 336, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_598);  convolution_64 = unsqueeze_598 = None
        mul_tensor_8: "f32[512, 336, 14, 14]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_3)
        sum_dim_int_list_3: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 2, 3]);  mul_tensor_8 = None
        mul_tensor_9: "f32[336]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 9.964923469387754e-06);  sum_dim_int_list_2 = None
        unsqueeze_default_9: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, 0);  mul_tensor_9 = None
        unsqueeze_default_10: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None
        mul_tensor_10: "f32[336]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 9.964923469387754e-06);  sum_dim_int_list_3 = None
        mul_tensor_11: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
        mul_tensor_12: "f32[336]" = torch.ops.aten.mul.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
        unsqueeze_default_12: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_tensor_12, 0);  mul_tensor_12 = None
        unsqueeze_default_13: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 2);  unsqueeze_default_12 = None
        unsqueeze_default_14: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 3);  unsqueeze_default_13 = None
        mul_tensor_13: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_358);  squeeze_169 = primals_358 = None
        unsqueeze_default_15: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_tensor_13, 0);  mul_tensor_13 = None
        unsqueeze_default_16: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 2);  unsqueeze_default_15 = None
        unsqueeze_default_17: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 3);  unsqueeze_default_16 = None
        mul_tensor_14: "f32[512, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_14);  sub_tensor_3 = unsqueeze_default_14 = None
        sub_tensor_4: "f32[512, 336, 14, 14]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_14);  where_self = mul_tensor_14 = None
        sub_tensor_5: "f32[512, 336, 14, 14]" = torch.ops.aten.sub.Tensor(sub_tensor_4, unsqueeze_default_11);  sub_tensor_4 = unsqueeze_default_11 = None
        mul_tensor_15: "f32[512, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_17);  sub_tensor_5 = unsqueeze_default_17 = None
        return (mul_tensor_7, mul_tensor_15)



def make_inputs():
    return [
    torch.randn(2809856, dtype=torch.float32, device='cuda').as_strided([512, 112, 7, 7], [5488, 1, 784, 112]),  # getitem_226
    torch.randn(2809856, dtype=torch.float32, device='cuda').as_strided([512, 112, 7, 7], [5488, 1, 784, 112]),  # convolution_71
    torch.randn([1, 112, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([112], dtype=torch.float32, device='cuda'),
    torch.randn([112], dtype=torch.float32, device='cuda'),
    torch.randn(67436544, dtype=torch.float32, device='cuda').as_strided([512, 672, 14, 14], [131712, 1, 9408, 672]),  # getitem_244
    torch.randn(33718272, dtype=torch.float32, device='cuda').as_strided([512, 336, 14, 14], [65856, 1, 4704, 336]),  # getitem_247
    torch.randn(33718272, dtype=torch.float32, device='cuda').as_strided([512, 336, 14, 14], [65856, 1, 4704, 336]),  # relu_27
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(33718272, dtype=torch.float32, device='cuda').as_strided([512, 336, 14, 14], [65856, 1, 4704, 336]),  # convolution_64
    torch.randn([1, 336, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([336], dtype=torch.float32, device='cuda'),
    torch.randn([336], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
