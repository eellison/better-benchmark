"""
Standalone repro captured via capture_hook.
Label: mobilenet_v2_training
Pattern hash: eb7785209263
Shape hash: de093855
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_6: "f32[4, 144, 56, 56]", getitem_13: "f32[1, 144, 1, 1]", rsqrt_6: "f32[1, 144, 1, 1]", primals_42: "f32[144]", primals_43: "f32[144]", full_default: "f32[]", getitem_236: "f32[4, 144, 56, 56]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv2.py:62 in forward, code: return x + self.conv(x)
        sub_tensor: "f32[4, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_13)
        mul_tensor: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_6);  sub_tensor = None
        unsqueeze_default: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1)
        unsqueeze_default_1: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1);  primals_43 = None
        unsqueeze_default_3: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[4, 144, 56, 56]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        le_scalar: "b8[4, 144, 56, 56]" = torch.ops.aten.le.Scalar(add_tensor, 0.0)
        ge_scalar: "b8[4, 144, 56, 56]" = torch.ops.aten.ge.Scalar(add_tensor, 6.0);  add_tensor = None
        bitwise_or_tensor: "b8[4, 144, 56, 56]" = torch.ops.aten.bitwise_or.Tensor(le_scalar, ge_scalar);  le_scalar = ge_scalar = None
        where_self: "f32[4, 144, 56, 56]" = torch.ops.aten.where.self(bitwise_or_tensor, full_default, getitem_236);  bitwise_or_tensor = full_default = getitem_236 = None
        squeeze_dims: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        unsqueeze_default_4: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_1: "f32[4, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_default_6);  convolution_6 = unsqueeze_default_6 = None
        mul_tensor_2: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_1)
        sum_dim_int_list_1: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3]);  mul_tensor_2 = None
        mul_tensor_3: "f32[144]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 7.971938775510203e-05);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 0);  mul_tensor_3 = None
        unsqueeze_default_8: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_4: "f32[144]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 7.971938775510203e-05);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[144]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_tensor_5: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_6: "f32[144]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default_10: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_11: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_7: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_42);  squeeze_dims_1 = primals_42 = None
        unsqueeze_default_13: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_14: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_8: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[4, 144, 56, 56]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_8);  where_self = mul_tensor_8 = None
        sub_tensor_3: "f32[4, 144, 56, 56]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_9: "f32[4, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        return mul_tensor_9



def make_inputs():
    return [
    torch.randn(1806336, dtype=torch.float32, device='cuda').as_strided([4, 144, 56, 56], [451584, 1, 8064, 144]),  # convolution_6
    torch.randn([1, 144, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 144, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([144], dtype=torch.float32, device='cuda'),
    torch.randn([144], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(1806336, dtype=torch.float32, device='cuda').as_strided([4, 144, 56, 56], [451584, 1, 8064, 144]),  # getitem_236
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
