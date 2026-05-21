"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: fdea164def2e
Shape hash: 55150907
"""
_shapes_config = "(T([512, 16, 112, 112], f32, stride=(200704, 1, 1792, 16)), T([512, 8, 112, 112], f32, stride=(100352, 1, 896, 8)), T([1, 8, 1, 1], f32), T([1, 8, 1, 1], f32), T([8], f32), T([8], f32), T([], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_433: "f32[512, 16, 112, 112]", convolution_2: "f32[512, 8, 112, 112]", getitem_5: "f32[1, 8, 1, 1]", rsqrt_2: "f32[1, 8, 1, 1]", primals_18: "f32[8]", primals_19: "f32[8]", full_default: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_tensor: "f32[512, 8, 112, 112]" = torch.ops.aten.slice.Tensor(getitem_433, 1, 8, 16);  getitem_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sub_tensor: "f32[512, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
        mul_tensor: "f32[512, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_2);  sub_tensor = None
        unsqueeze_default: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1)
        unsqueeze_default_1: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 8, 112, 112]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1);  primals_19 = None
        unsqueeze_default_3: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[512, 8, 112, 112]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        relu_default: "f32[512, 8, 112, 112]" = torch.ops.aten.relu.default(add_tensor);  add_tensor = None
        le_scalar: "b8[512, 8, 112, 112]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self: "f32[512, 8, 112, 112]" = torch.ops.aten.where.self(le_scalar, full_default, slice_tensor);  le_scalar = full_default = slice_tensor = None
        squeeze_dims: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        unsqueeze_default_4: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[8]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_1: "f32[512, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_default_6);  convolution_2 = unsqueeze_default_6 = None
        mul_tensor_2: "f32[512, 8, 112, 112]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_1)
        sum_dim_int_list_1: "f32[8]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3]);  mul_tensor_2 = None
        mul_tensor_3: "f32[8]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 1.5570192920918366e-07);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 0);  mul_tensor_3 = None
        unsqueeze_default_8: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_4: "f32[8]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 1.5570192920918366e-07);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[8]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_tensor_5: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_6: "f32[8]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default_10: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_11: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_7: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_18);  squeeze_dims_1 = primals_18 = None
        unsqueeze_default_13: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_14: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_8: "f32[512, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[512, 8, 112, 112]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_8);  where_self = mul_tensor_8 = None
        sub_tensor_3: "f32[512, 8, 112, 112]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_9: "f32[512, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        return mul_tensor_9



def make_inputs():
    return [
    torch.randn(102760448, dtype=torch.float32, device='cuda').as_strided([512, 16, 112, 112], [200704, 1, 1792, 16]),  # getitem_433
    torch.randn(51380224, dtype=torch.float32, device='cuda').as_strided([512, 8, 112, 112], [100352, 1, 896, 8]),  # convolution_2
    torch.randn([1, 8, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 8, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8], dtype=torch.float32, device='cuda'),
    torch.randn([8], dtype=torch.float32, device='cuda'),
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
