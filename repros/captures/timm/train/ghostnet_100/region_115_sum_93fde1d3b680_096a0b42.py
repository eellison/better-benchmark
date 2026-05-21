"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 93fde1d3b680
Shape hash: 096a0b42
"""
_shapes_config = "(T([512, 960, 7, 7], f32, stride=(47040, 1, 6720, 960)), T([512, 480, 7, 7], f32, stride=(23520, 1, 3360, 480)), T([512, 480, 7, 7], f32, stride=(23520, 1, 3360, 480)), T([], f32), T([512, 480, 7, 7], f32, stride=(23520, 1, 3360, 480)), T([1, 480, 1, 1], f32), T([480], f32), T([480], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_217: "f32[512, 960, 7, 7]", getitem_220: "f32[512, 480, 7, 7]", relu_30: "f32[512, 480, 7, 7]", full_default: "f32[]", convolution_73: "f32[512, 480, 7, 7]", unsqueeze_514: "f32[1, 480, 1, 1]", squeeze_190: "f32[480]", primals_404: "f32[480]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_tensor: "f32[512, 480, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_217, 1, 0, 480);  getitem_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor: "f32[512, 480, 7, 7]" = torch.ops.aten.add.Tensor(slice_tensor, getitem_220);  slice_tensor = getitem_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        le_scalar: "b8[512, 480, 7, 7]" = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
        where_self: "f32[512, 480, 7, 7]" = torch.ops.aten.where.self(le_scalar, full_default, add_tensor);  le_scalar = full_default = add_tensor = None
        sum_dim_int_list: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[512, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_514);  convolution_73 = unsqueeze_514 = None
        mul_tensor: "f32[512, 480, 7, 7]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[480]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 3.985969387755102e-05);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[480]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 3.985969387755102e-05);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_190, squeeze_190)
        mul_tensor_4: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_190, primals_404);  squeeze_190 = primals_404 = None
        unsqueeze_default_6: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[512, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[512, 480, 7, 7]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[512, 480, 7, 7]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[512, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        return mul_tensor_7



def make_inputs():
    return [
    torch.randn(24084480, dtype=torch.float32, device='cuda').as_strided([512, 960, 7, 7], [47040, 1, 6720, 960]),  # getitem_217
    torch.randn(12042240, dtype=torch.float32, device='cuda').as_strided([512, 480, 7, 7], [23520, 1, 3360, 480]),  # getitem_220
    torch.randn(12042240, dtype=torch.float32, device='cuda').as_strided([512, 480, 7, 7], [23520, 1, 3360, 480]),  # relu_30
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(12042240, dtype=torch.float32, device='cuda').as_strided([512, 480, 7, 7], [23520, 1, 3360, 480]),  # convolution_73
    torch.randn([1, 480, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
