"""
Standalone repro captured via capture_hook.
Label: timm_inception_v3_train
Pattern hash: 0c9c964d4069
Shape hash: 3f5858d9
"""
_shapes_config = "(T([128, 80, 73, 73], f32, stride=(426320, 1, 5840, 80)), T([], f32), T([128, 80, 73, 73], f32, stride=(426320, 1, 5840, 80)), T([128, 80, 73, 73], f32, stride=(426320, 1, 5840, 80)), T([1, 80, 1, 1], f32), T([80], f32), T([80], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, relu_3: "f32[128, 80, 73, 73]", full_default: "f32[]", getitem_463: "f32[128, 80, 73, 73]", convolution_3: "f32[128, 80, 73, 73]", unsqueeze_1458: "f32[1, 80, 1, 1]", squeeze_10: "f32[80]", primals_24: "f32[80]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_scalar: "b8[128, 80, 73, 73]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_self: "f32[128, 80, 73, 73]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_463);  le_scalar = full_default = getitem_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_dim_int_list: "f32[80]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[128, 80, 73, 73]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_1458);  convolution_3 = unsqueeze_1458 = None
        mul_tensor: "f32[128, 80, 73, 73]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[80]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 1.4660349033589792e-06);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[80]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 1.4660349033589792e-06);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_tensor_4: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_24);  squeeze_10 = primals_24 = None
        unsqueeze_default_6: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[128, 80, 73, 73]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[128, 80, 73, 73]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[128, 80, 73, 73]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[128, 80, 73, 73]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        return mul_tensor_7



def make_inputs():
    return [
    torch.randn(54568960, dtype=torch.float32, device='cuda').as_strided([128, 80, 73, 73], [426320, 1, 5840, 80]),  # relu_3
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(54568960, dtype=torch.float32, device='cuda').as_strided([128, 80, 73, 73], [426320, 1, 5840, 80]),  # getitem_463
    torch.randn(54568960, dtype=torch.float32, device='cuda').as_strided([128, 80, 73, 73], [426320, 1, 5840, 80]),  # convolution_3
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
