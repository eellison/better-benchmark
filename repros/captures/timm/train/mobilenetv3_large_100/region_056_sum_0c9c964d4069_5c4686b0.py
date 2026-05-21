"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_train
Pattern hash: 0c9c964d4069
Shape hash: 5c4686b0
"""
_shapes_config = "(T([512, 120, 28, 28], f32, stride=(94080, 1, 3360, 120)), T([], f32), T([512, 120, 28, 28], f32, stride=(94080, 1, 3360, 120)), T([512, 120, 28, 28], f32, stride=(94080, 1, 3360, 120)), T([1, 120, 1, 1], f32), T([120], f32), T([120], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, relu_8: "f32[512, 120, 28, 28]", full_default: "f32[]", getitem_233: "f32[512, 120, 28, 28]", convolution_14: "f32[512, 120, 28, 28]", unsqueeze_582: "f32[1, 120, 1, 1]", squeeze_37: "f32[120]", primals_82: "f32[120]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_scalar: "b8[512, 120, 28, 28]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_self: "f32[512, 120, 28, 28]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_233);  le_scalar = full_default = getitem_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_dim_int_list: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[512, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_582);  convolution_14 = unsqueeze_582 = None
        mul_tensor: "f32[512, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[120]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 2.4912308673469386e-06);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[120]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 2.4912308673469386e-06);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_tensor_4: "f32[120]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_82);  squeeze_37 = primals_82 = None
        unsqueeze_default_6: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[512, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[512, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[512, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[512, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        return mul_tensor_7



def make_inputs():
    return [
    torch.randn(48168960, dtype=torch.float32, device='cuda').as_strided([512, 120, 28, 28], [94080, 1, 3360, 120]),  # relu_8
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(48168960, dtype=torch.float32, device='cuda').as_strided([512, 120, 28, 28], [94080, 1, 3360, 120]),  # getitem_233
    torch.randn(48168960, dtype=torch.float32, device='cuda').as_strided([512, 120, 28, 28], [94080, 1, 3360, 120]),  # convolution_14
    torch.randn([1, 120, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([120], dtype=torch.float32, device='cuda'),
    torch.randn([120], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
