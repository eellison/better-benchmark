"""
Standalone repro captured via capture_hook.
Label: timm_inception_v3_train
Pattern hash: 0c9c964d4069
Shape hash: eb6a87e2
"""
_shapes_config = "(T([128, 32, 147, 147], f32, stride=(691488, 1, 4704, 32)), T([], f32), T([128, 32, 147, 147], f32, stride=(691488, 1, 4704, 32)), T([128, 32, 147, 147], f32, stride=(691488, 1, 4704, 32)), T([1, 32, 1, 1], f32), T([32], f32), T([32], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, relu_1: "f32[128, 32, 147, 147]", full_default: "f32[]", getitem_469: "f32[128, 32, 147, 147]", convolution_1: "f32[128, 32, 147, 147]", unsqueeze_1482: "f32[1, 32, 1, 1]", squeeze_4: "f32[32]", primals_12: "f32[32]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_scalar: "b8[128, 32, 147, 147]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_self: "f32[128, 32, 147, 147]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_469);  le_scalar = full_default = getitem_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_dim_int_list: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[128, 32, 147, 147]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1482);  convolution_1 = unsqueeze_1482 = None
        mul_tensor: "f32[128, 32, 147, 147]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 3.6153917349252627e-07);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 3.6153917349252627e-07);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_tensor_4: "f32[32]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_12);  squeeze_4 = primals_12 = None
        unsqueeze_default_6: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[128, 32, 147, 147]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[128, 32, 147, 147]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[128, 32, 147, 147]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[128, 32, 147, 147]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        return mul_tensor_7



def make_inputs():
    return [
    torch.randn(88510464, dtype=torch.float32, device='cuda').as_strided([128, 32, 147, 147], [691488, 1, 4704, 32]),  # relu_1
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(88510464, dtype=torch.float32, device='cuda').as_strided([128, 32, 147, 147], [691488, 1, 4704, 32]),  # getitem_469
    torch.randn(88510464, dtype=torch.float32, device='cuda').as_strided([128, 32, 147, 147], [691488, 1, 4704, 32]),  # convolution_1
    torch.randn([1, 32, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32], dtype=torch.float32, device='cuda'),
    torch.randn([32], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
