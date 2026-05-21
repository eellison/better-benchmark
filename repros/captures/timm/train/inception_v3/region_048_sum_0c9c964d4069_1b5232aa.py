"""
Standalone repro captured via capture_hook.
Label: timm_inception_v3_train
Pattern hash: 0c9c964d4069
Shape hash: 1b5232aa
"""
_shapes_config = "(T([128, 128, 17, 17], f32, stride=(36992, 1, 2176, 128)), T([], f32), T([128, 128, 17, 17], f32, stride=(36992, 1, 2176, 128)), T([128, 128, 17, 17], f32, stride=(36992, 1, 2176, 128)), T([1, 128, 1, 1], f32), T([128], f32), T([128], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, relu_37: "f32[128, 128, 17, 17]", full_default: "f32[]", getitem_361: "f32[128, 128, 17, 17]", convolution_37: "f32[128, 128, 17, 17]", unsqueeze_1050: "f32[1, 128, 1, 1]", squeeze_112: "f32[128]", primals_228: "f32[128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_scalar: "b8[128, 128, 17, 17]" = torch.ops.aten.le.Scalar(relu_37, 0);  relu_37 = None
        where_self: "f32[128, 128, 17, 17]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_361);  le_scalar = full_default = getitem_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_dim_int_list: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[128, 128, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_1050);  convolution_37 = unsqueeze_1050 = None
        mul_tensor: "f32[128, 128, 17, 17]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 2.703287197231834e-05);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 2.703287197231834e-05);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
        mul_tensor_4: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_228);  squeeze_112 = primals_228 = None
        unsqueeze_default_6: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[128, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[128, 128, 17, 17]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[128, 128, 17, 17]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[128, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        return mul_tensor_7



def make_inputs():
    return [
    torch.randn(4734976, dtype=torch.float32, device='cuda').as_strided([128, 128, 17, 17], [36992, 1, 2176, 128]),  # relu_37
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(4734976, dtype=torch.float32, device='cuda').as_strided([128, 128, 17, 17], [36992, 1, 2176, 128]),  # getitem_361
    torch.randn(4734976, dtype=torch.float32, device='cuda').as_strided([128, 128, 17, 17], [36992, 1, 2176, 128]),  # convolution_37
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
