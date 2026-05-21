"""
Standalone repro captured via capture_hook.
Label: timm_inception_v3_train
Pattern hash: 0ce0fdec34cd
Shape hash: dd941ad0
"""
_shapes_config = "(T([128, 192, 17, 17], f32, stride=(55488, 1, 3264, 192)), T([], f32), T([128, 192, 17, 17], f32, stride=(55488, 1, 3264, 192)), T([128, 192, 17, 17], f32, stride=(55488, 1, 3264, 192)), T([1, 192, 1, 1], f32), T([192], f32), T([192], f32), T([128, 192, 17, 17], f32, stride=(55488, 1, 3264, 192)), T([128, 192, 17, 17], f32, stride=(55488, 1, 3264, 192)), T([128, 192, 17, 17], f32, stride=(55488, 1, 3264, 192)), T([1, 192, 1, 1], f32), T([192], f32), T([192], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, relu_65: "f32[128, 192, 17, 17]", full_default: "f32[]", getitem_277: "f32[128, 192, 17, 17]", convolution_65: "f32[128, 192, 17, 17]", unsqueeze_714: "f32[1, 192, 1, 1]", squeeze_196: "f32[192]", primals_396: "f32[192]", relu_62: "f32[128, 192, 17, 17]", getitem_286: "f32[128, 192, 17, 17]", convolution_62: "f32[128, 192, 17, 17]", unsqueeze_750: "f32[1, 192, 1, 1]", squeeze_187: "f32[192]", primals_378: "f32[192]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_scalar: "b8[128, 192, 17, 17]" = torch.ops.aten.le.Scalar(relu_65, 0);  relu_65 = None
        where_self: "f32[128, 192, 17, 17]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_277);  le_scalar = getitem_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_dim_int_list: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_714);  convolution_65 = unsqueeze_714 = None
        mul_tensor: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 2.703287197231834e-05);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 2.703287197231834e-05);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_196, squeeze_196)
        mul_tensor_4: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_196, primals_396);  squeeze_196 = primals_396 = None
        unsqueeze_default_6: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_scalar_1: "b8[128, 192, 17, 17]" = torch.ops.aten.le.Scalar(relu_62, 0);  relu_62 = None
        where_self_1: "f32[128, 192, 17, 17]" = torch.ops.aten.where.self(le_scalar_1, full_default, getitem_286);  le_scalar_1 = full_default = getitem_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_dim_int_list_2: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_self_1, [0, 2, 3])
        sub_tensor_3: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_750);  convolution_62 = unsqueeze_750 = None
        mul_tensor_8: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(where_self_1, sub_tensor_3)
        sum_dim_int_list_3: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 2, 3]);  mul_tensor_8 = None
        mul_tensor_9: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 2.703287197231834e-05);  sum_dim_int_list_2 = None
        unsqueeze_default_9: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, 0);  mul_tensor_9 = None
        unsqueeze_default_10: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None
        mul_tensor_10: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 2.703287197231834e-05);  sum_dim_int_list_3 = None
        mul_tensor_11: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
        mul_tensor_12: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
        unsqueeze_default_12: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_12, 0);  mul_tensor_12 = None
        unsqueeze_default_13: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 2);  unsqueeze_default_12 = None
        unsqueeze_default_14: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 3);  unsqueeze_default_13 = None
        mul_tensor_13: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_187, primals_378);  squeeze_187 = primals_378 = None
        unsqueeze_default_15: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_13, 0);  mul_tensor_13 = None
        unsqueeze_default_16: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 2);  unsqueeze_default_15 = None
        unsqueeze_default_17: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 3);  unsqueeze_default_16 = None
        mul_tensor_14: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_14);  sub_tensor_3 = unsqueeze_default_14 = None
        sub_tensor_4: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(where_self_1, mul_tensor_14);  where_self_1 = mul_tensor_14 = None
        sub_tensor_5: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(sub_tensor_4, unsqueeze_default_11);  sub_tensor_4 = unsqueeze_default_11 = None
        mul_tensor_15: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_17);  sub_tensor_5 = unsqueeze_default_17 = None
        return (mul_tensor_7, mul_tensor_15)



def make_inputs():
    return [
    torch.randn(7102464, dtype=torch.float32, device='cuda').as_strided([128, 192, 17, 17], [55488, 1, 3264, 192]),  # relu_65
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(7102464, dtype=torch.float32, device='cuda').as_strided([128, 192, 17, 17], [55488, 1, 3264, 192]),  # getitem_277
    torch.randn(7102464, dtype=torch.float32, device='cuda').as_strided([128, 192, 17, 17], [55488, 1, 3264, 192]),  # convolution_65
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn(7102464, dtype=torch.float32, device='cuda').as_strided([128, 192, 17, 17], [55488, 1, 3264, 192]),  # relu_62
    torch.randn(7102464, dtype=torch.float32, device='cuda').as_strided([128, 192, 17, 17], [55488, 1, 3264, 192]),  # getitem_286
    torch.randn(7102464, dtype=torch.float32, device='cuda').as_strided([128, 192, 17, 17], [55488, 1, 3264, 192]),  # convolution_62
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
