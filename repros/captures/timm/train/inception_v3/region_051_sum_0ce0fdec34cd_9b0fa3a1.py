"""
Standalone repro captured via capture_hook.
Label: timm_inception_v3_train
Pattern hash: 0ce0fdec34cd
Shape hash: 9b0fa3a1
"""
_shapes_config = "(T([128, 160, 17, 17], f32, stride=(46240, 1, 2720, 160)), T([], f32), T([128, 160, 17, 17], f32, stride=(46240, 1, 2720, 160)), T([128, 160, 17, 17], f32, stride=(46240, 1, 2720, 160)), T([1, 160, 1, 1], f32), T([160], f32), T([160], f32), T([128, 160, 17, 17], f32, stride=(46240, 1, 2720, 160)), T([128, 160, 17, 17], f32, stride=(46240, 1, 2720, 160)), T([128, 160, 17, 17], f32, stride=(46240, 1, 2720, 160)), T([1, 160, 1, 1], f32), T([160], f32), T([160], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, relu_45: "f32[128, 160, 17, 17]", full_default: "f32[]", getitem_337: "f32[128, 160, 17, 17]", convolution_45: "f32[128, 160, 17, 17]", unsqueeze_954: "f32[1, 160, 1, 1]", squeeze_136: "f32[160]", primals_276: "f32[160]", relu_42: "f32[128, 160, 17, 17]", getitem_346: "f32[128, 160, 17, 17]", convolution_42: "f32[128, 160, 17, 17]", unsqueeze_990: "f32[1, 160, 1, 1]", squeeze_127: "f32[160]", primals_258: "f32[160]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_scalar: "b8[128, 160, 17, 17]" = torch.ops.aten.le.Scalar(relu_45, 0);  relu_45 = None
        where_self: "f32[128, 160, 17, 17]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_337);  le_scalar = getitem_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_dim_int_list: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[128, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_954);  convolution_45 = unsqueeze_954 = None
        mul_tensor: "f32[128, 160, 17, 17]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[160]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 2.703287197231834e-05);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[160]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 2.703287197231834e-05);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
        mul_tensor_4: "f32[160]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_276);  squeeze_136 = primals_276 = None
        unsqueeze_default_6: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[128, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[128, 160, 17, 17]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[128, 160, 17, 17]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[128, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_scalar_1: "b8[128, 160, 17, 17]" = torch.ops.aten.le.Scalar(relu_42, 0);  relu_42 = None
        where_self_1: "f32[128, 160, 17, 17]" = torch.ops.aten.where.self(le_scalar_1, full_default, getitem_346);  le_scalar_1 = full_default = getitem_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_dim_int_list_2: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_self_1, [0, 2, 3])
        sub_tensor_3: "f32[128, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_990);  convolution_42 = unsqueeze_990 = None
        mul_tensor_8: "f32[128, 160, 17, 17]" = torch.ops.aten.mul.Tensor(where_self_1, sub_tensor_3)
        sum_dim_int_list_3: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 2, 3]);  mul_tensor_8 = None
        mul_tensor_9: "f32[160]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 2.703287197231834e-05);  sum_dim_int_list_2 = None
        unsqueeze_default_9: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, 0);  mul_tensor_9 = None
        unsqueeze_default_10: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None
        mul_tensor_10: "f32[160]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 2.703287197231834e-05);  sum_dim_int_list_3 = None
        mul_tensor_11: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
        mul_tensor_12: "f32[160]" = torch.ops.aten.mul.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
        unsqueeze_default_12: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_tensor_12, 0);  mul_tensor_12 = None
        unsqueeze_default_13: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 2);  unsqueeze_default_12 = None
        unsqueeze_default_14: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 3);  unsqueeze_default_13 = None
        mul_tensor_13: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_258);  squeeze_127 = primals_258 = None
        unsqueeze_default_15: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_tensor_13, 0);  mul_tensor_13 = None
        unsqueeze_default_16: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 2);  unsqueeze_default_15 = None
        unsqueeze_default_17: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 3);  unsqueeze_default_16 = None
        mul_tensor_14: "f32[128, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_14);  sub_tensor_3 = unsqueeze_default_14 = None
        sub_tensor_4: "f32[128, 160, 17, 17]" = torch.ops.aten.sub.Tensor(where_self_1, mul_tensor_14);  where_self_1 = mul_tensor_14 = None
        sub_tensor_5: "f32[128, 160, 17, 17]" = torch.ops.aten.sub.Tensor(sub_tensor_4, unsqueeze_default_11);  sub_tensor_4 = unsqueeze_default_11 = None
        mul_tensor_15: "f32[128, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_17);  sub_tensor_5 = unsqueeze_default_17 = None
        return (mul_tensor_7, mul_tensor_15)



def make_inputs():
    return [
    torch.randn(5918720, dtype=torch.float32, device='cuda').as_strided([128, 160, 17, 17], [46240, 1, 2720, 160]),  # relu_45
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(5918720, dtype=torch.float32, device='cuda').as_strided([128, 160, 17, 17], [46240, 1, 2720, 160]),  # getitem_337
    torch.randn(5918720, dtype=torch.float32, device='cuda').as_strided([128, 160, 17, 17], [46240, 1, 2720, 160]),  # convolution_45
    torch.randn([1, 160, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn(5918720, dtype=torch.float32, device='cuda').as_strided([128, 160, 17, 17], [46240, 1, 2720, 160]),  # relu_42
    torch.randn(5918720, dtype=torch.float32, device='cuda').as_strided([128, 160, 17, 17], [46240, 1, 2720, 160]),  # getitem_346
    torch.randn(5918720, dtype=torch.float32, device='cuda').as_strided([128, 160, 17, 17], [46240, 1, 2720, 160]),  # convolution_42
    torch.randn([1, 160, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
