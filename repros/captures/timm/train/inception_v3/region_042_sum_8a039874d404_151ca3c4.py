"""
Standalone repro captured via capture_hook.
Label: timm_inception_v3_train
Pattern hash: 8a039874d404
Shape hash: 151ca3c4
"""
_shapes_config = "(T([128, 768, 17, 17], f32, stride=(221952, 1, 13056, 768)), T([128, 64, 35, 35], f32, stride=(78400, 1, 2240, 64)), T([], f32), T([128, 64, 35, 35], f32, stride=(78400, 1, 2240, 64)), T([128, 64, 35, 35], f32, stride=(78400, 1, 2240, 64)), T([1, 64, 1, 1], f32), T([64], f32), T([64], f32), T([128, 384, 17, 17], f32, stride=(110976, 1, 6528, 384)), T([1, 384, 1, 1], f32), T([1, 384, 1, 1], f32), T([384], f32), T([384], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, add_493: "f32[128, 768, 17, 17]", relu_27: "f32[128, 64, 35, 35]", full_default: "f32[]", getitem_391: "f32[128, 64, 35, 35]", convolution_27: "f32[128, 64, 35, 35]", unsqueeze_1170: "f32[1, 64, 1, 1]", squeeze_82: "f32[64]", primals_168: "f32[64]", convolution_26: "f32[128, 384, 17, 17]", getitem_57: "f32[1, 384, 1, 1]", rsqrt_26: "f32[1, 384, 1, 1]", primals_162: "f32[384]", primals_163: "f32[384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:100 in forward, code: return torch.cat(outputs, 1)
        slice_tensor: "f32[128, 384, 17, 17]" = torch.ops.aten.slice.Tensor(add_493, 1, 0, 384);  add_493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_scalar: "b8[128, 64, 35, 35]" = torch.ops.aten.le.Scalar(relu_27, 0);  relu_27 = None
        where_self: "f32[128, 64, 35, 35]" = torch.ops.aten.where.self(le_scalar, full_default, getitem_391);  le_scalar = getitem_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_dim_int_list: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_1170);  convolution_27 = unsqueeze_1170 = None
        mul_tensor: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 6.3775510204081635e-06);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 6.3775510204081635e-06);  sum_dim_int_list_1 = None
        mul_tensor_3: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_tensor_4: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_168);  squeeze_82 = primals_168 = None
        unsqueeze_default_6: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_5);  sub_tensor = unsqueeze_default_5 = None
        sub_tensor_1: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_2: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_7: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_8);  sub_tensor_2 = unsqueeze_default_8 = None
        sub_tensor_3: "f32[128, 384, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_57)
        mul_tensor_8: "f32[128, 384, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_3, rsqrt_26);  sub_tensor_3 = None
        unsqueeze_default_9: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_162, -1)
        unsqueeze_default_10: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, -1);  unsqueeze_default_9 = None
        mul_tensor_9: "f32[128, 384, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor_8, unsqueeze_default_10);  mul_tensor_8 = unsqueeze_default_10 = None
        unsqueeze_default_11: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_163, -1);  primals_163 = None
        unsqueeze_default_12: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, -1);  unsqueeze_default_11 = None
        add_tensor: "f32[128, 384, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_9, unsqueeze_default_12);  mul_tensor_9 = unsqueeze_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[128, 384, 17, 17]" = torch.ops.aten.relu.default(add_tensor);  add_tensor = None
        le_scalar_1: "b8[128, 384, 17, 17]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self_1: "f32[128, 384, 17, 17]" = torch.ops.aten.where.self(le_scalar_1, full_default, slice_tensor);  le_scalar_1 = full_default = slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
        unsqueeze_default_13: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_14: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        sum_dim_int_list_2: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_self_1, [0, 2, 3])
        sub_tensor_4: "f32[128, 384, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_default_15);  convolution_26 = unsqueeze_default_15 = None
        mul_tensor_10: "f32[128, 384, 17, 17]" = torch.ops.aten.mul.Tensor(where_self_1, sub_tensor_4)
        sum_dim_int_list_3: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 2, 3]);  mul_tensor_10 = None
        mul_tensor_11: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 2.703287197231834e-05);  sum_dim_int_list_2 = None
        unsqueeze_default_16: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_11, 0);  mul_tensor_11 = None
        unsqueeze_default_17: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 2);  unsqueeze_default_16 = None
        unsqueeze_default_18: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_17, 3);  unsqueeze_default_17 = None
        mul_tensor_12: "f32[384]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 2.703287197231834e-05);  sum_dim_int_list_3 = None
        squeeze_dims_1: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
        mul_tensor_13: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_14: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_12, mul_tensor_13);  mul_tensor_12 = mul_tensor_13 = None
        unsqueeze_default_19: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_14, 0);  mul_tensor_14 = None
        unsqueeze_default_20: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_19, 2);  unsqueeze_default_19 = None
        unsqueeze_default_21: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 3);  unsqueeze_default_20 = None
        mul_tensor_15: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_162);  squeeze_dims_1 = primals_162 = None
        unsqueeze_default_22: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_tensor_15, 0);  mul_tensor_15 = None
        unsqueeze_default_23: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, 2);  unsqueeze_default_22 = None
        unsqueeze_default_24: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 3);  unsqueeze_default_23 = None
        mul_tensor_16: "f32[128, 384, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_4, unsqueeze_default_21);  sub_tensor_4 = unsqueeze_default_21 = None
        sub_tensor_5: "f32[128, 384, 17, 17]" = torch.ops.aten.sub.Tensor(where_self_1, mul_tensor_16);  where_self_1 = mul_tensor_16 = None
        sub_tensor_6: "f32[128, 384, 17, 17]" = torch.ops.aten.sub.Tensor(sub_tensor_5, unsqueeze_default_18);  sub_tensor_5 = unsqueeze_default_18 = None
        mul_tensor_17: "f32[128, 384, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_6, unsqueeze_default_24);  sub_tensor_6 = unsqueeze_default_24 = None
        return (mul_tensor_7, mul_tensor_17)



def make_inputs():
    return [
    torch.randn(28409856, dtype=torch.float32, device='cuda').as_strided([128, 768, 17, 17], [221952, 1, 13056, 768]),  # add_493
    torch.randn(10035200, dtype=torch.float32, device='cuda').as_strided([128, 64, 35, 35], [78400, 1, 2240, 64]),  # relu_27
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn(10035200, dtype=torch.float32, device='cuda').as_strided([128, 64, 35, 35], [78400, 1, 2240, 64]),  # getitem_391
    torch.randn(10035200, dtype=torch.float32, device='cuda').as_strided([128, 64, 35, 35], [78400, 1, 2240, 64]),  # convolution_27
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn(14204928, dtype=torch.float32, device='cuda').as_strided([128, 384, 17, 17], [110976, 1, 6528, 384]),  # convolution_26
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
