"""
Standalone repro captured via capture_hook.
Label: timm_tf_efficientnet_b0_train
Pattern hash: 3148fac8fafd
Shape hash: 10afea94
"""
_shapes_config = "(T([128, 240, 28, 28], f32, stride=(188160, 1, 6720, 240)), T([1, 240, 1, 1], f32), T([1, 240, 1, 1], f32), T([240], f32), T([240], f32), T([128, 240, 28, 28], f32, stride=(188160, 1, 6720, 240)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_20: "f32[128, 240, 28, 28]", getitem_25: "f32[1, 240, 1, 1]", rsqrt_12: "f32[1, 240, 1, 1]", primals_94: "f32[240]", primals_95: "f32[240]", getitem_275: "f32[128, 240, 28, 28]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_tensor: "f32[128, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_25)
        mul_tensor: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_12);  sub_tensor = None
        unsqueeze_default: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_94, -1)
        unsqueeze_default_1: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1);  primals_95 = None
        unsqueeze_default_3: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[128, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_default: "f32[128, 240, 28, 28]" = torch.ops.aten.neg.default(add_tensor)
        exp_default: "f32[128, 240, 28, 28]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_1: "f32[128, 240, 28, 28]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        reciprocal_default: "f32[128, 240, 28, 28]" = torch.ops.aten.reciprocal.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_2: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_3: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_275, mul_tensor_2);  getitem_275 = None
        sub_tensor_1: "f32[128, 240, 28, 28]" = torch.ops.aten.sub.Tensor(1, mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_4: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_tensor, sub_tensor_1);  add_tensor = sub_tensor_1 = None
        add_tensor_2: "f32[128, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_4, 1);  mul_tensor_4 = None
        mul_tensor_5: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor_3, add_tensor_2);  mul_tensor_3 = add_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        unsqueeze_default_4: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 2, 3])
        sub_tensor_2: "f32[128, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_default_6);  convolution_20 = unsqueeze_default_6 = None
        mul_tensor_6: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor_5, sub_tensor_2)
        sum_dim_int_list_1: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 2, 3]);  mul_tensor_6 = None
        mul_tensor_7: "f32[240]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 9.964923469387754e-06);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_8: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_8: "f32[240]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 9.964923469387754e-06);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_tensor_9: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_10: "f32[240]" = torch.ops.aten.mul.Tensor(mul_tensor_8, mul_tensor_9);  mul_tensor_8 = mul_tensor_9 = None
        unsqueeze_default_10: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_tensor_10, 0);  mul_tensor_10 = None
        unsqueeze_default_11: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_11: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_94);  squeeze_dims_1 = primals_94 = None
        unsqueeze_default_13: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_tensor_11, 0);  mul_tensor_11 = None
        unsqueeze_default_14: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_12: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_12);  sub_tensor_2 = unsqueeze_default_12 = None
        sub_tensor_3: "f32[128, 240, 28, 28]" = torch.ops.aten.sub.Tensor(mul_tensor_5, mul_tensor_12);  mul_tensor_5 = mul_tensor_12 = None
        sub_tensor_4: "f32[128, 240, 28, 28]" = torch.ops.aten.sub.Tensor(sub_tensor_3, unsqueeze_default_9);  sub_tensor_3 = unsqueeze_default_9 = None
        mul_tensor_13: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_4, unsqueeze_default_15);  sub_tensor_4 = unsqueeze_default_15 = None
        return mul_tensor_13



def make_inputs():
    return [
    torch.randn(24084480, dtype=torch.float32, device='cuda').as_strided([128, 240, 28, 28], [188160, 1, 6720, 240]),  # convolution_20
    torch.randn([1, 240, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 240, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn(24084480, dtype=torch.float32, device='cuda').as_strided([128, 240, 28, 28], [188160, 1, 6720, 240]),  # getitem_275
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
