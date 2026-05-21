"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_train
Pattern hash: 08ddb588e2ef
Shape hash: 755723c5
"""
_shapes_config = "(T([512, 200, 14, 14], f32, stride=(39200, 1, 2800, 200)), T([1, 200, 1, 1], f32), T([1, 200, 1, 1], f32), T([200], f32), T([200], f32), T([512, 200, 14, 14], f32, stride=(39200, 1, 2800, 200)), T([], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_27: "f32[512, 200, 14, 14]", getitem_43: "f32[1, 200, 1, 1]", rsqrt_21: "f32[1, 200, 1, 1]", primals_144: "f32[200]", primals_145: "f32[200]", getitem_194: "f32[512, 200, 14, 14]", full_default: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_tensor: "f32[512, 200, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_43)
        mul_tensor: "f32[512, 200, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_21);  sub_tensor = None
        unsqueeze_default: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(primals_144, -1)
        unsqueeze_default_1: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 200, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(primals_145, -1);  primals_145 = None
        unsqueeze_default_3: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[512, 200, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_scalar: "b8[512, 200, 14, 14]" = torch.ops.aten.le.Scalar(add_tensor, -3)
        lt_scalar: "b8[512, 200, 14, 14]" = torch.ops.aten.lt.Scalar(add_tensor, 3)
        div_tensor: "f32[512, 200, 14, 14]" = torch.ops.aten.div.Tensor(add_tensor, 3);  add_tensor = None
        add_tensor_1: "f32[512, 200, 14, 14]" = torch.ops.aten.add.Tensor(div_tensor, 0.5);  div_tensor = None
        mul_tensor_2: "f32[512, 200, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_194, add_tensor_1);  add_tensor_1 = None
        where_self: "f32[512, 200, 14, 14]" = torch.ops.aten.where.self(lt_scalar, mul_tensor_2, getitem_194);  lt_scalar = mul_tensor_2 = getitem_194 = None
        where_self_1: "f32[512, 200, 14, 14]" = torch.ops.aten.where.self(le_scalar, full_default, where_self);  le_scalar = full_default = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims: "f32[200]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        unsqueeze_default_4: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[200]" = torch.ops.aten.sum.dim_IntList(where_self_1, [0, 2, 3])
        sub_tensor_1: "f32[512, 200, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_default_6);  convolution_27 = unsqueeze_default_6 = None
        mul_tensor_3: "f32[512, 200, 14, 14]" = torch.ops.aten.mul.Tensor(where_self_1, sub_tensor_1)
        sum_dim_int_list_1: "f32[200]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 2, 3]);  mul_tensor_3 = None
        mul_tensor_4: "f32[200]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 9.964923469387754e-06);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_8: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_5: "f32[200]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 9.964923469387754e-06);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[200]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
        mul_tensor_6: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_7: "f32[200]" = torch.ops.aten.mul.Tensor(mul_tensor_5, mul_tensor_6);  mul_tensor_5 = mul_tensor_6 = None
        unsqueeze_default_10: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_11: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_8: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_144);  squeeze_dims_1 = primals_144 = None
        unsqueeze_default_13: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_tensor_8, 0);  mul_tensor_8 = None
        unsqueeze_default_14: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_9: "f32[512, 200, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[512, 200, 14, 14]" = torch.ops.aten.sub.Tensor(where_self_1, mul_tensor_9);  where_self_1 = mul_tensor_9 = None
        sub_tensor_3: "f32[512, 200, 14, 14]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_10: "f32[512, 200, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        return mul_tensor_10



def make_inputs():
    return [
    torch.randn(20070400, dtype=torch.float32, device='cuda').as_strided([512, 200, 14, 14], [39200, 1, 2800, 200]),  # convolution_27
    torch.randn([1, 200, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 200, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([200], dtype=torch.float32, device='cuda'),
    torch.randn([200], dtype=torch.float32, device='cuda'),
    torch.randn(20070400, dtype=torch.float32, device='cuda').as_strided([512, 200, 14, 14], [39200, 1, 2800, 200]),  # getitem_194
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
