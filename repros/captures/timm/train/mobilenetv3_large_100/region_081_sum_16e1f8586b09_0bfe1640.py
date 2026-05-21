"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_train
Pattern hash: 16e1f8586b09
Shape hash: 0bfe1640
"""
_shapes_config = "(T([512, 960, 1, 1], f32), T([512, 960, 7, 7], f32, stride=(47040, 1, 6720, 960)), T([512, 960, 1, 1], f32), T([512, 960, 7, 7], f32, stride=(47040, 1, 6720, 960)), T([], f32), T([1, 960, 1, 1], f32), T([512, 960, 7, 7], f32, stride=(47040, 1, 6720, 960)), T([1, 960, 1, 1], f32), T([960], f32), S([512, 960, 7, 7]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_54: "f32[512, 960, 1, 1]", getitem_113: "f32[512, 960, 7, 7]", getitem_119: "f32[512, 960, 1, 1]", add_234: "f32[512, 960, 7, 7]", full_default: "f32[]", getitem_81: "f32[1, 960, 1, 1]", convolution_52: "f32[512, 960, 7, 7]", rsqrt_40: "f32[1, 960, 1, 1]", primals_270: "f32[960]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        add_tensor: "f32[512, 960, 1, 1]" = torch.ops.aten.add.Tensor(convolution_54, 3);  convolution_54 = None
        clamp_min_default: "f32[512, 960, 1, 1]" = torch.ops.aten.clamp_min.default(add_tensor, 0);  add_tensor = None
        clamp_max_default: "f32[512, 960, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        div_tensor: "f32[512, 960, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_default, 6);  clamp_max_default = None
        mul_tensor: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_113, div_tensor);  getitem_113 = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_default: "f32[512, 960, 7, 7]" = torch.ops.aten.expand.default(getitem_119, _shape_param_0);  getitem_119 = _shape_param_0 = None
        div_scalar: "f32[512, 960, 7, 7]" = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None
        add_tensor_1: "f32[512, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor, div_scalar);  mul_tensor = div_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_scalar: "b8[512, 960, 7, 7]" = torch.ops.aten.le.Scalar(add_234, -3)
        lt_scalar: "b8[512, 960, 7, 7]" = torch.ops.aten.lt.Scalar(add_234, 3)
        div_tensor_1: "f32[512, 960, 7, 7]" = torch.ops.aten.div.Tensor(add_234, 3);  add_234 = None
        add_tensor_2: "f32[512, 960, 7, 7]" = torch.ops.aten.add.Tensor(div_tensor_1, 0.5);  div_tensor_1 = None
        mul_tensor_1: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_tensor_1, add_tensor_2);  add_tensor_2 = None
        where_self: "f32[512, 960, 7, 7]" = torch.ops.aten.where.self(lt_scalar, mul_tensor_1, add_tensor_1);  lt_scalar = mul_tensor_1 = add_tensor_1 = None
        where_self_1: "f32[512, 960, 7, 7]" = torch.ops.aten.where.self(le_scalar, full_default, where_self);  le_scalar = full_default = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
        unsqueeze_default: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_1: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        sum_dim_int_list: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_self_1, [0, 2, 3])
        sub_tensor: "f32[512, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_default_2);  convolution_52 = unsqueeze_default_2 = None
        mul_tensor_2: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_self_1, sub_tensor)
        sum_dim_int_list_1: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3]);  mul_tensor_2 = None
        mul_tensor_3: "f32[960]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 3.985969387755102e-05);  sum_dim_int_list = None
        unsqueeze_default_3: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 0);  mul_tensor_3 = None
        unsqueeze_default_4: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_4: "f32[960]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 3.985969387755102e-05);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[960]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
        mul_tensor_5: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_6: "f32[960]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default_6: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_7: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_7: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_270);  squeeze_dims_1 = primals_270 = None
        unsqueeze_default_9: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_10: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None
        mul_tensor_8: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_8);  sub_tensor = unsqueeze_default_8 = None
        sub_tensor_1: "f32[512, 960, 7, 7]" = torch.ops.aten.sub.Tensor(where_self_1, mul_tensor_8);  where_self_1 = mul_tensor_8 = None
        sub_tensor_2: "f32[512, 960, 7, 7]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_5);  sub_tensor_1 = unsqueeze_default_5 = None
        mul_tensor_9: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_11);  sub_tensor_2 = unsqueeze_default_11 = None
        return mul_tensor_9



def make_inputs():
    return [
    torch.randn([512, 960, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(24084480, dtype=torch.float32, device='cuda').as_strided([512, 960, 7, 7], [47040, 1, 6720, 960]),  # getitem_113
    torch.randn([512, 960, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(24084480, dtype=torch.float32, device='cuda').as_strided([512, 960, 7, 7], [47040, 1, 6720, 960]),  # add_234
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([1, 960, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(24084480, dtype=torch.float32, device='cuda').as_strided([512, 960, 7, 7], [47040, 1, 6720, 960]),  # convolution_52
    torch.randn([1, 960, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    [512, 960, 7, 7],  # _shape_param_0
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
