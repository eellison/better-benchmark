"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 77042b70d5f5
Shape hash: 3f38dae0
"""
_shapes_config = "(T([512, 960, 1, 1], f32), T([512, 960, 7, 7], f32, stride=(47040, 1, 6720, 960)), T([1, 960, 1, 1], f32), T([1, 960, 1, 1], f32), T([960], f32), T([960], f32), T([], f32), S([512, 960, 7, 7]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_160: "f32[512, 960, 1, 1]", convolution_93: "f32[512, 960, 7, 7]", getitem_159: "f32[1, 960, 1, 1]", rsqrt_79: "f32[1, 960, 1, 1]", primals_508: "f32[960]", primals_509: "f32[960]", full_default: "f32[]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        squeeze_dim: "f32[512, 960, 1]" = torch.ops.aten.squeeze.dim(getitem_160, 3);  getitem_160 = None
        squeeze_dim_1: "f32[512, 960]" = torch.ops.aten.squeeze.dim(squeeze_dim, 2);  squeeze_dim = None
        full_default_1: "f32[491520]" = torch.ops.aten.full.default([491520], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        as_strided_scatter_default: "f32[491520]" = torch.ops.aten.as_strided_scatter.default(full_default_1, squeeze_dim_1, [512, 960], [960, 1], 0);  full_default_1 = squeeze_dim_1 = None
        as_strided_default: "f32[512, 960, 1, 1]" = torch.ops.aten.as_strided.default(as_strided_scatter_default, [512, 960, 1, 1], [960, 1, 1, 1], 0);  as_strided_scatter_default = None
        expand_default: "f32[512, 960, 7, 7]" = torch.ops.aten.expand.default(as_strided_default, _shape_param_0);  as_strided_default = _shape_param_0 = None
        div_scalar: "f32[512, 960, 7, 7]" = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_tensor: "f32[512, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_93, getitem_159)
        mul_tensor: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_79);  sub_tensor = None
        unsqueeze_default: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_508, -1)
        unsqueeze_default_1: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_509, -1);  primals_509 = None
        unsqueeze_default_3: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[512, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[512, 960, 7, 7]" = torch.ops.aten.relu.default(add_tensor);  add_tensor = None
        le_scalar: "b8[512, 960, 7, 7]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self: "f32[512, 960, 7, 7]" = torch.ops.aten.where.self(le_scalar, full_default, div_scalar);  le_scalar = full_default = div_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_159, [0, 2, 3]);  getitem_159 = None
        unsqueeze_default_4: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_1: "f32[512, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_default_6);  convolution_93 = unsqueeze_default_6 = None
        mul_tensor_2: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_1)
        sum_dim_int_list_1: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3]);  mul_tensor_2 = None
        mul_tensor_3: "f32[960]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 3.985969387755102e-05);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 0);  mul_tensor_3 = None
        unsqueeze_default_8: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_4: "f32[960]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 3.985969387755102e-05);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[960]" = torch.ops.aten.squeeze.dims(rsqrt_79, [0, 2, 3]);  rsqrt_79 = None
        mul_tensor_5: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_6: "f32[960]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default_10: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_11: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_7: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_508);  squeeze_dims_1 = primals_508 = None
        unsqueeze_default_13: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_14: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_8: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[512, 960, 7, 7]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_8);  where_self = mul_tensor_8 = None
        sub_tensor_3: "f32[512, 960, 7, 7]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_9: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        return mul_tensor_9



def make_inputs():
    return [
    torch.randn([512, 960, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(24084480, dtype=torch.float32, device='cuda').as_strided([512, 960, 7, 7], [47040, 1, 6720, 960]),  # convolution_93
    torch.randn([1, 960, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 960, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
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
