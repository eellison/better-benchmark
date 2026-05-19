"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_train
Pattern hash: d27541d3a0c8
Shape hash: e4e4a480
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, convolution_12: "f32[512, 72, 1, 1]", getitem_239: "f32[512, 72, 28, 28]", getitem_245: "f32[512, 72, 1, 1]", relu_6: "f32[512, 72, 28, 28]", full_default: "f32[]", getitem_21: "f32[1, 72, 1, 1]", convolution_10: "f32[512, 72, 28, 28]", rsqrt_10: "f32[1, 72, 1, 1]", primals_66: "f32[72]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        add_tensor: "f32[512, 72, 1, 1]" = torch.ops.aten.add.Tensor(convolution_12, 3);  convolution_12 = None
        clamp_min_default: "f32[512, 72, 1, 1]" = torch.ops.aten.clamp_min.default(add_tensor, 0);  add_tensor = None
        clamp_max_default: "f32[512, 72, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        div_tensor: "f32[512, 72, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_default, 6);  clamp_max_default = None
        mul_tensor: "f32[512, 72, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_239, div_tensor);  getitem_239 = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_default: "f32[512, 72, 28, 28]" = torch.ops.aten.expand.default(getitem_245, _shape_param_0);  getitem_245 = _shape_param_0 = None
        div_scalar: "f32[512, 72, 28, 28]" = torch.ops.aten.div.Scalar(expand_default, 784);  expand_default = None
        add_tensor_1: "f32[512, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor, div_scalar);  mul_tensor = div_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_scalar: "b8[512, 72, 28, 28]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_self: "f32[512, 72, 28, 28]" = torch.ops.aten.where.self(le_scalar, full_default, add_tensor_1);  le_scalar = full_default = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        unsqueeze_default: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_1: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        sum_dim_int_list: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor: "f32[512, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_default_2);  convolution_10 = unsqueeze_default_2 = None
        mul_tensor_1: "f32[512, 72, 28, 28]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor)
        sum_dim_int_list_1: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 2, 3]);  mul_tensor_1 = None
        mul_tensor_2: "f32[72]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 2.4912308673469386e-06);  sum_dim_int_list = None
        unsqueeze_default_3: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_tensor_2, 0);  mul_tensor_2 = None
        unsqueeze_default_4: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_3: "f32[72]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 2.4912308673469386e-06);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_tensor_4: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_5: "f32[72]" = torch.ops.aten.mul.Tensor(mul_tensor_3, mul_tensor_4);  mul_tensor_3 = mul_tensor_4 = None
        unsqueeze_default_6: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_66);  squeeze_dims_1 = primals_66 = None
        unsqueeze_default_9: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_10: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None
        mul_tensor_7: "f32[512, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_8);  sub_tensor = unsqueeze_default_8 = None
        sub_tensor_1: "f32[512, 72, 28, 28]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_7);  where_self = mul_tensor_7 = None
        sub_tensor_2: "f32[512, 72, 28, 28]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_5);  sub_tensor_1 = unsqueeze_default_5 = None
        mul_tensor_8: "f32[512, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_11);  sub_tensor_2 = unsqueeze_default_11 = None
        return mul_tensor_8


def _default_make_inputs():
    return [
    torch.randn([512, 72, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(28901376, dtype=torch.float32, device='cuda').as_strided([512, 72, 28, 28], [56448, 1, 2016, 72]),  # getitem_239
    torch.randn([512, 72, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(28901376, dtype=torch.float32, device='cuda').as_strided([512, 72, 28, 28], [56448, 1, 2016, 72]),  # relu_6
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([1, 72, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn(28901376, dtype=torch.float32, device='cuda').as_strided([512, 72, 28, 28], [56448, 1, 2016, 72]),  # convolution_10
    torch.randn([1, 72, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([72], dtype=torch.float32, device='cuda'),
    [512, 72, 28, 28],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
