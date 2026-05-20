"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 26b2c6c2c4b5
Shape hash: f5e9c73b
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([512, 72, 1, 1], f32), T([512, 72, 28, 28], f32, stride=(56448, 1, 2016, 72)), T([512, 72, 1, 1], f32), T([1, 72, 1, 1], f32), T([512, 72, 28, 28], f32, stride=(56448, 1, 2016, 72)), T([1, 72, 1, 1], f32), T([72], f32), S([512, 72, 28, 28]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_20: "f32[512, 72, 1, 1]", getitem_379: "f32[512, 72, 28, 28]", getitem_385: "f32[512, 72, 1, 1]", getitem_37: "f32[1, 72, 1, 1]", convolution_18: "f32[512, 72, 28, 28]", rsqrt_18: "f32[1, 72, 1, 1]", primals_114: "f32[72]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        add_tensor: "f32[512, 72, 1, 1]" = torch.ops.aten.add.Tensor(convolution_20, 3);  convolution_20 = None
        clamp_min_default: "f32[512, 72, 1, 1]" = torch.ops.aten.clamp_min.default(add_tensor, 0);  add_tensor = None
        clamp_max_default: "f32[512, 72, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        div_tensor: "f32[512, 72, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_default, 6);  clamp_max_default = None
        mul_tensor: "f32[512, 72, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_379, div_tensor);  getitem_379 = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_default: "f32[512, 72, 28, 28]" = torch.ops.aten.expand.default(getitem_385, _shape_param_0);  getitem_385 = _shape_param_0 = None
        div_scalar: "f32[512, 72, 28, 28]" = torch.ops.aten.div.Scalar(expand_default, 784);  expand_default = None
        add_tensor_1: "f32[512, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor, div_scalar);  mul_tensor = div_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:436 in forward, code: x = self.bn_dw(x)
        squeeze_dims: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        unsqueeze_default: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_1: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        sum_dim_int_list: "f32[72]" = torch.ops.aten.sum.dim_IntList(add_tensor_1, [0, 2, 3])
        sub_tensor: "f32[512, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_default_2);  convolution_18 = unsqueeze_default_2 = None
        mul_tensor_1: "f32[512, 72, 28, 28]" = torch.ops.aten.mul.Tensor(add_tensor_1, sub_tensor)
        sum_dim_int_list_1: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 2, 3]);  mul_tensor_1 = None
        mul_tensor_2: "f32[72]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 2.4912308673469386e-06);  sum_dim_int_list = None
        unsqueeze_default_3: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_tensor_2, 0);  mul_tensor_2 = None
        unsqueeze_default_4: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_3: "f32[72]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 2.4912308673469386e-06);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_tensor_4: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_5: "f32[72]" = torch.ops.aten.mul.Tensor(mul_tensor_3, mul_tensor_4);  mul_tensor_3 = mul_tensor_4 = None
        unsqueeze_default_6: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_114);  squeeze_dims_1 = primals_114 = None
        unsqueeze_default_9: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_10: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None
        mul_tensor_7: "f32[512, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_8);  sub_tensor = unsqueeze_default_8 = None
        sub_tensor_1: "f32[512, 72, 28, 28]" = torch.ops.aten.sub.Tensor(add_tensor_1, mul_tensor_7);  add_tensor_1 = mul_tensor_7 = None
        sub_tensor_2: "f32[512, 72, 28, 28]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_5);  sub_tensor_1 = unsqueeze_default_5 = None
        mul_tensor_8: "f32[512, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_11);  sub_tensor_2 = unsqueeze_default_11 = None
        return mul_tensor_8


def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
