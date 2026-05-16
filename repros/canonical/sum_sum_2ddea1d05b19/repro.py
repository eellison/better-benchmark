"""
Standalone repro captured via capture_hook.
Label: efficientnet_b0_training
Pattern hash: 2ddea1d05b19
Shape hash: d64242f4
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[4, 1280]", _shape_param_0, _shape_param_1, lt_9: "b8[4, 1280]", _shape_param_2, _shape_param_3, convolution_80: "f32[4, 1280, 7, 7]", getitem_97: "f32[1, 1280, 1, 1]", rsqrt_48: "f32[1, 1280, 1, 1]", primals_358: "f32[1280]", primals_359: "f32[1280]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:339 in _forward_impl, code: x = self.classifier(x)
        reshape_default: "f32[4, 1280, 1, 1]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        reshape_default_1: "f32[4, 1280]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        convert_element_type_default: "f32[4, 1280]" = torch.ops.prims.convert_element_type.default(lt_9, torch.float32);  lt_9 = None
        div_scalar: "f32[4, 1280]" = torch.ops.aten.div.Scalar(convert_element_type_default, 0.8);  convert_element_type_default = None
        mul_tensor: "f32[4, 1280]" = torch.ops.aten.mul.Tensor(reshape_default_1, div_scalar);  reshape_default_1 = div_scalar = None
        reshape_default_2: "f32[4, 1280, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_2);  mul_tensor = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:336 in _forward_impl, code: x = self.avgpool(x)
        expand_default: "f32[4, 1280, 7, 7]" = torch.ops.aten.expand.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        div_scalar_1: "f32[4, 1280, 7, 7]" = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:334 in _forward_impl, code: x = self.features(x)
        sub_tensor: "f32[4, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_80, getitem_97)
        mul_tensor_1: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_48);  sub_tensor = None
        unsqueeze_default: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_358, -1)
        unsqueeze_default_1: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_2: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_1);  mul_tensor_1 = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_359, -1);  primals_359 = None
        unsqueeze_default_3: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[4, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_3);  mul_tensor_2 = unsqueeze_default_3 = None
        neg_default: "f32[4, 1280, 7, 7]" = torch.ops.aten.neg.default(add_tensor)
        exp_default: "f32[4, 1280, 7, 7]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_1: "f32[4, 1280, 7, 7]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        reciprocal_default: "f32[4, 1280, 7, 7]" = torch.ops.aten.reciprocal.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_3: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_4: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(div_scalar_1, mul_tensor_3);  div_scalar_1 = None
        sub_tensor_1: "f32[4, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(1, mul_tensor_3);  mul_tensor_3 = None
        mul_tensor_5: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(add_tensor, sub_tensor_1);  add_tensor = sub_tensor_1 = None
        add_tensor_2: "f32[4, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_5, 1);  mul_tensor_5 = None
        mul_tensor_6: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_4, add_tensor_2);  mul_tensor_4 = add_tensor_2 = None
        squeeze_dims: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
        unsqueeze_default_4: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 2, 3])
        sub_tensor_2: "f32[4, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_default_6);  convolution_80 = unsqueeze_default_6 = None
        mul_tensor_7: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_6, sub_tensor_2)
        sum_dim_int_list_1: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 2, 3]);  mul_tensor_7 = None
        mul_tensor_8: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.00510204081632653);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_tensor_8, 0);  mul_tensor_8 = None
        unsqueeze_default_8: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_9: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.00510204081632653);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[1280]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
        mul_tensor_10: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_11: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_tensor_9, mul_tensor_10);  mul_tensor_9 = mul_tensor_10 = None
        unsqueeze_default_10: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_tensor_11, 0);  mul_tensor_11 = None
        unsqueeze_default_11: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_12: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_358);  squeeze_dims_1 = primals_358 = None
        unsqueeze_default_13: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_tensor_12, 0);  mul_tensor_12 = None
        unsqueeze_default_14: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_13: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_12);  sub_tensor_2 = unsqueeze_default_12 = None
        sub_tensor_3: "f32[4, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(mul_tensor_6, mul_tensor_13);  mul_tensor_6 = mul_tensor_13 = None
        sub_tensor_4: "f32[4, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(sub_tensor_3, unsqueeze_default_9);  sub_tensor_3 = unsqueeze_default_9 = None
        mul_tensor_14: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_4, unsqueeze_default_15);  sub_tensor_4 = unsqueeze_default_15 = None
        return mul_tensor_14


def _default_make_inputs():
    return [
    torch.randn([4, 1280], dtype=torch.float32, device='cuda'),
    [4, 1280, 1, 1],  # _shape_param_0
    [4, 1280],  # _shape_param_1
    torch.randint(0, 2, [4, 1280], dtype=torch.bool, device='cuda'),
    [4, 1280, 1, 1],  # _shape_param_2
    [4, 1280, 7, 7],  # _shape_param_3
    torch.randn(250880, dtype=torch.float32, device='cuda').as_strided([4, 1280, 7, 7], [62720, 1, 8960, 1280]),  # convolution_80
    torch.randn([1, 1280, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1280, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1280], dtype=torch.float32, device='cuda'),
    torch.randn([1280], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
