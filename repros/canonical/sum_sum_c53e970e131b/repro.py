"""
Standalone repro captured via capture_hook.
Label: mobilenet_v2_training
Pattern hash: c53e970e131b
Shape hash: d64242f4
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, gt: "b8[4, 1280]", mm: "f32[4, 1280]", _shape_param_0, _shape_param_1, convolution_51: "f32[4, 1280, 7, 7]", getitem_103: "f32[1, 1280, 1, 1]", rsqrt_51: "f32[1, 1280, 1, 1]", primals_312: "f32[1280]", primals_313: "f32[1280]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv2.py:170 in _forward_impl, code: x = self.classifier(x)
        convert_element_type_default: "f32[4, 1280]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor: "f32[4, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.25);  convert_element_type_default = None
        mul_tensor_1: "f32[4, 1280]" = torch.ops.aten.mul.Tensor(mm, mul_tensor);  mm = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv2.py:169 in _forward_impl, code: x = torch.flatten(x, 1)
        reshape_default: "f32[4, 1280, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_0);  mul_tensor_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv2.py:168 in _forward_impl, code: x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        expand_default: "f32[4, 1280, 7, 7]" = torch.ops.aten.expand.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        div_scalar: "f32[4, 1280, 7, 7]" = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv2.py:166 in _forward_impl, code: x = self.features(x)
        sub_tensor: "f32[4, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_103)
        mul_tensor_2: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_51);  sub_tensor = None
        unsqueeze_default: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_312, -1)
        unsqueeze_default_1: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_3: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_2, unsqueeze_default_1);  mul_tensor_2 = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_313, -1);  primals_313 = None
        unsqueeze_default_3: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[4, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_3, unsqueeze_default_3);  mul_tensor_3 = unsqueeze_default_3 = None
        le_scalar: "b8[4, 1280, 7, 7]" = torch.ops.aten.le.Scalar(add_tensor, 0.0)
        ge_scalar: "b8[4, 1280, 7, 7]" = torch.ops.aten.ge.Scalar(add_tensor, 6.0);  add_tensor = None
        bitwise_or_tensor: "b8[4, 1280, 7, 7]" = torch.ops.aten.bitwise_or.Tensor(le_scalar, ge_scalar);  le_scalar = ge_scalar = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[4, 1280, 7, 7]" = torch.ops.aten.where.self(bitwise_or_tensor, full_default, div_scalar);  bitwise_or_tensor = full_default = div_scalar = None
        squeeze_dims: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
        unsqueeze_default_4: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[1280]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_1: "f32[4, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_default_6);  convolution_51 = unsqueeze_default_6 = None
        mul_tensor_4: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_1)
        sum_dim_int_list_1: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 2, 3]);  mul_tensor_4 = None
        mul_tensor_5: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.00510204081632653);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_8: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_6: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.00510204081632653);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[1280]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
        mul_tensor_7: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_8: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_tensor_6, mul_tensor_7);  mul_tensor_6 = mul_tensor_7 = None
        unsqueeze_default_10: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_tensor_8, 0);  mul_tensor_8 = None
        unsqueeze_default_11: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_9: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_312);  squeeze_dims_1 = primals_312 = None
        unsqueeze_default_13: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, 0);  mul_tensor_9 = None
        unsqueeze_default_14: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_10: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[4, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_10);  where_self = mul_tensor_10 = None
        sub_tensor_3: "f32[4, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_11: "f32[4, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        return mul_tensor_11


def _default_make_inputs():
    return [
    torch.randint(0, 2, [4, 1280], dtype=torch.bool, device='cuda'),
    torch.randn([4, 1280], dtype=torch.float32, device='cuda'),
    [4, 1280, 1, 1],  # _shape_param_0
    [4, 1280, 7, 7],  # _shape_param_1
    torch.randn(250880, dtype=torch.float32, device='cuda').as_strided([4, 1280, 7, 7], [62720, 1, 8960, 1280]),  # convolution_51
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
