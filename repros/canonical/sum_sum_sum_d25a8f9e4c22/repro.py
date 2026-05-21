"""
Standalone repro captured via capture_hook.
Label: torchbench_functorch_dp_cifar10_train
Pattern hash: d25a8f9e4c22
Shape hash: fd648f47
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([64, 128, 4, 4], f32, stride=(2048, 1, 512, 128)), T([64, 128, 4, 4], f32, stride=(2048, 1, 512, 128)), T([128], f32), T([64, 128], f32), T([64, 32], f32), T([64, 32], f32), T([64, 32, 4, 16], f32, stride=(2048, 4, 1, 128)), S([64, 128, 16]), S([64, 32, 4]), S([64, 32, 4]), S([1, 32, 4]), S([64, 32, 4, 16]), S([64, 128, 4, 4]))"

class Repro(torch.nn.Module):
    def forward(self, where_10: "f32[64, 128, 4, 4]", convolution_7: "f32[64, 128, 4, 4]", primals_24: "f32[128]", sum_75: "f32[64, 128]", squeeze_15: "f32[64, 32]", squeeze_14: "f32[64, 32]", view_180: "f32[64, 32, 4, 16]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        mul_tensor: "f32[64, 128, 4, 4]" = torch.ops.aten.mul.Tensor(where_10, convolution_7);  where_10 = None
        reshape_default: "f32[64, 128, 16]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_0);  mul_tensor = _shape_param_0 = None
        sum_dim_int_list: "f32[64, 128]" = torch.ops.aten.sum.dim_IntList(reshape_default, [2]);  reshape_default = None
        unsqueeze_default: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(primals_24, 0)
        mul_tensor_1: "f32[64, 128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, unsqueeze_default);  sum_dim_int_list = None
        reshape_default_1: "f32[64, 32, 4]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_1);  mul_tensor_1 = _shape_param_1 = None
        sum_dim_int_list_1: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(reshape_default_1, [2]);  reshape_default_1 = None
        mul_tensor_2: "f32[64, 128]" = torch.ops.aten.mul.Tensor(sum_75, unsqueeze_default);  sum_75 = unsqueeze_default = None
        reshape_default_2: "f32[64, 32, 4]" = torch.ops.aten.reshape.default(mul_tensor_2, _shape_param_2);  mul_tensor_2 = _shape_param_2 = None
        sum_dim_int_list_2: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(reshape_default_2, [2]);  reshape_default_2 = None
        unsqueeze_default_1: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_15, -1)
        reshape_default_3: "f32[1, 32, 4]" = torch.ops.aten.reshape.default(primals_24, _shape_param_3);  primals_24 = _shape_param_3 = None
        mul_tensor_3: "f32[64, 32, 4]" = torch.ops.aten.mul.Tensor(unsqueeze_default_1, reshape_default_3);  unsqueeze_default_1 = reshape_default_3 = None
        mul_tensor_4: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, squeeze_14)
        sub_tensor: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_tensor_4, sum_dim_int_list_1);  mul_tensor_4 = sum_dim_int_list_1 = None
        mul_tensor_5: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sub_tensor, squeeze_15);  sub_tensor = None
        mul_tensor_6: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_tensor_5, squeeze_15);  mul_tensor_5 = None
        mul_tensor_7: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_tensor_6, squeeze_15);  mul_tensor_6 = None
        mul_tensor_8: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.015625);  mul_tensor_7 = None
        neg_default: "f32[64, 32]" = torch.ops.aten.neg.default(mul_tensor_8)
        mul_tensor_9: "f32[64, 32]" = torch.ops.aten.mul.Tensor(neg_default, squeeze_14);  neg_default = squeeze_14 = None
        mul_tensor_10: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, squeeze_15);  sum_dim_int_list_2 = squeeze_15 = None
        mul_tensor_11: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_tensor_10, 0.015625);  mul_tensor_10 = None
        sub_tensor_1: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_tensor_9, mul_tensor_11);  mul_tensor_9 = mul_tensor_11 = None
        unsqueeze_default_2: "f32[64, 32, 4, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, -1);  mul_tensor_3 = None
        unsqueeze_default_3: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_8, -1);  mul_tensor_8 = None
        unsqueeze_default_4: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, -1);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_tensor_1, -1);  sub_tensor_1 = None
        unsqueeze_default_6: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, -1);  unsqueeze_default_5 = None
        mul_tensor_12: "f32[64, 32, 4, 16]" = torch.ops.aten.mul.Tensor(view_180, unsqueeze_default_2);  view_180 = unsqueeze_default_2 = None
        reshape_default_4: "f32[64, 32, 4, 16]" = torch.ops.aten.reshape.default(convolution_7, _shape_param_4);  convolution_7 = _shape_param_4 = None
        mul_tensor_13: "f32[64, 32, 4, 16]" = torch.ops.aten.mul.Tensor(reshape_default_4, unsqueeze_default_4);  reshape_default_4 = unsqueeze_default_4 = None
        add_tensor: "f32[64, 32, 4, 16]" = torch.ops.aten.add.Tensor(mul_tensor_12, mul_tensor_13);  mul_tensor_12 = mul_tensor_13 = None
        add_tensor_1: "f32[64, 32, 4, 16]" = torch.ops.aten.add.Tensor(add_tensor, unsqueeze_default_6);  add_tensor = unsqueeze_default_6 = None
        reshape_default_5: "f32[64, 128, 4, 4]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_5);  add_tensor_1 = _shape_param_5 = None
        return reshape_default_5



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
