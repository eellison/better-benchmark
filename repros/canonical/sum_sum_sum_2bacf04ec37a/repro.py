"""
Standalone repro captured via capture_hook.
Label: torchbench_functorch_dp_cifar10_train
Pattern hash: 2bacf04ec37a
Shape hash: bd904f60
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
_shapes_config = "(T([64, 512], f32), T([64, 512, 1, 1], b8), T([64, 512, 1, 1], f32), T([512], f32), T([64, 32], f32), T([64, 32], f32), S([64, 512, 1, 1]), S([64, 512, 1, 1]), S([64, 512, 1]), S([64, 512, 1]), S([64, 32, 16]), S([64, 32, 16]), S([1, 32, 16]), S([64, 32, 16, 1]), S([64, 32, 16, 1]), S([64, 512, 1, 1]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[64, 512]", le: "b8[64, 512, 1, 1]", convolution_19: "f32[64, 512, 1, 1]", primals_60: "f32[512]", squeeze_39: "f32[64, 32]", squeeze_38: "f32[64, 32]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:279 in _forward_impl, code: x = torch.flatten(x, 1)
        reshape_default: "f32[64, 512, 1, 1]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:278 in _forward_impl, code: x = self.avgpool(x)
        expand_default: "f32[64, 512, 1, 1]" = torch.ops.aten.expand.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        div_scalar: "f32[64, 512, 1, 1]" = torch.ops.aten.div.Scalar(expand_default, 1);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[64, 512, 1, 1]" = torch.ops.aten.where.self(le, full_default, div_scalar);  le = full_default = div_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        mul_tensor: "f32[64, 512, 1, 1]" = torch.ops.aten.mul.Tensor(where_self, convolution_19)
        reshape_default_1: "f32[64, 512, 1]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_2);  mul_tensor = _shape_param_2 = None
        sum_dim_int_list: "f32[64, 512]" = torch.ops.aten.sum.dim_IntList(reshape_default_1, [2]);  reshape_default_1 = None
        reshape_default_2: "f32[64, 512, 1]" = torch.ops.aten.reshape.default(where_self, _shape_param_3);  _shape_param_3 = None
        sum_dim_int_list_1: "f32[64, 512]" = torch.ops.aten.sum.dim_IntList(reshape_default_2, [2]);  reshape_default_2 = None
        unsqueeze_default: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(primals_60, 0)
        mul_tensor_1: "f32[64, 512]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, unsqueeze_default);  sum_dim_int_list = None
        reshape_default_3: "f32[64, 32, 16]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_4);  mul_tensor_1 = _shape_param_4 = None
        sum_dim_int_list_2: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(reshape_default_3, [2]);  reshape_default_3 = None
        mul_tensor_2: "f32[64, 512]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, unsqueeze_default);  sum_dim_int_list_1 = unsqueeze_default = None
        reshape_default_4: "f32[64, 32, 16]" = torch.ops.aten.reshape.default(mul_tensor_2, _shape_param_5);  mul_tensor_2 = _shape_param_5 = None
        sum_dim_int_list_3: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(reshape_default_4, [2]);  reshape_default_4 = None
        unsqueeze_default_1: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_39, -1)
        reshape_default_5: "f32[1, 32, 16]" = torch.ops.aten.reshape.default(primals_60, _shape_param_6);  primals_60 = _shape_param_6 = None
        mul_tensor_3: "f32[64, 32, 16]" = torch.ops.aten.mul.Tensor(unsqueeze_default_1, reshape_default_5);  unsqueeze_default_1 = reshape_default_5 = None
        mul_tensor_4: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, squeeze_38)
        sub_tensor: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_tensor_4, sum_dim_int_list_2);  mul_tensor_4 = sum_dim_int_list_2 = None
        mul_tensor_5: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sub_tensor, squeeze_39);  sub_tensor = None
        mul_tensor_6: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_tensor_5, squeeze_39);  mul_tensor_5 = None
        mul_tensor_7: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_tensor_6, squeeze_39);  mul_tensor_6 = None
        mul_tensor_8: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.0625);  mul_tensor_7 = None
        neg_default: "f32[64, 32]" = torch.ops.aten.neg.default(mul_tensor_8)
        mul_tensor_9: "f32[64, 32]" = torch.ops.aten.mul.Tensor(neg_default, squeeze_38);  neg_default = squeeze_38 = None
        mul_tensor_10: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, squeeze_39);  sum_dim_int_list_3 = squeeze_39 = None
        mul_tensor_11: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_tensor_10, 0.0625);  mul_tensor_10 = None
        sub_tensor_1: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_tensor_9, mul_tensor_11);  mul_tensor_9 = mul_tensor_11 = None
        unsqueeze_default_2: "f32[64, 32, 16, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, -1);  mul_tensor_3 = None
        unsqueeze_default_3: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_8, -1);  mul_tensor_8 = None
        unsqueeze_default_4: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, -1);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_tensor_1, -1);  sub_tensor_1 = None
        unsqueeze_default_6: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, -1);  unsqueeze_default_5 = None
        reshape_default_6: "f32[64, 32, 16, 1]" = torch.ops.aten.reshape.default(where_self, _shape_param_7);  where_self = _shape_param_7 = None
        mul_tensor_12: "f32[64, 32, 16, 1]" = torch.ops.aten.mul.Tensor(reshape_default_6, unsqueeze_default_2);  reshape_default_6 = unsqueeze_default_2 = None
        reshape_default_7: "f32[64, 32, 16, 1]" = torch.ops.aten.reshape.default(convolution_19, _shape_param_8);  convolution_19 = _shape_param_8 = None
        mul_tensor_13: "f32[64, 32, 16, 1]" = torch.ops.aten.mul.Tensor(reshape_default_7, unsqueeze_default_4);  reshape_default_7 = unsqueeze_default_4 = None
        add_tensor: "f32[64, 32, 16, 1]" = torch.ops.aten.add.Tensor(mul_tensor_12, mul_tensor_13);  mul_tensor_12 = mul_tensor_13 = None
        add_tensor_1: "f32[64, 32, 16, 1]" = torch.ops.aten.add.Tensor(add_tensor, unsqueeze_default_6);  add_tensor = unsqueeze_default_6 = None
        reshape_default_8: "f32[64, 512, 1, 1]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_9);  add_tensor_1 = _shape_param_9 = None
        return reshape_default_8



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
