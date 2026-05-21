"""
Standalone repro captured via capture_hook.
Label: torchbench_functorch_dp_cifar10_train
Pattern hash: 9e0dfaba8d1c
Shape hash: b38545a6
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
_shapes_config = "(T([64, 64, 8, 8], f32, stride=(4096, 1, 512, 64)), T([64, 64, 8, 8], f32, stride=(4096, 1, 512, 64)), T([64, 64, 8, 8], i8, stride=(4096, 1, 512, 64), gen=Index(9)), T([64, 64, 16, 16], f32, stride=(16384, 1, 1024, 64)), T([64, 32, 1, 1], f32), T([64, 32, 1, 1], f32), T([64], f32), T([64], f32), T([], f32), S([4096, 64]), S([4096, 64]), S([64, 64, 16, 16]), S([64, 32, 2, 256]), S([64, 64, 16, 16]), S([64, 64, 256]), S([64, 64, 256]), S([64, 32, 2]), S([64, 32, 2]), S([1, 32, 2]), S([64, 32, 2, 256]), S([64, 64, 16, 16]))"

class Repro(torch.nn.Module):
    def forward(self, where_14: "f32[64, 64, 8, 8]", getitem_96: "f32[64, 64, 8, 8]", getitem_3: "i8[64, 64, 8, 8]", convolution: "f32[64, 64, 16, 16]", getitem_1: "f32[64, 32, 1, 1]", rsqrt: "f32[64, 32, 1, 1]", primals_3: "f32[64]", primals_4: "f32[64]", full_default: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        add_tensor: "f32[64, 64, 8, 8]" = torch.ops.aten.add.Tensor(where_14, getitem_96);  where_14 = getitem_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:271 in _forward_impl, code: x = self.maxpool(x)
        full_default_1: "f32[4096, 256]" = torch.ops.aten.full.default([4096, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        reshape_default: "f32[4096, 64]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_0);  add_tensor = _shape_param_0 = None
        _low_memory_max_pool_offsets_to_indices_default: "i64[64, 64, 8, 8]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_3, [3, 3], [16, 16], [2, 2], [1, 1], [1, 1]);  getitem_3 = None
        reshape_default_1: "i64[4096, 64]" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices_default, _shape_param_1);  _low_memory_max_pool_offsets_to_indices_default = _shape_param_1 = None
        scatter_add_default: "f32[4096, 256]" = torch.ops.aten.scatter_add.default(full_default_1, 1, reshape_default_1, reshape_default);  full_default_1 = reshape_default_1 = reshape_default = None
        reshape_default_2: "f32[64, 64, 16, 16]" = torch.ops.aten.reshape.default(scatter_add_default, _shape_param_2);  scatter_add_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:269 in _forward_impl, code: x = self.bn1(x)
        reshape_default_3: "f32[64, 32, 2, 256]" = torch.ops.aten.reshape.default(convolution, _shape_param_3);  _shape_param_3 = None
        sub_tensor: "f32[64, 32, 2, 256]" = torch.ops.aten.sub.Tensor(reshape_default_3, getitem_1)
        mul_tensor: "f32[64, 32, 2, 256]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        reshape_default_4: "f32[64, 64, 16, 16]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_4);  mul_tensor = _shape_param_4 = None
        unsqueeze_default: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_3, 0)
        unsqueeze_default_1: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2)
        unsqueeze_default_2: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_1: "f32[64, 64, 16, 16]" = torch.ops.aten.mul.Tensor(reshape_default_4, unsqueeze_default_2);  reshape_default_4 = unsqueeze_default_2 = None
        unsqueeze_default_3: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_4, 0);  primals_4 = None
        unsqueeze_default_4: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        add_tensor_1: "f32[64, 64, 16, 16]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:270 in _forward_impl, code: x = self.relu(x)
        relu_default: "f32[64, 64, 16, 16]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        le_scalar: "b8[64, 64, 16, 16]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self: "f32[64, 64, 16, 16]" = torch.ops.aten.where.self(le_scalar, full_default, reshape_default_2);  le_scalar = full_default = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:269 in _forward_impl, code: x = self.bn1(x)
        mul_tensor_2: "f32[64, 64, 16, 16]" = torch.ops.aten.mul.Tensor(where_self, convolution);  convolution = None
        reshape_default_5: "f32[64, 64, 256]" = torch.ops.aten.reshape.default(mul_tensor_2, _shape_param_5);  mul_tensor_2 = _shape_param_5 = None
        sum_dim_int_list: "f32[64, 64]" = torch.ops.aten.sum.dim_IntList(reshape_default_5, [2]);  reshape_default_5 = None
        reshape_default_6: "f32[64, 64, 256]" = torch.ops.aten.reshape.default(where_self, _shape_param_6);  _shape_param_6 = None
        sum_dim_int_list_1: "f32[64, 64]" = torch.ops.aten.sum.dim_IntList(reshape_default_6, [2]);  reshape_default_6 = None
        mul_tensor_3: "f32[64, 64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, unsqueeze_default);  sum_dim_int_list = None
        reshape_default_7: "f32[64, 32, 2]" = torch.ops.aten.reshape.default(mul_tensor_3, _shape_param_7);  mul_tensor_3 = _shape_param_7 = None
        sum_dim_int_list_2: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(reshape_default_7, [2]);  reshape_default_7 = None
        mul_tensor_4: "f32[64, 64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, unsqueeze_default);  sum_dim_int_list_1 = unsqueeze_default = None
        reshape_default_8: "f32[64, 32, 2]" = torch.ops.aten.reshape.default(mul_tensor_4, _shape_param_8);  mul_tensor_4 = _shape_param_8 = None
        sum_dim_int_list_3: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(reshape_default_8, [2]);  reshape_default_8 = None
        squeeze_dims: "f32[64, 32]" = torch.ops.aten.squeeze.dims(rsqrt, [2, 3]);  rsqrt = None
        unsqueeze_default_6: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(squeeze_dims, -1)
        reshape_default_9: "f32[1, 32, 2]" = torch.ops.aten.reshape.default(primals_3, _shape_param_9);  primals_3 = _shape_param_9 = None
        mul_tensor_5: "f32[64, 32, 2]" = torch.ops.aten.mul.Tensor(unsqueeze_default_6, reshape_default_9);  unsqueeze_default_6 = reshape_default_9 = None
        squeeze_dims_1: "f32[64, 32]" = torch.ops.aten.squeeze.dims(getitem_1, [2, 3]);  getitem_1 = None
        mul_tensor_6: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, squeeze_dims_1)
        sub_tensor_1: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_tensor_6, sum_dim_int_list_2);  mul_tensor_6 = sum_dim_int_list_2 = None
        mul_tensor_7: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sub_tensor_1, squeeze_dims);  sub_tensor_1 = None
        mul_tensor_8: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_tensor_7, squeeze_dims);  mul_tensor_7 = None
        mul_tensor_9: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_tensor_8, squeeze_dims);  mul_tensor_8 = None
        mul_tensor_10: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_tensor_9, 0.001953125);  mul_tensor_9 = None
        neg_default: "f32[64, 32]" = torch.ops.aten.neg.default(mul_tensor_10)
        mul_tensor_11: "f32[64, 32]" = torch.ops.aten.mul.Tensor(neg_default, squeeze_dims_1);  neg_default = squeeze_dims_1 = None
        mul_tensor_12: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, squeeze_dims);  sum_dim_int_list_3 = squeeze_dims = None
        mul_tensor_13: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_tensor_12, 0.001953125);  mul_tensor_12 = None
        sub_tensor_2: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_tensor_11, mul_tensor_13);  mul_tensor_11 = mul_tensor_13 = None
        unsqueeze_default_7: "f32[64, 32, 2, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, -1);  mul_tensor_5 = None
        unsqueeze_default_8: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_10, -1);  mul_tensor_10 = None
        unsqueeze_default_9: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1);  unsqueeze_default_8 = None
        unsqueeze_default_10: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_tensor_2, -1);  sub_tensor_2 = None
        unsqueeze_default_11: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, -1);  unsqueeze_default_10 = None
        reshape_default_10: "f32[64, 32, 2, 256]" = torch.ops.aten.reshape.default(where_self, _shape_param_10);  where_self = _shape_param_10 = None
        mul_tensor_14: "f32[64, 32, 2, 256]" = torch.ops.aten.mul.Tensor(reshape_default_10, unsqueeze_default_7);  reshape_default_10 = unsqueeze_default_7 = None
        mul_tensor_15: "f32[64, 32, 2, 256]" = torch.ops.aten.mul.Tensor(reshape_default_3, unsqueeze_default_9);  reshape_default_3 = unsqueeze_default_9 = None
        add_tensor_2: "f32[64, 32, 2, 256]" = torch.ops.aten.add.Tensor(mul_tensor_14, mul_tensor_15);  mul_tensor_14 = mul_tensor_15 = None
        add_tensor_3: "f32[64, 32, 2, 256]" = torch.ops.aten.add.Tensor(add_tensor_2, unsqueeze_default_11);  add_tensor_2 = unsqueeze_default_11 = None
        reshape_default_11: "f32[64, 64, 16, 16]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_11);  add_tensor_3 = _shape_param_11 = None
        return reshape_default_11



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
