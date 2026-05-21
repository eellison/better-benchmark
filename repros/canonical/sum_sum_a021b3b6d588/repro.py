"""
Standalone repro captured via capture_hook.
Label: torchbench_pytorch_unet_train
Pattern hash: a021b3b6d588
Shape hash: 90a2f096
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
    def forward(self, getitem_50: "f32[8, 128, 640, 959]", getitem_92: "f32[8, 64, 320, 479]", getitem_5: "i8[8, 64, 320, 479]", convolution_1: "f32[8, 64, 640, 959]", getitem_3: "f32[1, 64, 1, 1]", rsqrt_1: "f32[1, 64, 1, 1]", primals_14: "f32[64]", primals_15: "f32[64]", full_default: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:67 in forward, code: x = torch.cat([x2, x1], dim=1)
        slice_tensor: "f32[8, 64, 640, 959]" = torch.ops.aten.slice.Tensor(getitem_50, 1, 0, 64);  getitem_50 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:38 in forward, code: return self.maxpool_conv(x)
        full_default_1: "f32[512, 613760]" = torch.ops.aten.full.default([512, 613760], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        reshape_default: "f32[512, 153280]" = torch.ops.aten.reshape.default(getitem_92, _shape_param_0);  getitem_92 = _shape_param_0 = None
        _low_memory_max_pool_offsets_to_indices_default: "i64[8, 64, 320, 479]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_5, [2, 2], [640, 959], [2, 2], [0, 0], [1, 1]);  getitem_5 = None
        reshape_default_1: "i64[512, 153280]" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices_default, _shape_param_1);  _low_memory_max_pool_offsets_to_indices_default = _shape_param_1 = None
        scatter_add_default: "f32[512, 613760]" = torch.ops.aten.scatter_add.default(full_default_1, 1, reshape_default_1, reshape_default);  full_default_1 = reshape_default_1 = reshape_default = None
        reshape_default_2: "f32[8, 64, 640, 959]" = torch.ops.aten.reshape.default(scatter_add_default, _shape_param_2);  scatter_add_default = _shape_param_2 = None
        add_tensor: "f32[8, 64, 640, 959]" = torch.ops.aten.add.Tensor(slice_tensor, reshape_default_2);  slice_tensor = reshape_default_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/unet_parts.py:25 in forward, code: return self.double_conv(x)
        sub_tensor: "f32[8, 64, 640, 959]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
        mul_tensor: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_1);  sub_tensor = None
        unsqueeze_default: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1)
        unsqueeze_default_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1);  primals_15 = None
        unsqueeze_default_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[8, 64, 640, 959]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        relu_default: "f32[8, 64, 640, 959]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        le_scalar: "b8[8, 64, 640, 959]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self: "f32[8, 64, 640, 959]" = torch.ops.aten.where.self(le_scalar, full_default, add_tensor);  le_scalar = full_default = add_tensor = None
        squeeze_dims: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        unsqueeze_default_4: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_1: "f32[8, 64, 640, 959]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_default_6);  convolution_1 = unsqueeze_default_6 = None
        mul_tensor_2: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_1)
        sum_dim_int_list_1: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3]);  mul_tensor_2 = None
        mul_tensor_3: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 2.0366266944734097e-07);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 0);  mul_tensor_3 = None
        unsqueeze_default_8: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_4: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 2.0366266944734097e-07);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_tensor_5: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_6: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default_10: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_11: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_7: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_14);  squeeze_dims_1 = primals_14 = None
        unsqueeze_default_13: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_14: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_8: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[8, 64, 640, 959]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_8);  where_self = mul_tensor_8 = None
        sub_tensor_3: "f32[8, 64, 640, 959]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_9: "f32[8, 64, 640, 959]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        return mul_tensor_9


def _default_make_inputs():
    return [
    torch.randn([8, 128, 640, 959], dtype=torch.float32, device='cuda'),
    torch.randn([8, 64, 320, 479], dtype=torch.float32, device='cuda'),
    torch.randint(0, 4, (78512064,), dtype=torch.int8, device='cuda').as_strided([8, 64, 320, 479], [9814016, 153344, 479, 1]),  # getitem_5
    torch.randn([8, 64, 640, 959], dtype=torch.float32, device='cuda'),
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    [512, 153280],  # _shape_param_0
    [512, 153280],  # _shape_param_1
    [8, 64, 640, 959],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
