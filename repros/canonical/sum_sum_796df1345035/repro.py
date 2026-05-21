"""
Standalone repro captured via capture_hook.
Label: torchbench_shufflenet_v2_x1_0_train
Pattern hash: 796df1345035
Shape hash: 9f575245
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
_shapes_config = "(T([512, 1024], f32), T([512, 1024, 7, 7], f32, stride=(50176, 1, 7168, 1024)), T([1, 1024, 1, 1], f32), T([1, 1024, 1, 1], f32), T([1024], f32), T([1024], f32), S([512, 1024, 7, 7]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[512, 1024]", convolution_55: "f32[512, 1024, 7, 7]", getitem_139: "f32[1, 1024, 1, 1]", rsqrt_55: "f32[1, 1024, 1, 1]", primals_336: "f32[1024]", primals_337: "f32[1024]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:161 in _forward_impl, code: x = x.mean([2, 3])  # globalpool
        unsqueeze_default: "f32[512, 1024, 1]" = torch.ops.aten.unsqueeze.default(mm, 2);  mm = None
        unsqueeze_default_1: "f32[512, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 3);  unsqueeze_default = None
        expand_default: "f32[512, 1024, 7, 7]" = torch.ops.aten.expand.default(unsqueeze_default_1, _shape_param_0);  unsqueeze_default_1 = _shape_param_0 = None
        div_scalar: "f32[512, 1024, 7, 7]" = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:160 in _forward_impl, code: x = self.conv5(x)
        sub_tensor: "f32[512, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_55, getitem_139)
        mul_tensor: "f32[512, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_55);  sub_tensor = None
        unsqueeze_default_2: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_336, -1)
        unsqueeze_default_3: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[512, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_3);  mul_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_337, -1);  primals_337 = None
        unsqueeze_default_5: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        add_tensor: "f32[512, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        relu_default: "f32[512, 1024, 7, 7]" = torch.ops.aten.relu.default(add_tensor);  add_tensor = None
        le_scalar: "b8[512, 1024, 7, 7]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[512, 1024, 7, 7]" = torch.ops.aten.where.self(le_scalar, full_default, div_scalar);  le_scalar = full_default = div_scalar = None
        squeeze_dims: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_139, [0, 2, 3]);  getitem_139 = None
        unsqueeze_default_6: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_7: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        sum_dim_int_list: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_1: "f32[512, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_default_8);  convolution_55 = unsqueeze_default_8 = None
        mul_tensor_2: "f32[512, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_1)
        sum_dim_int_list_1: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3]);  mul_tensor_2 = None
        mul_tensor_3: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 3.985969387755102e-05);  sum_dim_int_list = None
        unsqueeze_default_9: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 0);  mul_tensor_3 = None
        unsqueeze_default_10: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None
        mul_tensor_4: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 3.985969387755102e-05);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
        mul_tensor_5: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_6: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default_12: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_13: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 2);  unsqueeze_default_12 = None
        unsqueeze_default_14: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 3);  unsqueeze_default_13 = None
        mul_tensor_7: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, primals_336);  squeeze_dims_1 = primals_336 = None
        unsqueeze_default_15: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_16: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 2);  unsqueeze_default_15 = None
        unsqueeze_default_17: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 3);  unsqueeze_default_16 = None
        mul_tensor_8: "f32[512, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_14);  sub_tensor_1 = unsqueeze_default_14 = None
        sub_tensor_2: "f32[512, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_8);  where_self = mul_tensor_8 = None
        sub_tensor_3: "f32[512, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_11);  sub_tensor_2 = unsqueeze_default_11 = None
        mul_tensor_9: "f32[512, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_17);  sub_tensor_3 = unsqueeze_default_17 = None
        return mul_tensor_9



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
