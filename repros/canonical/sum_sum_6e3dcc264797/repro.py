"""
Standalone repro captured via capture_hook.
Label: torchbench_mobilenet_v2_train_001
Pattern hash: 6e3dcc264797
Shape hash: 24dc51cb
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
_shapes_config = "(T([128, 1280], b8), T([128, 1280], f32), T([128, 1280, 7, 7], f32), T([1, 1280, 1, 1], f32), T([1, 1280, 1, 1], f32), T([1280], f32), T([1280], f32), S([128, 1280, 1, 1]), S([128, 1280, 7, 7]))"

class Repro(torch.nn.Module):
    def forward(self, arg331_1: "b8[128, 1280]", mm: "f32[128, 1280]", arg328_1: "f32[128, 1280, 7, 7]", arg329_1: "f32[1, 1280, 1, 1]", arg330_1: "f32[1, 1280, 1, 1]", arg138_1: "f32[1280]", arg139_1: "f32[1280]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[128, 1280]" = torch.ops.prims.convert_element_type.default(arg331_1, torch.float32);  arg331_1 = None
        mul_tensor: "f32[128, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.25);  convert_element_type_default = None
        mul_tensor_1: "f32[128, 1280]" = torch.ops.aten.mul.Tensor(mm, mul_tensor);  mm = mul_tensor = None
        view_default: "f32[128, 1280, 1, 1]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_0);  mul_tensor_1 = _shape_param_0 = None
        expand_default: "f32[128, 1280, 7, 7]" = torch.ops.aten.expand.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        div_scalar: "f32[128, 1280, 7, 7]" = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None
        sub_tensor: "f32[128, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(arg328_1, arg329_1)
        mul_tensor_2: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, arg330_1);  sub_tensor = None
        unsqueeze_default: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg138_1, -1)
        unsqueeze_default_1: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_3: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_2, unsqueeze_default_1);  mul_tensor_2 = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_default_3: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[128, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_3, unsqueeze_default_3);  mul_tensor_3 = unsqueeze_default_3 = None
        le_scalar: "b8[128, 1280, 7, 7]" = torch.ops.aten.le.Scalar(add_tensor, 0.0)
        ge_scalar: "b8[128, 1280, 7, 7]" = torch.ops.aten.ge.Scalar(add_tensor, 6.0);  add_tensor = None
        bitwise_or_tensor: "b8[128, 1280, 7, 7]" = torch.ops.aten.bitwise_or.Tensor(le_scalar, ge_scalar);  le_scalar = ge_scalar = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[128, 1280, 7, 7]" = torch.ops.aten.where.self(bitwise_or_tensor, full_default, div_scalar);  bitwise_or_tensor = full_default = div_scalar = None
        squeeze_dims: "f32[1280]" = torch.ops.aten.squeeze.dims(arg329_1, [0, 2, 3]);  arg329_1 = None
        unsqueeze_default_4: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[1280]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_1: "f32[128, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(arg328_1, unsqueeze_default_6);  arg328_1 = unsqueeze_default_6 = None
        mul_tensor_4: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_1)
        sum_dim_int_list_1: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 2, 3]);  mul_tensor_4 = None
        mul_tensor_5: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.00015943877551020407);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_8: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_6: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.00015943877551020407)
        squeeze_dims_1: "f32[1280]" = torch.ops.aten.squeeze.dims(arg330_1, [0, 2, 3]);  arg330_1 = None
        mul_tensor_7: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_8: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_tensor_6, mul_tensor_7);  mul_tensor_6 = mul_tensor_7 = None
        unsqueeze_default_10: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_tensor_8, 0);  mul_tensor_8 = None
        unsqueeze_default_11: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_9: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, arg138_1);  arg138_1 = None
        unsqueeze_default_13: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, 0);  mul_tensor_9 = None
        unsqueeze_default_14: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_10: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[128, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_10);  where_self = mul_tensor_10 = None
        sub_tensor_3: "f32[128, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_11: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        mul_tensor_12: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, squeeze_dims_1);  sum_dim_int_list_1 = squeeze_dims_1 = None
        return (mul_tensor_11, mul_tensor_12)



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
