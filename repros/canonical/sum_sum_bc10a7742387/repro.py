"""
Standalone repro captured via capture_hook.
Label: torchbench_timm_vovnet_train_001
Pattern hash: bc10a7742387
Shape hash: 14e8c18d
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
_shapes_config = "(T([32, 1024], f32), T([32, 1024, 7, 7], f32), T([1, 1024, 1, 1], f32), T([1, 1024, 1, 1], f32), T([1024], f32), T([1024], f32), S([32, 1024, 1, 1]), S([32, 1024, 7, 7]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[32, 1024]", arg216_1: "f32[32, 1024, 7, 7]", arg217_1: "f32[1, 1024, 1, 1]", arg218_1: "f32[1, 1024, 1, 1]", arg87_1: "f32[1024]", arg88_1: "f32[1024]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[32, 1024, 1, 1]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        expand_default: "f32[32, 1024, 7, 7]" = torch.ops.aten.expand.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        div_scalar: "f32[32, 1024, 7, 7]" = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None
        sub_tensor: "f32[32, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(arg216_1, arg217_1)
        mul_tensor: "f32[32, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, arg218_1);  sub_tensor = None
        unsqueeze_default: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg87_1, -1)
        unsqueeze_default_1: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[32, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg88_1, -1);  arg88_1 = None
        unsqueeze_default_3: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[32, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        relu_default: "f32[32, 1024, 7, 7]" = torch.ops.aten.relu.default(add_tensor);  add_tensor = None
        le_scalar: "b8[32, 1024, 7, 7]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[32, 1024, 7, 7]" = torch.ops.aten.where.self(le_scalar, full_default, div_scalar);  le_scalar = full_default = div_scalar = None
        squeeze_dims: "f32[1024]" = torch.ops.aten.squeeze.dims(arg217_1, [0, 2, 3]);  arg217_1 = None
        unsqueeze_default_4: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_1: "f32[32, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(arg216_1, unsqueeze_default_6);  arg216_1 = unsqueeze_default_6 = None
        mul_tensor_2: "f32[32, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_1)
        sum_dim_int_list_1: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3]);  mul_tensor_2 = None
        mul_tensor_3: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.0006377551020408163);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 0);  mul_tensor_3 = None
        unsqueeze_default_8: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_4: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.0006377551020408163)
        squeeze_dims_1: "f32[1024]" = torch.ops.aten.squeeze.dims(arg218_1, [0, 2, 3]);  arg218_1 = None
        mul_tensor_5: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_6: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default_10: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_11: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_7: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, arg87_1);  arg87_1 = None
        unsqueeze_default_13: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_14: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_8: "f32[32, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[32, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_8);  where_self = mul_tensor_8 = None
        sub_tensor_3: "f32[32, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_9: "f32[32, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        mul_tensor_10: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, squeeze_dims_1);  sum_dim_int_list_1 = squeeze_dims_1 = None
        return (mul_tensor_9, mul_tensor_10)



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
