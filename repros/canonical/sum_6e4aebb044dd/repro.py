"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_train_001
Pattern hash: 6e4aebb044dd
Shape hash: 369f714a
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 2304], f32), T([128, 2304, 7, 7], f32), S([128, 2304, 1, 1]), S([128, 2304, 7, 7]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[128, 2304]", arg408_1: "f32[128, 2304, 7, 7]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[128, 2304, 1, 1]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        squeeze_dim: "f32[128, 2304, 1]" = torch.ops.aten.squeeze.dim(view_default, 3);  view_default = None
        squeeze_dim_1: "f32[128, 2304]" = torch.ops.aten.squeeze.dim(squeeze_dim, 2);  squeeze_dim = None
        full_default: "f32[294912]" = torch.ops.aten.full.default([294912], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        as_strided_scatter_default: "f32[294912]" = torch.ops.aten.as_strided_scatter.default(full_default, squeeze_dim_1, [128, 2304], [2304, 1], 0);  full_default = squeeze_dim_1 = None
        as_strided_default: "f32[128, 2304, 1, 1]" = torch.ops.aten.as_strided.default(as_strided_scatter_default, [128, 2304, 1, 1], [2304, 1, 1, 1], 0);  as_strided_scatter_default = None
        expand_default: "f32[128, 2304, 7, 7]" = torch.ops.aten.expand.default(as_strided_default, _shape_param_1);  as_strided_default = _shape_param_1 = None
        div_scalar: "f32[128, 2304, 7, 7]" = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None
        neg_default: "f32[128, 2304, 7, 7]" = torch.ops.aten.neg.default(arg408_1)
        exp_default: "f32[128, 2304, 7, 7]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[128, 2304, 7, 7]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        reciprocal_default: "f32[128, 2304, 7, 7]" = torch.ops.aten.reciprocal.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[128, 2304, 7, 7]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_1: "f32[128, 2304, 7, 7]" = torch.ops.aten.mul.Tensor(div_scalar, mul_tensor);  div_scalar = None
        sub_tensor: "f32[128, 2304, 7, 7]" = torch.ops.aten.sub.Tensor(1, mul_tensor);  mul_tensor = None
        mul_tensor_2: "f32[128, 2304, 7, 7]" = torch.ops.aten.mul.Tensor(arg408_1, sub_tensor);  arg408_1 = sub_tensor = None
        add_tensor_1: "f32[128, 2304, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, 1);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 2304, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, add_tensor_1);  mul_tensor_1 = add_tensor_1 = None
        sum_dim_int_list: "f32[2304]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 2, 3]);  mul_tensor_3 = None
        return sum_dim_int_list

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
