"""
Standalone repro captured via capture_hook.
Label: torchbench_timm_nfnet_train_001
Pattern hash: 27f8a4b9ab09
Shape hash: 70c9d47c
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 3072], f32), T([128, 3072, 6, 6], f32), S([128, 3072, 1, 1]), S([128, 3072, 6, 6]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[128, 3072]", arg413_1: "f32[128, 3072, 6, 6]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[128, 3072, 1, 1]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        expand_default: "f32[128, 3072, 6, 6]" = torch.ops.aten.expand.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        div_scalar: "f32[128, 3072, 6, 6]" = torch.ops.aten.div.Scalar(expand_default, 36);  expand_default = None
        mul_tensor: "f32[128, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(div_scalar, 1.7015043497085571);  div_scalar = None
        mul_tensor_1: "f32[128, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(arg413_1, 0.7071067811865476)
        erf_default: "f32[128, 3072, 6, 6]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[128, 3072, 6, 6]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[128, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(add_tensor, 0.5);  add_tensor = None
        mul_tensor_3: "f32[128, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(arg413_1, arg413_1)
        mul_tensor_4: "f32[128, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(mul_tensor_3, -0.5);  mul_tensor_3 = None
        exp_default: "f32[128, 3072, 6, 6]" = torch.ops.aten.exp.default(mul_tensor_4);  mul_tensor_4 = None
        mul_tensor_5: "f32[128, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_6: "f32[128, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(arg413_1, mul_tensor_5);  arg413_1 = mul_tensor_5 = None
        add_tensor_1: "f32[128, 3072, 6, 6]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_6);  mul_tensor_2 = mul_tensor_6 = None
        mul_tensor_7: "f32[128, 3072, 6, 6]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor_1);  mul_tensor = add_tensor_1 = None
        sum_dim_int_list: "f32[3072]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 2, 3]);  mul_tensor_7 = None
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
