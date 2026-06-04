"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_infer_000
Pattern hash: 2cc7c346a6db
Shape hash: d898d112
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([4096, 1024], f32), T([4096, 1024], f32), S([32, 128, 1024]), S([32, 128, 1024]), S([4096, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, mm_141: "f32[4096, 1024]", mm_142: "f32[4096, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[32, 128, 1024]" = torch.ops.aten.view.default(mm_141, _shape_param_0);  mm_141 = _shape_param_0 = None
        mul_tensor: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_default, 0.5)
        pow_tensor_scalar: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_default, 3.0)
        mul_tensor_1: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_default, mul_tensor_1);  view_default = mul_tensor_1 = None
        mul_tensor_2: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_tensor_2);  mul_tensor_2 = None
        add_tensor_1: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_default, 1.0);  tanh_default = None
        mul_tensor_3: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor_1);  mul_tensor = add_tensor_1 = None
        view_default_1: "f32[32, 128, 1024]" = torch.ops.aten.view.default(mm_142, _shape_param_1);  mm_142 = _shape_param_1 = None
        mul_tensor_4: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_3, view_default_1);  mul_tensor_3 = view_default_1 = None
        view_default_2: "f32[4096, 1024]" = torch.ops.aten.view.default(mul_tensor_4, _shape_param_2);  mul_tensor_4 = _shape_param_2 = None
        return view_default_2

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
