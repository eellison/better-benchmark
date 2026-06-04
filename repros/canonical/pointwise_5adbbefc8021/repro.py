"""
Standalone repro captured via capture_hook.
Label: hf_DistillGPT2_train_000
Pattern hash: 5adbbefc8021
Shape hash: 59beeb25
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([16384, 3072], f32), S([32, 512, 3072]), S([-1, 3072]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_22: "f32[16384, 3072]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[32, 512, 3072]" = torch.ops.aten.view.default(addmm_22, _shape_param_0);  addmm_22 = _shape_param_0 = None
        mul_tensor: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_default, 0.5)
        pow_tensor_scalar: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_default, 3.0)
        mul_tensor_1: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_default, mul_tensor_1);  view_default = mul_tensor_1 = None
        mul_tensor_2: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_tensor_2);  mul_tensor_2 = None
        add_tensor_1: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_default, 1.0);  tanh_default = None
        mul_tensor_3: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor_1);  mul_tensor = add_tensor_1 = None
        view_default_1: "f32[16384, 3072]" = torch.ops.aten.view.default(mul_tensor_3, _shape_param_1);  mul_tensor_3 = _shape_param_1 = None
        permute_default: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_default_1, [1, 0]);  view_default_1 = None
        return permute_default

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
