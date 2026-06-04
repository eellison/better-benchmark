"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_T5_train_000
Pattern hash: 9e64ab4c6018
Shape hash: 080cbb7f
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8192, 2048], f32), T([64], i64), S([8, 1024, 2048]), S([8192, 2048]))"

class Repro(torch.nn.Module):
    def forward(self, mm_94: "f32[8192, 2048]", inductor_seeds: "i64[64]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[8, 1024, 2048]" = torch.ops.aten.view.default(mm_94, _shape_param_0);  mm_94 = _shape_param_0 = None
        relu_default: "f32[8, 1024, 2048]" = torch.ops.aten.relu.default(view_default);  view_default = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 61);  inductor_seeds = None
        inductor_random_default: "f32[8, 1024, 2048]" = torch.ops.prims.inductor_random.default([8, 1024, 2048], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8, 1024, 2048]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt_scalar, relu_default);  gt_scalar = None
        mul_tensor_1: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        view_default_1: "f32[8192, 2048]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_1);  mul_tensor_1 = _shape_param_1 = None
        le_scalar: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        return (view_default_1, le_scalar)

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
