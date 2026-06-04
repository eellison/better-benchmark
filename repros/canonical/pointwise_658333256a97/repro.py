"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_train_003
Pattern hash: 658333256a97
Shape hash: 46afd36c
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([2], i64), T([32768, 256], f32), T([8, 4096, 256], f32), S([8, 4096, 256]))"

class Repro(torch.nn.Module):
    def forward(self, inductor_seeds: "i64[2]", mm_3: "f32[32768, 256]", arg7_1: "f32[8, 4096, 256]", _shape_param_0):
        # No stacktrace found for following nodes
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 1);  inductor_seeds = None
        inductor_random_default: "f32[8, 4096, 256]" = torch.ops.prims.inductor_random.default([8, 4096, 256], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8, 4096, 256]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.05);  inductor_random_default = None
        view_default: "f32[8, 4096, 256]" = torch.ops.aten.view.default(mm_3, _shape_param_0);  mm_3 = _shape_param_0 = None
        mul_tensor: "f32[8, 4096, 256]" = torch.ops.aten.mul.Tensor(gt_scalar, view_default);  gt_scalar = view_default = None
        mul_tensor_1: "f32[8, 4096, 256]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.0526315789473684);  mul_tensor = None
        add_tensor: "f32[8, 4096, 256]" = torch.ops.aten.add.Tensor(arg7_1, mul_tensor_1);  arg7_1 = mul_tensor_1 = None
        return add_tensor

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
