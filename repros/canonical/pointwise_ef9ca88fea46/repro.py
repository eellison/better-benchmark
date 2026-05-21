"""
Standalone repro captured via capture_hook.
Label: torchbench_vgg16_train_000
Pattern hash: ef9ca88fea46
Shape hash: 1d564620
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
_shapes_config = "(T([128, 4096], f32), T([2], i64))"

class Repro(torch.nn.Module):
    def forward(self, addmm_1: "f32[128, 4096]", inductor_seeds: "i64[2]"):
        # No stacktrace found for following nodes
        relu_default: "f32[128, 4096]" = torch.ops.aten.relu.default(addmm_1);  addmm_1 = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 1);  inductor_seeds = None
        inductor_random_default: "f32[128, 4096]" = torch.ops.prims.inductor_random.default([128, 4096], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[128, 4096]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.5);  inductor_random_default = None
        mul_tensor: "f32[128, 4096]" = torch.ops.aten.mul.Tensor(gt_scalar, relu_default);  gt_scalar = None
        mul_tensor_1: "f32[128, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor, 2.0);  mul_tensor = None
        le_scalar: "b8[128, 4096]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        return (mul_tensor_1, le_scalar)



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
