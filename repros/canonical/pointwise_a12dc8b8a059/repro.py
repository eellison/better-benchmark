"""
Standalone repro captured via capture_hook.
Label: torchbench_nvidia_deeprecommender_train_000
Pattern hash: a12dc8b8a059
Shape hash: 22c6460b
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
_shapes_config = "(T([1024, 1024], f32))"

class Repro(torch.nn.Module):
    def forward(self, addmm_2: "f32[1024, 1024]"):
        # No stacktrace found for following nodes
        gt_scalar: "b8[1024, 1024]" = torch.ops.aten.gt.Scalar(addmm_2, 0)
        mul_tensor: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(addmm_2, 1.0507009873554805)
        mul_tensor_1: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(addmm_2, 1.0);  addmm_2 = None
        expm1_default: "f32[1024, 1024]" = torch.ops.aten.expm1.default(mul_tensor_1);  mul_tensor_1 = None
        mul_tensor_2: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(expm1_default, 1.7580993408473766);  expm1_default = None
        where_self: "f32[1024, 1024]" = torch.ops.aten.where.self(gt_scalar, mul_tensor, mul_tensor_2);  gt_scalar = mul_tensor = mul_tensor_2 = None
        inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[1024, 1024]" = torch.ops.prims.inductor_random.default([1024, 1024], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar_1: "b8[1024, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.8);  inductor_random_default = None
        mul_tensor_3: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(gt_scalar_1, where_self);  gt_scalar_1 = where_self = None
        mul_tensor_4: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 5.000000000000001);  mul_tensor_3 = None
        return mul_tensor_4



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
