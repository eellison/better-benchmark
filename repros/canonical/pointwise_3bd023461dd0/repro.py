"""
Standalone repro captured via capture_hook.
Label: tritonbench_dropout_8192x8192
Pattern hash: 3bd023466743
Shape hash: 1dd01484
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f16[8192, 8192]"):
        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[8192, 8192]" = torch.ops.prims.inductor_random.default([8192, 8192], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        convert_element_type_default: "f16[8192, 8192]" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.float16);  inductor_random_default = None
        gt_scalar: "b8[8192, 8192]" = torch.ops.aten.gt.Scalar(convert_element_type_default, 0.1);  convert_element_type_default = None
        mul_tensor: "f16[8192, 8192]" = torch.ops.aten.mul.Tensor(gt_scalar, arg0_1);  gt_scalar = arg0_1 = None
        mul_tensor_1: "f16[8192, 8192]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        return mul_tensor_1


def _default_make_inputs():
    return [
    torch.randn([8192, 8192], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
