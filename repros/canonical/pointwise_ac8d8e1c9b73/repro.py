"""
Standalone repro captured via capture_hook.
Label: torchbench_nvidia_deeprecommender_train
Pattern hash: ac8d8e1c9b73
Shape hash: 0f3e2fa1
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[256, 1024]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "f32[256, 1024]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        gt: "b8[256, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type, 0)
        mul: "f32[256, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.0507009873554805)
        mul_1: "f32[256, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.0);  convert_element_type = None
        expm1: "f32[256, 1024]" = torch.ops.aten.expm1.default(mul_1);  mul_1 = None
        mul_2: "f32[256, 1024]" = torch.ops.aten.mul.Tensor(expm1, 1.7580993408473766);  expm1 = None
        where: "f32[256, 1024]" = torch.ops.aten.where.self(gt, mul, mul_2);  gt = mul = mul_2 = None
        convert_element_type_1: "bf16[256, 1024]" = torch.ops.prims.convert_element_type.default(where, torch.bfloat16);  where = None
        inductor_seeds: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0);  inductor_seeds = None
        inductor_random: "f32[256, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_0, inductor_lookup_seed, 'rand');  _shape_param_0 = inductor_lookup_seed = None
        convert_element_type_2: "bf16[256, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt_1: "b8[256, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_2, 0.8);  convert_element_type_2 = None
        mul_3: "bf16[256, 1024]" = torch.ops.aten.mul.Tensor(gt_1, convert_element_type_1);  convert_element_type_1 = None
        mul_4: "bf16[256, 1024]" = torch.ops.aten.mul.Tensor(mul_3, 5.000000000000001);  mul_3 = None
        return (gt_1, mul_4)



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
