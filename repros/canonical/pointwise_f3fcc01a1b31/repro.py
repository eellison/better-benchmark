"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Bart_train_004
Pattern hash: f3fcc01a1b31
Shape hash: 2a61d2f2
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
_shapes_config = "(T([2048, 3072], f32), T([4], i64), S([4, 512, 3072]), S([2048, 3072]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_4: "f32[2048, 3072]", inductor_seeds: "i64[4]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[4, 512, 3072]" = torch.ops.aten.view.default(addmm_4, _shape_param_0);  addmm_4 = _shape_param_0 = None
        mul_tensor: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_default, 0.5)
        mul_tensor_1: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_default, 0.7071067811865476);  view_default = None
        erf_default: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 2);  inductor_seeds = None
        inductor_random_default: "f32[4, 512, 3072]" = torch.ops.prims.inductor_random.default([4, 512, 3072], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[4, 512, 3072]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor_3: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(gt_scalar, mul_tensor_2);  gt_scalar = mul_tensor_2 = None
        mul_tensor_4: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 1.1111111111111112);  mul_tensor_3 = None
        view_default_1: "f32[2048, 3072]" = torch.ops.aten.view.default(mul_tensor_4, _shape_param_1);  mul_tensor_4 = _shape_param_1 = None
        return view_default_1



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
