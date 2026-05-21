"""
Standalone repro captured via capture_hook.
Label: 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_train_003
Pattern hash: 331daacd1ee0
Shape hash: 41a00152
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
_shapes_config = "(T([2048, 2560], f32), T([2], i64), T([16, 128, 2560], f32), S([16, 128, 2560]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_5: "f32[2048, 2560]", inductor_seeds: "i64[2]", add_3: "f32[16, 128, 2560]", _shape_param_0):
        # No stacktrace found for following nodes
        view_default: "f32[16, 128, 2560]" = torch.ops.aten.view.default(addmm_5, _shape_param_0);  addmm_5 = _shape_param_0 = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 1);  inductor_seeds = None
        inductor_random_default: "f32[16, 128, 2560]" = torch.ops.prims.inductor_random.default([16, 128, 2560], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[16, 128, 2560]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(gt_scalar, view_default);  gt_scalar = view_default = None
        mul_tensor_1: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        add_tensor: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_3, mul_tensor_1);  add_3 = mul_tensor_1 = None
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
