"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_T5_train_000
Pattern hash: d87780a73e70
Shape hash: c64bf2bd
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8192, 512], f32), T([64], i64), T([8, 1024, 512], f32), T([512], f32), S([8, 1024, 512]), S([8192, 512]), S([8192, 512]), S([8192, 512]), S([8192, 512]), S([8192, 512]), S([8192, 512]), S([8192, 512]), S([8192, 512]), S([8192, 512]), S([8192, 512]), S([8192, 512]), S([8192, 512]))"

class Repro(torch.nn.Module):
    def forward(self, mm_35: "f32[8192, 512]", inductor_seeds: "i64[64]", add_33: "f32[8, 1024, 512]", arg51_1: "f32[512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12):
        # No stacktrace found for following nodes
        view_default: "f32[8, 1024, 512]" = torch.ops.aten.view.default(mm_35, _shape_param_0);  mm_35 = _shape_param_0 = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 24)
        inductor_random_default: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_scalar, view_default);  gt_scalar = view_default = None
        mul_tensor_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        add_tensor: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_33, mul_tensor_1);  add_33 = mul_tensor_1 = None
        pow_tensor_scalar: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor, 2)
        mean_dim: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None
        add_tensor_1: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_2: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor, rsqrt_default);  add_tensor = rsqrt_default = None
        mul_tensor_3: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg51_1, mul_tensor_2);  arg51_1 = mul_tensor_2 = None
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 25);  inductor_seeds = None
        inductor_random_default_1: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default([8, 1024, 512], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_scalar_1: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul_tensor_4: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_scalar_1, mul_tensor_3);  gt_scalar_1 = mul_tensor_3 = None
        mul_tensor_5: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_4, 1.1111111111111112);  mul_tensor_4 = None
        view_default_1: "f32[8192, 512]" = torch.ops.aten.view.default(mul_tensor_5, _shape_param_1);  _shape_param_1 = None
        view_default_2: "f32[8192, 512]" = torch.ops.aten.view.default(mul_tensor_5, _shape_param_2);  _shape_param_2 = None
        view_default_3: "f32[8192, 512]" = torch.ops.aten.view.default(mul_tensor_5, _shape_param_3);  _shape_param_3 = None
        view_default_4: "f32[8192, 512]" = torch.ops.aten.view.default(mul_tensor_5, _shape_param_4);  _shape_param_4 = None
        view_default_5: "f32[8192, 512]" = torch.ops.aten.view.default(mul_tensor_5, _shape_param_5);  _shape_param_5 = None
        view_default_6: "f32[8192, 512]" = torch.ops.aten.view.default(mul_tensor_5, _shape_param_6);  _shape_param_6 = None
        view_default_7: "f32[8192, 512]" = torch.ops.aten.view.default(mul_tensor_5, _shape_param_7);  _shape_param_7 = None
        view_default_8: "f32[8192, 512]" = torch.ops.aten.view.default(mul_tensor_5, _shape_param_8);  _shape_param_8 = None
        view_default_9: "f32[8192, 512]" = torch.ops.aten.view.default(mul_tensor_5, _shape_param_9);  _shape_param_9 = None
        view_default_10: "f32[8192, 512]" = torch.ops.aten.view.default(mul_tensor_5, _shape_param_10);  _shape_param_10 = None
        view_default_11: "f32[8192, 512]" = torch.ops.aten.view.default(mul_tensor_5, _shape_param_11);  _shape_param_11 = None
        view_default_12: "f32[8192, 512]" = torch.ops.aten.view.default(mul_tensor_5, _shape_param_12);  mul_tensor_5 = _shape_param_12 = None
        return (view_default_1, view_default_2, view_default_3, view_default_4, view_default_5, view_default_6, view_default_7, view_default_8, view_default_9, view_default_10, view_default_11, view_default_12)

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
