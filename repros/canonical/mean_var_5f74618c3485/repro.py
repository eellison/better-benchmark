"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train_000
Pattern hash: 5f74618c3485
Shape hash: 317524f3
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
_shapes_config = "(T([16384, 768], f32), T([61], i64), T([128, 128, 768], f32), T([768], f32), T([768], f32), S([128, 128, 768]), S([16384, 768]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_65: "f32[16384, 768]", inductor_seeds: "i64[61]", add_74: "f32[128, 128, 768]", arg181_1: "f32[768]", arg182_1: "f32[768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_65, _shape_param_0);  addmm_65 = _shape_param_0 = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 54)
        inductor_random_default: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_scalar, view_default);  gt_scalar = view_default = None
        mul_tensor_1: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        add_tensor: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_74, mul_tensor_1);  add_74 = mul_tensor_1 = None
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 55);  inductor_seeds = None
        inductor_random_default_1: "f32[128, 128, 768]" = torch.ops.prims.inductor_random.default([128, 128, 768], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_scalar_1: "b8[128, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul_tensor_2: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(gt_scalar_1, add_tensor);  gt_scalar_1 = add_tensor = None
        mul_tensor_3: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.1111111111111112);  mul_tensor_2 = None
        mean_dim: "f32[128, 128, 1]" = torch.ops.aten.mean.dim(mul_tensor_3, [-1], True)
        var_correction: "f32[128, 128, 1]" = torch.ops.aten.var.correction(mul_tensor_3, [-1], correction = 1.0, keepdim = True)
        sqrt_default: "f32[128, 128, 1]" = torch.ops.aten.sqrt.default(var_correction);  var_correction = None
        sub_tensor: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_3, mean_dim);  mul_tensor_3 = mean_dim = None
        mul_tensor_4: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(arg181_1, sub_tensor);  arg181_1 = sub_tensor = None
        add_tensor_1: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_default, 1e-06);  sqrt_default = None
        div_tensor: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_tensor_4, add_tensor_1);  mul_tensor_4 = add_tensor_1 = None
        add_tensor_2: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(div_tensor, arg182_1);  div_tensor = arg182_1 = None
        view_default_1: "f32[16384, 768]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
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
