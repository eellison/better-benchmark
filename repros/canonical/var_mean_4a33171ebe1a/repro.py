"""
Standalone repro captured via capture_hook.
Label: hf_GoogleFnet_train_000
Pattern hash: 4a33171ebe1a
Shape hash: 9b24bf40
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
_shapes_config = "(T([16384, 768], f32), T([13], i64), T([32, 512, 768], f32), T([768], f32), T([768], f32), S([32, 512, 768]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_22: "f32[16384, 768]", inductor_seeds: "i64[13]", add_86: "f32[32, 512, 768]", arg96_1: "f32[768]", arg97_1: "f32[768]", _shape_param_0):
        # No stacktrace found for following nodes
        view_default: "f32[32, 512, 768]" = torch.ops.aten.view.default(addmm_22, _shape_param_0);  addmm_22 = _shape_param_0 = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 11);  inductor_seeds = None
        inductor_random_default: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default([32, 512, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 1e-30);  inductor_random_default = None
        mul_tensor: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_scalar, view_default);  gt_scalar = view_default = None
        mul_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.0);  mul_tensor = None
        add_tensor: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, add_86);  mul_tensor_1 = add_86 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, arg96_1);  mul_tensor_2 = arg96_1 = None
        add_tensor_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_3, arg97_1);  mul_tensor_3 = arg97_1 = None
        convert_element_type_default: "c64[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.complex64);  add_tensor_2 = None
        return convert_element_type_default



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
