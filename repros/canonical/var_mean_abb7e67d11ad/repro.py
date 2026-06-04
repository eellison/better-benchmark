"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_train_006
Pattern hash: abb7e67d11ad
Shape hash: 868396b3
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8192, 1024], f32), T([4], i64), T([64, 128, 1024], f32), T([1024], f32), T([1024], f32), S([64, 128, 1024]), S([8192, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_7: "f32[8192, 1024]", inductor_seeds: "i64[4]", add_2: "f32[64, 128, 1024]", arg24_1: "f32[1024]", arg25_1: "f32[1024]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[64, 128, 1024]" = torch.ops.aten.view.default(addmm_7, _shape_param_0);  addmm_7 = _shape_param_0 = None
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 2);  inductor_seeds = None
        inductor_random_default: "f32[64, 128, 1024]" = torch.ops.prims.inductor_random.default([64, 128, 1024], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[64, 128, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(gt_scalar, view_default);  gt_scalar = view_default = None
        mul_tensor_1: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        add_tensor: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(add_2, mul_tensor_1);  add_2 = mul_tensor_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[64, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[64, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[64, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[64, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor_2: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_3: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, arg24_1);  mul_tensor_2 = arg24_1 = None
        add_tensor_2: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_3, arg25_1);  mul_tensor_3 = arg25_1 = None
        view_default_1: "f32[8192, 1024]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        div_tensor: "f32[64, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_default, 1024);  rsqrt_default = None
        return (view_default_1, div_tensor)

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
