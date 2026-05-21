"""
Standalone repro captured via capture_hook.
Label: torchbench_mobilenet_v2_train_000
Pattern hash: 8d6fc761298a
Shape hash: d7975883
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
_shapes_config = "(T([128, 1280, 7, 7], f32), T([1280], f32), T([1280], f32), T([1280], f32), T([1280], f32), S([128, 1280]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_51: "f32[128, 1280, 7, 7]", arg309_1: "f32[1280]", arg310_1: "f32[1280]", arg311_1: "f32[1280]", arg312_1: "f32[1280]", _shape_param_0):
        # No stacktrace found for following nodes
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 1280, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 1280, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 1280, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt_default: "f32[1, 1280, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_1);  convolution_51 = None
        mul_tensor: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        squeeze_dims: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        mul_tensor_1: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1);  squeeze_dims = None
        mul_tensor_2: "f32[1280]" = torch.ops.aten.mul.Tensor(arg309_1, 0.9)
        add_tensor_1: "f32[1280]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        squeeze_dims_1: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_3: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, 1.0001594642002871);  squeeze_dims_1 = None
        mul_tensor_4: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.1);  mul_tensor_3 = None
        mul_tensor_5: "f32[1280]" = torch.ops.aten.mul.Tensor(arg310_1, 0.9)
        add_tensor_2: "f32[1280]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg311_1, -1);  arg311_1 = None
        unsqueeze_default_1: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_6: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg312_1, -1);  arg312_1 = None
        unsqueeze_default_3: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[128, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_6, unsqueeze_default_3);  mul_tensor_6 = unsqueeze_default_3 = None
        clamp_min_default: "f32[128, 1280, 7, 7]" = torch.ops.aten.clamp_min.default(add_tensor_3, 0.0);  add_tensor_3 = None
        clamp_max_default: "f32[128, 1280, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6.0);  clamp_min_default = None
        mean_dim: "f32[128, 1280, 1, 1]" = torch.ops.aten.mean.dim(clamp_max_default, [-1, -2], True);  clamp_max_default = None
        view_default: "f32[128, 1280]" = torch.ops.aten.view.default(mean_dim, _shape_param_0);  mean_dim = _shape_param_0 = None
        inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[128, 1280]" = torch.ops.prims.inductor_random.default([128, 1280], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[128, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.2);  inductor_random_default = None
        mul_tensor_7: "f32[128, 1280]" = torch.ops.aten.mul.Tensor(gt_scalar, view_default);  gt_scalar = view_default = None
        mul_tensor_8: "f32[128, 1280]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 1.25);  mul_tensor_7 = None
        copy__default: "f32[1280]" = torch.ops.aten.copy_.default(arg309_1, add_tensor_1);  arg309_1 = add_tensor_1 = None
        copy__default_1: "f32[1280]" = torch.ops.aten.copy_.default(arg310_1, add_tensor_2);  arg310_1 = add_tensor_2 = None
        return (mul_tensor_8, copy__default, copy__default_1)



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
