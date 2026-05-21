"""
Standalone repro captured via capture_hook.
Label: torchbench_phlippe_densenet_train_000
Pattern hash: 626fb0807512
Shape hash: ac3b5b46
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
_shapes_config = "(T([128, 16, 4, 4], f32), T([128, 168, 4, 4], f32), T([184], f32), T([184], f32), T([184], f32), T([184], f32), S([128, 184]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_51: "f32[128, 16, 4, 4]", cat_22: "f32[128, 168, 4, 4]", arg310_1: "f32[184]", arg311_1: "f32[184]", arg312_1: "f32[184]", arg313_1: "f32[184]", _shape_param_0):
        # No stacktrace found for following nodes
        cat_default: "f32[128, 184, 4, 4]" = torch.ops.aten.cat.default([convolution_51, cat_22], 1);  convolution_51 = cat_22 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(cat_default, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 184, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 184, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 184, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt_default: "f32[1, 184, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 184, 4, 4]" = torch.ops.aten.sub.Tensor(cat_default, getitem_1);  cat_default = None
        mul_tensor: "f32[128, 184, 4, 4]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        squeeze_dims: "f32[184]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        mul_tensor_1: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1);  squeeze_dims = None
        mul_tensor_2: "f32[184]" = torch.ops.aten.mul.Tensor(arg310_1, 0.9)
        add_tensor_1: "f32[184]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        squeeze_dims_1: "f32[184]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_3: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, 1.0004885197850513);  squeeze_dims_1 = None
        mul_tensor_4: "f32[184]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.1);  mul_tensor_3 = None
        mul_tensor_5: "f32[184]" = torch.ops.aten.mul.Tensor(arg311_1, 0.9)
        add_tensor_2: "f32[184]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg312_1, -1);  arg312_1 = None
        unsqueeze_default_1: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_6: "f32[128, 184, 4, 4]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(arg313_1, -1);  arg313_1 = None
        unsqueeze_default_3: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[128, 184, 4, 4]" = torch.ops.aten.add.Tensor(mul_tensor_6, unsqueeze_default_3);  mul_tensor_6 = unsqueeze_default_3 = None
        relu_default: "f32[128, 184, 4, 4]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None
        mean_dim: "f32[128, 184, 1, 1]" = torch.ops.aten.mean.dim(relu_default, [-1, -2], True);  relu_default = None
        view_default: "f32[128, 184]" = torch.ops.aten.view.default(mean_dim, _shape_param_0);  mean_dim = _shape_param_0 = None
        copy__default: "f32[184]" = torch.ops.aten.copy_.default(arg310_1, add_tensor_1);  arg310_1 = add_tensor_1 = None
        copy__default_1: "f32[184]" = torch.ops.aten.copy_.default(arg311_1, add_tensor_2);  arg311_1 = add_tensor_2 = None
        return (view_default, copy__default, copy__default_1)



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
