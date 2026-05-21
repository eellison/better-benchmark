"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_train_000
Pattern hash: 767e1de714d1
Shape hash: b9ba30e7
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
_shapes_config = "(T([128, 640, 8, 8], f32), T([640], f32), T([640], f32), T([640], f32), T([640], f32), S([128, 640]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_34: "f32[128, 640, 8, 8]", arg306_1: "f32[640]", arg307_1: "f32[640]", arg308_1: "f32[640]", arg309_1: "f32[640]", _shape_param_0):
        # No stacktrace found for following nodes
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 640, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 640, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 640, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt_default: "f32[1, 640, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_1);  convolution_34 = None
        mul_tensor: "f32[128, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        squeeze_dims: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        mul_tensor_1: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1);  squeeze_dims = None
        mul_tensor_2: "f32[640]" = torch.ops.aten.mul.Tensor(arg306_1, 0.9)
        add_tensor_1: "f32[640]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        squeeze_dims_1: "f32[640]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_3: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, 1.0001220852154804);  squeeze_dims_1 = None
        mul_tensor_4: "f32[640]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.1);  mul_tensor_3 = None
        mul_tensor_5: "f32[640]" = torch.ops.aten.mul.Tensor(arg307_1, 0.9)
        add_tensor_2: "f32[640]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(arg308_1, -1);  arg308_1 = None
        unsqueeze_default_1: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_6: "f32[128, 640, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(arg309_1, -1);  arg309_1 = None
        unsqueeze_default_3: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[128, 640, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_6, unsqueeze_default_3);  mul_tensor_6 = unsqueeze_default_3 = None
        neg_default: "f32[128, 640, 8, 8]" = torch.ops.aten.neg.default(add_tensor_3)
        exp_default: "f32[128, 640, 8, 8]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_4: "f32[128, 640, 8, 8]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[128, 640, 8, 8]" = torch.ops.aten.div.Tensor(add_tensor_3, add_tensor_4);  add_tensor_3 = add_tensor_4 = None
        mean_dim: "f32[128, 640, 1, 1]" = torch.ops.aten.mean.dim(div_tensor, [-1, -2], True);  div_tensor = None
        as_strided_default: "f32[128, 640, 1, 1]" = torch.ops.aten.as_strided.default(mean_dim, [128, 640, 1, 1], [640, 1, 640, 640]);  mean_dim = None
        view_default: "f32[128, 640]" = torch.ops.aten.view.default(as_strided_default, _shape_param_0);  as_strided_default = _shape_param_0 = None
        copy__default: "f32[640]" = torch.ops.aten.copy_.default(arg306_1, add_tensor_1);  arg306_1 = add_tensor_1 = None
        copy__default_1: "f32[640]" = torch.ops.aten.copy_.default(arg307_1, add_tensor_2);  arg307_1 = add_tensor_2 = None
        return (copy__default, copy__default_1, view_default)



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
