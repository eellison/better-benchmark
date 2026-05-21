"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_infer_000
Pattern hash: 327c3b97bf10
Shape hash: 197afd65
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
_shapes_config = "(T([4096, 128], f32), T([128], f32), T([128], f32), S([8, 512, 128]), S([4096, 128]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_73: "f32[4096, 128]", arg28_1: "f32[128]", arg29_1: "f32[128]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[8, 512, 128]" = torch.ops.aten.view.default(addmm_73, _shape_param_0);  addmm_73 = _shape_param_0 = None
        mul_tensor: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(view_default, 0.5)
        pow_tensor_scalar: "f32[8, 512, 128]" = torch.ops.aten.pow.Tensor_Scalar(view_default, 3.0)
        mul_tensor_1: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(view_default, mul_tensor_1);  view_default = mul_tensor_1 = None
        mul_tensor_2: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default: "f32[8, 512, 128]" = torch.ops.aten.tanh.default(mul_tensor_2);  mul_tensor_2 = None
        add_tensor_1: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(tanh_default, 1.0);  tanh_default = None
        mul_tensor_3: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor_1);  mul_tensor = add_tensor_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(mul_tensor_3, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(mul_tensor_3, getitem_1);  mul_tensor_3 = getitem_1 = None
        add_tensor_2: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor_4: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_5: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_4, arg28_1);  mul_tensor_4 = arg28_1 = None
        add_tensor_3: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(mul_tensor_5, arg29_1);  mul_tensor_5 = arg29_1 = None
        view_default_1: "f32[4096, 128]" = torch.ops.aten.view.default(add_tensor_3, _shape_param_1);  add_tensor_3 = _shape_param_1 = None
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
