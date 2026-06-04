"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Albert_infer_000
Pattern hash: 2ceed1dd2572
Shape hash: 8a8123f1
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([512, 128], f16), T([128], f16), T([128], f16), S([1, 512, 128]), S([512, 128]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_73: "f16[512, 128]", arg28_1: "f16[128]", arg29_1: "f16[128]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f16[1, 512, 128]" = torch.ops.aten.view.default(addmm_73, _shape_param_0);  addmm_73 = _shape_param_0 = None
        mul_tensor: "f16[1, 512, 128]" = torch.ops.aten.mul.Tensor(view_default, 0.5)
        pow_tensor_scalar: "f16[1, 512, 128]" = torch.ops.aten.pow.Tensor_Scalar(view_default, 3.0)
        mul_tensor_1: "f16[1, 512, 128]" = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor: "f16[1, 512, 128]" = torch.ops.aten.add.Tensor(view_default, mul_tensor_1);  view_default = mul_tensor_1 = None
        mul_tensor_2: "f16[1, 512, 128]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default: "f16[1, 512, 128]" = torch.ops.aten.tanh.default(mul_tensor_2);  mul_tensor_2 = None
        add_tensor_1: "f16[1, 512, 128]" = torch.ops.aten.add.Tensor(tanh_default, 1.0);  tanh_default = None
        mul_tensor_3: "f16[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor_1);  mul_tensor = add_tensor_1 = None
        convert_element_type_default: "f32[1, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.float32);  mul_tensor_3 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1);  convert_element_type_default = getitem_1 = None
        add_tensor_2: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor_4: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_5: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_4, arg28_1);  mul_tensor_4 = arg28_1 = None
        add_tensor_3: "f32[1, 512, 128]" = torch.ops.aten.add.Tensor(mul_tensor_5, arg29_1);  mul_tensor_5 = arg29_1 = None
        convert_element_type_default_1: "f16[1, 512, 128]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float16);  add_tensor_3 = None
        view_default_1: "f16[512, 128]" = torch.ops.aten.view.default(convert_element_type_default_1, _shape_param_1);  convert_element_type_default_1 = _shape_param_1 = None
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
