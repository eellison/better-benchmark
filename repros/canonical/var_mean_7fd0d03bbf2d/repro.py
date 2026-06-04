"""
Standalone repro captured via capture_hook.
Label: torchbench_timm_nfnet_infer_000
Pattern hash: 7fd0d03bbf2d
Shape hash: b2f656f9
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([3072, 1536, 1, 1], f16), T([3072, 1, 1, 1], f16), S([1, 3072, -1]), S([3072, 1536, 1, 1]))"

class Repro(torch.nn.Module):
    def forward(self, arg229_1: "f16[3072, 1536, 1, 1]", arg230_1: "f16[3072, 1, 1, 1]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f16[1, 3072, 1536]" = torch.ops.aten.view.default(arg229_1, _shape_param_0);  arg229_1 = _shape_param_0 = None
        convert_element_type_default: "f32[1, 3072, 1536]" = torch.ops.prims.convert_element_type.default(view_default, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [0, 2], correction = 0, keepdim = True);  convert_element_type_default = None
        getitem: "f32[1, 3072, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 3072, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[1, 3072, 1536]" = torch.ops.aten.sub.Tensor(view_default, getitem_1);  view_default = getitem_1 = None
        add_tensor: "f32[1, 3072, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 3072, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[1, 3072, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f16[3072, 1, 1, 1]" = torch.ops.aten.mul.Tensor(arg230_1, 0.02551551815399144);  arg230_1 = None
        view_default_1: "f16[3072]" = torch.ops.aten.view.default(mul_tensor_1, [-1]);  mul_tensor_1 = None
        unsqueeze_default: "f16[3072, 1]" = torch.ops.aten.unsqueeze.default(view_default_1, -1);  view_default_1 = None
        mul_tensor_2: "f32[1, 3072, 1536]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default);  mul_tensor = unsqueeze_default = None
        convert_element_type_default_1: "f16[1, 3072, 1536]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.float16);  mul_tensor_2 = None
        view_default_2: "f16[3072, 1536, 1, 1]" = torch.ops.aten.view.default(convert_element_type_default_1, _shape_param_1);  convert_element_type_default_1 = _shape_param_1 = None
        return view_default_2

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
