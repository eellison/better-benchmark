"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_infer_000
Pattern hash: 9ce39d6fae7b
Shape hash: 751024d7
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([25088, 512], f32), T([512], f32), T([512], f32), S([128, 14, 14, 512]), S([128, 2, 7, 2, 7, 512]), S([-1, 7, 7, 512]), S([-1, 49, 512]), S([25088, 512]))"

class Repro(torch.nn.Module):
    def forward(self, mm_1: "f32[25088, 512]", arg69_1: "f32[512]", arg70_1: "f32[512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view_default: "f32[128, 14, 14, 512]" = torch.ops.aten.view.default(mm_1, _shape_param_0);  mm_1 = _shape_param_0 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(view_default, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 14, 14, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 14, 14, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_default, getitem_1);  view_default = getitem_1 = None
        add_tensor: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, arg69_1);  mul_tensor = arg69_1 = None
        add_tensor_1: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg70_1);  mul_tensor_1 = arg70_1 = None
        view_default_1: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(add_tensor_1, _shape_param_1);  add_tensor_1 = _shape_param_1 = None
        permute_default: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_default_1, [0, 1, 3, 2, 4, 5]);  view_default_1 = None
        clone_default: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_2: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None
        view_default_3: "f32[512, 49, 512]" = torch.ops.aten.view.default(view_default_2, _shape_param_3);  view_default_2 = _shape_param_3 = None
        view_default_4: "f32[25088, 512]" = torch.ops.aten.view.default(view_default_3, _shape_param_4);  view_default_3 = _shape_param_4 = None
        return view_default_4

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
