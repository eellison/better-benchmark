"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_infer_000
Pattern hash: f2580942698a
Shape hash: 73126f4c
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
_shapes_config = "(T([25088, 512], f32), T([128, 196, 512], f32), T([2048], f32), T([2048], f32), S([128, 196, 512]), S([128, 14, 14, 512]), S([128, 7, 2, 7, 2, 512]), S([128, 7, 7, 2048]), S([6272, 2048]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_87: "f32[25088, 512]", view_594: "f32[128, 196, 512]", arg330_1: "f32[2048]", arg331_1: "f32[2048]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view_default: "f32[128, 196, 512]" = torch.ops.aten.view.default(addmm_87, _shape_param_0);  addmm_87 = _shape_param_0 = None
        add_tensor: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_594, view_default);  view_594 = view_default = None
        view_default_1: "f32[128, 14, 14, 512]" = torch.ops.aten.view.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None
        view_default_2: "f32[128, 7, 2, 7, 2, 512]" = torch.ops.aten.view.default(view_default_1, _shape_param_2);  view_default_1 = _shape_param_2 = None
        permute_default: "f32[128, 7, 7, 2, 2, 512]" = torch.ops.aten.permute.default(view_default_2, [0, 1, 3, 4, 2, 5]);  view_default_2 = None
        clone_default: "f32[128, 7, 7, 2, 2, 512]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_3: "f32[128, 7, 7, 2048]" = torch.ops.aten.view.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(view_default_3, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 7, 7, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 7, 7, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[128, 7, 7, 2048]" = torch.ops.aten.sub.Tensor(view_default_3, getitem_1);  view_default_3 = getitem_1 = None
        add_tensor_1: "f32[128, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[128, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[128, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[128, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor, arg330_1);  mul_tensor = arg330_1 = None
        add_tensor_2: "f32[128, 7, 7, 2048]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg331_1);  mul_tensor_1 = arg331_1 = None
        view_default_4: "f32[6272, 2048]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_4);  add_tensor_2 = _shape_param_4 = None
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
