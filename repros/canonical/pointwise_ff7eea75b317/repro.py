"""
Standalone repro captured via capture_hook.
Label: timm_timm_visformer_small_infer_infer_000
Pattern hash: ff7eea75b317
Shape hash: a109de2f
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 2304, 7, 7], f32), S([128, 3, 6, 128, 49]), S([128, 6, 49, 128]), S([768, 49, 128]), S([128, 6, 128, 49]), S([768, 128, 49]), S([128, 6, 49, 128]), S([768, 49, 128]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_53: "f32[128, 2304, 7, 7]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        view_default: "f32[128, 3, 6, 128, 49]" = torch.ops.aten.view.default(convolution_53, _shape_param_0);  convolution_53 = _shape_param_0 = None
        permute_default: "f32[3, 128, 6, 49, 128]" = torch.ops.aten.permute.default(view_default, [1, 0, 2, 4, 3]);  view_default = None
        unbind_int = torch.ops.aten.unbind.int(permute_default);  permute_default = None
        getitem: "f32[128, 6, 49, 128]" = unbind_int[0]
        getitem_1: "f32[128, 6, 49, 128]" = unbind_int[1]
        getitem_2: "f32[128, 6, 49, 128]" = unbind_int[2];  unbind_int = None
        expand_default: "f32[128, 6, 49, 128]" = torch.ops.aten.expand.default(getitem, _shape_param_1);  getitem = _shape_param_1 = None
        clone_default: "f32[128, 6, 49, 128]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_1: "f32[768, 49, 128]" = torch.ops.aten.view.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None
        permute_default_1: "f32[128, 6, 128, 49]" = torch.ops.aten.permute.default(getitem_1, [0, 1, 3, 2]);  getitem_1 = None
        expand_default_1: "f32[128, 6, 128, 49]" = torch.ops.aten.expand.default(permute_default_1, _shape_param_3);  permute_default_1 = _shape_param_3 = None
        clone_default_1: "f32[128, 6, 128, 49]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_2: "f32[768, 128, 49]" = torch.ops.aten.view.default(clone_default_1, _shape_param_4);  clone_default_1 = _shape_param_4 = None
        expand_default_2: "f32[128, 6, 49, 128]" = torch.ops.aten.expand.default(getitem_2, _shape_param_5);  getitem_2 = _shape_param_5 = None
        clone_default_2: "f32[128, 6, 49, 128]" = torch.ops.aten.clone.default(expand_default_2, memory_format = torch.contiguous_format);  expand_default_2 = None
        view_default_3: "f32[768, 49, 128]" = torch.ops.aten.view.default(clone_default_2, _shape_param_6);  clone_default_2 = _shape_param_6 = None
        return (view_default_1, view_default_2, view_default_3)

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
