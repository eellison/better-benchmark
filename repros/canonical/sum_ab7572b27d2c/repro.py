"""
Standalone repro captured via capture_hook.
Label: hf_LayoutLMForMaskedLM_train_001
Pattern hash: ab7572b27d2c
Shape hash: 4d35693e
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([32, 12, 512, 64], f32, stride=(393216, 64, 768, 1)), S([32, 512, 768]), S([16384, 768]), S([768]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_45: "f32[32, 12, 512, 64]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        permute_default: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_45, [0, 2, 1, 3]);  getitem_45 = None
        view_default: "f32[32, 512, 768]" = torch.ops.aten.view.default(permute_default, _shape_param_0);  permute_default = _shape_param_0 = None
        clone_default: "f32[32, 512, 768]" = torch.ops.aten.clone.default(view_default, memory_format = torch.contiguous_format);  view_default = None
        view_default_1: "f32[16384, 768]" = torch.ops.aten.view.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        permute_default_1: "f32[768, 16384]" = torch.ops.aten.permute.default(view_default_1, [1, 0])
        sum_dim_int_list: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_default_1, [0], True);  view_default_1 = None
        view_default_2: "f32[768]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_2);  sum_dim_int_list = _shape_param_2 = None
        return (permute_default_1, view_default_2)

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
