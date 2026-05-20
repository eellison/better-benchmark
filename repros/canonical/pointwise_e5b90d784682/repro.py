"""
Standalone repro captured via capture_hook.
Label: inductor_timm_perf_cuda_h100-2-7-linux.aws.h100_graph49
Pattern hash: e5b90d784682
Shape hash: 5b0f1539
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([8, 12, 198, 64], f16, stride=(152064, 64, 768, 1)), T([768], f32), T([768, 768], f32), S([8, 198, 768]), S([1584, 768]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_181: "f16[8, 12, 198, 64]", arg143_1: "f32[768]", arg142_1: "f32[768, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        permute_default: "f16[8, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_181, [0, 2, 1, 3]);  getitem_181 = None
        view_default: "f16[8, 198, 768]" = torch.ops.aten.view.default(permute_default, _shape_param_0);  permute_default = _shape_param_0 = None
        convert_element_type_default: "f16[768]" = torch.ops.prims.convert_element_type.default(arg143_1, torch.float16);  arg143_1 = None
        convert_element_type_default_1: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(arg142_1, torch.float16);  arg142_1 = None
        view_default_1: "f16[1584, 768]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        permute_default_1: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        return (convert_element_type_default, view_default_1, permute_default_1)


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
