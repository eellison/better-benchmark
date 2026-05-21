"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Albert_infer_000
Pattern hash: 7c7235fc35c3
Shape hash: f7800ac8
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
_shapes_config = "(T([12, 512, 64], f32), S([1, 12, 512, 64]), S([1, 512, -1]), S([512, 768]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_23: "f32[12, 512, 64]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_23, _shape_param_0);  bmm_23 = _shape_param_0 = None
        convert_element_type_default: "f16[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(view_default, torch.float16);  view_default = None
        permute_default: "f16[1, 512, 12, 64]" = torch.ops.aten.permute.default(convert_element_type_default, [0, 2, 1, 3]);  convert_element_type_default = None
        clone_default: "f16[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_1: "f16[1, 512, 768]" = torch.ops.aten.view.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        view_default_2: "f16[512, 768]" = torch.ops.aten.view.default(view_default_1, _shape_param_2);  view_default_1 = _shape_param_2 = None
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
