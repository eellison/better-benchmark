"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Albert_infer_000
Pattern hash: 0decf4e7ecd3
Shape hash: 07b94259
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
_shapes_config = "(T([512, 768], f16), S([1, 512, 768]), S([1, 512, -1, 64]), S([1, 12, 64, 512]), S([12, 64, 512]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_68: "f16[512, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f16[1, 512, 768]" = torch.ops.aten.view.default(addmm_68, _shape_param_0);  addmm_68 = _shape_param_0 = None
        view_default_1: "f16[1, 512, 12, 64]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        permute_default: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3]);  view_default_1 = None
        convert_element_type_default: "f32[1, 12, 512, 64]" = torch.ops.prims.convert_element_type.default(permute_default, torch.float32);  permute_default = None
        permute_default_1: "f32[1, 12, 64, 512]" = torch.ops.aten.permute.default(convert_element_type_default, [0, 1, 3, 2]);  convert_element_type_default = None
        mul_scalar: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_1, 0.3535533905932738);  permute_default_1 = None
        expand_default: "f32[1, 12, 64, 512]" = torch.ops.aten.expand.default(mul_scalar, _shape_param_2);  mul_scalar = _shape_param_2 = None
        view_default_2: "f32[12, 64, 512]" = torch.ops.aten.view.default(expand_default, _shape_param_3);  expand_default = _shape_param_3 = None
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
