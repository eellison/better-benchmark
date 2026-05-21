"""
Standalone repro captured via capture_hook.
Label: torchbench_llava_infer_000
Pattern hash: 391b0009ff83
Shape hash: 8063b7c1
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
_shapes_config = "(T([512, 11008], f16), T([512, 11008], f16), S([1, 512, 11008]), S([1, 512, 11008]), S([512, 11008]))"

class Repro(torch.nn.Module):
    def forward(self, mm_221: "f16[512, 11008]", mm_222: "f16[512, 11008]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f16[1, 512, 11008]" = torch.ops.aten.view.default(mm_221, _shape_param_0);  mm_221 = _shape_param_0 = None
        convert_element_type_default: "f32[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(view_default, torch.float32);  view_default = None
        neg_default: "f32[1, 512, 11008]" = torch.ops.aten.neg.default(convert_element_type_default)
        exp_default: "f32[1, 512, 11008]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[1, 512, 11008]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[1, 512, 11008]" = torch.ops.aten.div.Tensor(convert_element_type_default, add_tensor);  convert_element_type_default = add_tensor = None
        convert_element_type_default_1: "f16[1, 512, 11008]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.float16);  div_tensor = None
        view_default_1: "f16[1, 512, 11008]" = torch.ops.aten.view.default(mm_222, _shape_param_1);  mm_222 = _shape_param_1 = None
        mul_tensor: "f16[1, 512, 11008]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, view_default_1);  convert_element_type_default_1 = view_default_1 = None
        view_default_2: "f16[512, 11008]" = torch.ops.aten.view.default(mul_tensor, _shape_param_2);  mul_tensor = _shape_param_2 = None
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
