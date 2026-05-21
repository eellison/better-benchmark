"""
Standalone repro captured via capture_hook.
Label: torchbench_timm_efficientnet_infer_000
Pattern hash: 69b13df5a532
Shape hash: 584aabd9
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
_shapes_config = "(T([64, 48, 1, 1], f16))"

class Repro(torch.nn.Module):
    def forward(self, convolution_77: "f16[64, 48, 1, 1]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[64, 48, 1, 1]" = torch.ops.prims.convert_element_type.default(convolution_77, torch.float32);  convolution_77 = None
        neg_default: "f32[64, 48, 1, 1]" = torch.ops.aten.neg.default(convert_element_type_default)
        exp_default: "f32[64, 48, 1, 1]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[64, 48, 1, 1]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[64, 48, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, add_tensor);  convert_element_type_default = add_tensor = None
        convert_element_type_default_1: "f16[64, 48, 1, 1]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.float16);  div_tensor = None
        return convert_element_type_default_1



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
