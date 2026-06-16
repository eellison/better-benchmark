"""
Standalone repro captured via capture_hook.
Label: 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_train
Pattern hash: 514cc391d8e4
Shape hash: a3b18231
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[2048, 2560]", arg1_1: "bf16[2048, 2560]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        view_1: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(arg1_1, _shape_param_1);  arg1_1 = _shape_param_1 = None
        convert_element_type_1: "f32[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        add: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(convert_element_type, convert_element_type_1);  convert_element_type = convert_element_type_1 = None
        return add



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
