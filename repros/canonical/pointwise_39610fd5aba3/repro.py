"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_train
Pattern hash: 39610fd5aba3
Shape hash: d102a86e
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
    def forward(self, arg0_1: "f32[1024, 16, 64]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        unsqueeze: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, 3);  convert_element_type = None
        unsqueeze_1: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 4);  unsqueeze = None
        view: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_1, _shape_param_0);  unsqueeze_1 = _shape_param_0 = None
        squeeze: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view, 0);  view = None
        return squeeze



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
