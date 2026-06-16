"""
Standalone repro captured via capture_hook.
Label: hf_Qwen/Qwen3-0.6B_infer
Pattern hash: b3f01ad9586c
Shape hash: ed385436
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
    def forward(self, arg0_1: "bf16[1000, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view: "bf16[1, 1000, 1024]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        view_1: "bf16[1, 1000, 8, 128]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute: "bf16[1, 8, 1000, 128]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        unsqueeze: "bf16[1, 8, 1, 1000, 128]" = torch.ops.aten.unsqueeze.default(permute, 2)
        expand: "bf16[1, 8, 2, 1000, 128]" = torch.ops.aten.expand.default(unsqueeze, _shape_param_2);  unsqueeze = _shape_param_2 = None
        clone: "bf16[1, 8, 2, 1000, 128]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_2: "bf16[1, 16, 1000, 128]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        return (permute, view_2)



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
