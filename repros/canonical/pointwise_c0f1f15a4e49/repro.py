"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_infer
Pattern hash: c0f1f15a4e49
Shape hash: 2973ba0c
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
    def forward(self, arg0_1: "bf16[8, 1024, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        permute: "bf16[1024, 8, 768]" = torch.ops.aten.permute.default(arg0_1, [1, 0, 2]);  arg0_1 = None
        clone: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format)
        view: "bf16[8192, 768]" = torch.ops.aten.view.default(clone, _shape_param_0);  clone = _shape_param_0 = None
        clone_1: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format)
        view_1: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_1, _shape_param_1);  clone_1 = _shape_param_1 = None
        clone_2: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_2: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_2, _shape_param_2);  clone_2 = _shape_param_2 = None
        return (view, view_1, view_2)



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
