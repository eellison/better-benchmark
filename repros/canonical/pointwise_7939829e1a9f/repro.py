"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train
Pattern hash: 7939829e1a9f
Shape hash: 07bfd41e
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
    def forward(self, arg0_1: "bf16[8192, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        permute: "bf16[1024, 8, 768]" = torch.ops.aten.permute.default(view, [1, 0, 2]);  view = None
        view_1: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(permute, _shape_param_1);  permute = _shape_param_1 = None
        permute_1: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1, [1, 0, 2, 3]);  view_1 = None
        permute_2: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_1, [0, 2, 1, 3]);  permute_1 = None
        clone: "bf16[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_2: "bf16[96, 4, 256, 64]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        view_3: "bf16[96, 4, 256, 64, 1]" = torch.ops.aten.view.default(view_2, _shape_param_3);  view_2 = _shape_param_3 = None
        permute_3: "bf16[96, 4, 256, 1, 64]" = torch.ops.aten.permute.default(view_3, [0, 1, 2, 4, 3]);  view_3 = None
        view_4: "bf16[384, 256, 64]" = torch.ops.aten.view.default(permute_3, _shape_param_4);  permute_3 = _shape_param_4 = None
        return view_4



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
