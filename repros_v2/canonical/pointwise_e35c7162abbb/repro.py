"""
Standalone repro captured via capture_hook.
Label: hf_YituTechConvBert_train
Pattern hash: e35c7162abbb
Shape hash: d20f46e2
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
    def forward(self, arg0_1: "bf16[16384, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view: "bf16[32, 512, 768]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        view_1: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        slice_1: "bf16[32, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_1, 2, 0, 6)
        slice_2: "bf16[32, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_1, 2, 6, 12);  view_1 = None
        view_2: "bf16[16384, 384]" = torch.ops.aten.view.default(slice_2, _shape_param_2);  slice_2 = _shape_param_2 = None
        permute: "bf16[32, 6, 512, 64]" = torch.ops.aten.permute.default(slice_1, [0, 2, 1, 3]);  slice_1 = None
        clone: "bf16[32, 6, 512, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        clone_1: "bf16[16384, 384]" = torch.ops.aten.clone.default(view_2, memory_format = torch.contiguous_format);  view_2 = None
        view_3: "bf16[98304, 64, 1]" = torch.ops.aten.view.default(clone_1, _shape_param_3);  clone_1 = _shape_param_3 = None
        return (clone, view_3)



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
