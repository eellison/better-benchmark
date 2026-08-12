"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_train
Pattern hash: 4e2a690272cd
Shape hash: 1e7ad64a
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
    def forward(self, arg0_1: "bf16[192, 64, 128]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view: "bf16[32, 6, 64, 128]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        permute: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view, [0, 1, 3, 2]);  view = None
        permute_1: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute, [0, 2, 1, 3]);  permute = None
        view_1: "bf16[32, 128, 384]" = torch.ops.aten.view.default(permute_1, _shape_param_1);  permute_1 = _shape_param_1 = None
        clone: "bf16[32, 128, 384]" = torch.ops.aten.clone.default(view_1, memory_format = torch.contiguous_format);  view_1 = None
        view_2: "bf16[4096, 384]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        permute_2: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_2, [1, 0])
        return (view_2, permute_2)



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
