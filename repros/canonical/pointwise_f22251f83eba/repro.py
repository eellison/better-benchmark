"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_train
Pattern hash: f22251f83eba
Shape hash: 144bae60
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
    def forward(self, arg0_1: "bf16[256, 64, 512]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        permute: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(view, [0, 1, 4, 3, 2]);  view = None
        permute_1: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute, [3, 0, 1, 4, 2]);  permute = None
        squeeze: "bf16[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1, 4);  permute_1 = None
        view_1: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(squeeze, _shape_param_1);  squeeze = _shape_param_1 = None
        permute_2: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.permute.default(view_1, [0, 1, 4, 2, 3]);  view_1 = None
        clone: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_2: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        squeeze_1: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_2, 0);  view_2 = None
        return squeeze_1



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
