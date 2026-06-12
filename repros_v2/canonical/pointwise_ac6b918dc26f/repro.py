"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_train
Pattern hash: ac6b918dc26f
Shape hash: 2cdbce9d
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
    def forward(self, arg0_1: "bf16[256, 512, 64]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        permute: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view, [2, 0, 1, 4, 3]);  view = None
        view_1: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute, _shape_param_1);  permute = _shape_param_1 = None
        unsqueeze: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_1, 4);  view_1 = None
        permute_1: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze, [0, 1, 4, 3, 2]);  unsqueeze = None
        permute_2: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_1, [0, 1, 3, 4, 2]);  permute_1 = None
        clone: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_2: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        squeeze: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_2, 0)
        permute_3: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_2, [0, 2, 1]);  view_2 = None
        squeeze_1: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_3, 0);  permute_3 = None
        return (squeeze, squeeze_1)



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
