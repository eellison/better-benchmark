"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_infer
Pattern hash: bae56b847d32
Shape hash: 86e210b7
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
    def forward(self, arg0_1: "i64[197, 197]", arg1_1: "bf16[732, 12]", _shape_param_0):
        # No stacktrace found for following nodes
        view: "i64[38809]" = torch.ops.aten.view.default(arg0_1, [-1]);  arg0_1 = None
        index: "bf16[38809, 12]" = torch.ops.aten.index.Tensor(arg1_1, [view]);  arg1_1 = view = None
        view_1: "bf16[197, 197, 12]" = torch.ops.aten.view.default(index, _shape_param_0);  index = _shape_param_0 = None
        permute: "bf16[12, 197, 197]" = torch.ops.aten.permute.default(view_1, [2, 0, 1]);  view_1 = None
        clone: "bf16[12, 197, 197]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        unsqueeze: "bf16[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone, 0);  clone = None
        return unsqueeze



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
