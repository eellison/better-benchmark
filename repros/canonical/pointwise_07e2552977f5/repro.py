"""
Standalone repro captured via capture_hook.
Label: hf_YituTechConvBert_infer
Pattern hash: 07e2552977f5
Shape hash: 75c13bb9
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
    def forward(self, arg0_1: "bf16[16384, 384]", arg1_1: "bf16[32, 384, 512]", arg2_1: "bf16[384, 1]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view: "bf16[32, 512, 384]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        view_1: "bf16[32, 512, 6, 64]" = torch.ops.aten.view.default(view, _shape_param_1);  _shape_param_1 = None
        permute: "bf16[32, 6, 512, 64]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        add: "bf16[32, 384, 512]" = torch.ops.aten.add.Tensor(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        permute_1: "bf16[32, 512, 384]" = torch.ops.aten.permute.default(add, [0, 2, 1]);  add = None
        mul: "bf16[32, 512, 384]" = torch.ops.aten.mul.Tensor(permute_1, view);  permute_1 = view = None
        clone: "bf16[32, 512, 384]" = torch.ops.aten.clone.default(mul, memory_format = torch.contiguous_format);  mul = None
        view_2: "bf16[16384, 384]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
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
