"""
Standalone repro captured via capture_hook.
Label: hf_DistillGPT2_infer
Pattern hash: 1611fb0c7845
Shape hash: 74f1c5d9
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
    def forward(self, arg0_1: "bf16[16384, 2304]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view: "bf16[32, 512, 2304]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        split = torch.ops.aten.split.Tensor(view, 768, 2);  view = None
        getitem: "bf16[32, 512, 768]" = split[0]
        getitem_1: "bf16[32, 512, 768]" = split[1]
        getitem_2: "bf16[32, 512, 768]" = split[2];  split = None
        view_1: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(getitem, _shape_param_1);  getitem = _shape_param_1 = None
        permute: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        view_2: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(getitem_1, _shape_param_2);  getitem_1 = _shape_param_2 = None
        permute_1: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        view_3: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(getitem_2, _shape_param_3);  getitem_2 = _shape_param_3 = None
        permute_2: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        return (permute, permute_1, permute_2)



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
