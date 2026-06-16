"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_infer
Pattern hash: a42e20fae37b
Shape hash: 3d46c2a2
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
    def forward(self, arg0_1: "bf16[25216, 2304]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view: "bf16[128, 197, 2304]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        view_1: "bf16[128, 197, 3, 12, 64]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute: "bf16[3, 128, 12, 197, 64]" = torch.ops.aten.permute.default(view_1, [2, 0, 3, 1, 4]);  view_1 = None
        unbind = torch.ops.aten.unbind.int(permute);  permute = None
        getitem: "bf16[128, 12, 197, 64]" = unbind[0]
        getitem_1: "bf16[128, 12, 197, 64]" = unbind[1]
        getitem_2: "bf16[128, 12, 197, 64]" = unbind[2];  unbind = None
        return (getitem, getitem_1, getitem_2)



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
