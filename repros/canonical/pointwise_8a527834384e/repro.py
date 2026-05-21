"""
Standalone repro captured via capture_hook.
Label: hf_DistillGPT2_train_000
Pattern hash: 8a527834384e
Shape hash: 39e37d3d
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([16384, 2304], f32), S([32, 512, 2304]), S([32, 512, 12, 64]), S([32, 512, 12, 64]), S([32, 512, 12, 64]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_20: "f32[16384, 2304]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f32[32, 512, 2304]" = torch.ops.aten.view.default(addmm_20, _shape_param_0);  addmm_20 = _shape_param_0 = None
        split_tensor = torch.ops.aten.split.Tensor(view_default, 768, 2);  view_default = None
        getitem: "f32[32, 512, 768]" = split_tensor[0]
        getitem_1: "f32[32, 512, 768]" = split_tensor[1]
        getitem_2: "f32[32, 512, 768]" = split_tensor[2];  split_tensor = None
        view_default_1: "f32[32, 512, 12, 64]" = torch.ops.aten.view.default(getitem_1, _shape_param_1);  getitem_1 = _shape_param_1 = None
        permute_default: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3]);  view_default_1 = None
        view_default_2: "f32[32, 512, 12, 64]" = torch.ops.aten.view.default(getitem_2, _shape_param_2);  getitem_2 = _shape_param_2 = None
        permute_default_1: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        view_default_3: "f32[32, 512, 12, 64]" = torch.ops.aten.view.default(getitem, _shape_param_3);  getitem = _shape_param_3 = None
        permute_default_2: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_default_3, [0, 2, 1, 3]);  view_default_3 = None
        return (permute_default_2, permute_default_1, permute_default)



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
