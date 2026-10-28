"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/AllenaiLongformerBase/full_graph_005.py
Pattern hash: 42be4dbe49aa
Shape hash: 72099925
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
    def forward(self, arg0_1: "bf16[6291456]", arg1_1: "bf16[8192, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        as_strided: "bf16[96, 2, 512, 64]" = torch.ops.aten.as_strided.default(arg0_1, _shape_param_0, _shape_param_1, 0);  arg0_1 = _shape_param_0 = _shape_param_1 = None
        view: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(as_strided, _shape_param_2);  as_strided = _shape_param_2 = None
        view_1: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view, _shape_param_3);  view = _shape_param_3 = None
        permute: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        permute_1: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute, [1, 0, 2, 3]);  permute = None
        view_2: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(permute_1, _shape_param_4);  permute_1 = _shape_param_4 = None
        div: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(view_2, 8.0);  view_2 = None
        view_3: "bf16[8192, 768]" = torch.ops.aten.view.default(div, _shape_param_5);  div = _shape_param_5 = None
        permute_2: "bf16[768, 8192]" = torch.ops.aten.permute.default(view_3, [1, 0]);  view_3 = None
        mm: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_2, arg1_1);  permute_2 = arg1_1 = None
        convert_element_type: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        return convert_element_type



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
