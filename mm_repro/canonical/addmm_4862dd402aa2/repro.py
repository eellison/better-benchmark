"""
Standalone repro captured via capture_hook.
Label: repros/models/timm/train/mobilevit_s/full_graph_000.py
Pattern hash: 4862dd402aa2
Shape hash: eaf94f9f
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
    def forward(self, arg0_1: "f32[432]", arg1_1: "f32[432, 144]", arg2_1: "bf16[131072, 144]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[432]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[432, 144]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        permute: "bf16[144, 432]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[131072, 432]" = torch.ops.aten.addmm.default(convert_element_type, arg2_1, permute);  convert_element_type = arg2_1 = permute = None
        view: "bf16[512, 256, 432]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        view_1: "bf16[512, 256, 3, 4, 36]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute_1: "bf16[3, 512, 4, 256, 36]" = torch.ops.aten.permute.default(view_1, [2, 0, 3, 1, 4]);  view_1 = None
        unbind = torch.ops.aten.unbind.int(permute_1);  permute_1 = None
        getitem: "bf16[512, 4, 256, 36]" = unbind[0]
        getitem_1: "bf16[512, 4, 256, 36]" = unbind[1]
        getitem_2: "bf16[512, 4, 256, 36]" = unbind[2];  unbind = None
        constant_pad_nd: "bf16[512, 4, 256, 40]" = torch.ops.aten.constant_pad_nd.default(getitem, _shape_param_2, 0.0);  getitem = _shape_param_2 = None
        constant_pad_nd_1: "bf16[512, 4, 256, 40]" = torch.ops.aten.constant_pad_nd.default(getitem_1, _shape_param_3, 0.0);  getitem_1 = _shape_param_3 = None
        constant_pad_nd_2: "bf16[512, 4, 256, 40]" = torch.ops.aten.constant_pad_nd.default(getitem_2, _shape_param_4, 0.0);  getitem_2 = _shape_param_4 = None
        return (constant_pad_nd, constant_pad_nd_1, constant_pad_nd_2)



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
