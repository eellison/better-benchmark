"""
Standalone repro captured via capture_hook.
Label: repros/models/timm/train/visformer_small/full_graph_000.py
Pattern hash: c8e6fc2c14b9
Shape hash: 97e22389
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
    def forward(self, arg0_1: "bf16[128, 1152, 14, 14]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        view: "bf16[128, 3, 6, 64, 196]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        permute: "bf16[3, 128, 6, 196, 64]" = torch.ops.aten.permute.default(view, [1, 0, 2, 4, 3]);  view = None
        unbind = torch.ops.aten.unbind.int(permute);  permute = None
        getitem: "bf16[128, 6, 196, 64]" = unbind[0]
        getitem_1: "bf16[128, 6, 196, 64]" = unbind[1]
        getitem_2: "bf16[128, 6, 196, 64]" = unbind[2];  unbind = getitem_2 = None
        permute_1: "bf16[128, 6, 64, 196]" = torch.ops.aten.permute.default(getitem_1, [0, 1, 3, 2]);  getitem_1 = None
        expand: "bf16[128, 6, 196, 64]" = torch.ops.aten.expand.default(getitem, _shape_param_1);  getitem = _shape_param_1 = None
        clone: "bf16[128, 6, 196, 64]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_1: "bf16[768, 196, 64]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        expand_1: "bf16[128, 6, 64, 196]" = torch.ops.aten.expand.default(permute_1, _shape_param_3);  permute_1 = _shape_param_3 = None
        clone_1: "bf16[128, 6, 64, 196]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_2: "bf16[768, 64, 196]" = torch.ops.aten.view.default(clone_1, _shape_param_4);  clone_1 = _shape_param_4 = None
        constant_pad_nd: "bf16[768, 200, 64]" = torch.ops.aten.constant_pad_nd.default(view_1, _shape_param_5);  view_1 = _shape_param_5 = None
        constant_pad_nd_1: "bf16[768, 64, 200]" = torch.ops.aten.constant_pad_nd.default(view_2, _shape_param_6);  view_2 = _shape_param_6 = None
        bmm: "bf16[768, 200, 200]" = torch.ops.aten.bmm.default(constant_pad_nd, constant_pad_nd_1);  constant_pad_nd = constant_pad_nd_1 = None
        slice_1: "bf16[768, 196, 200]" = torch.ops.aten.slice.Tensor(bmm, 1, 0, -4)
        slice_2: "bf16[768, 196, 196]" = torch.ops.aten.slice.Tensor(slice_1, 2, 0, -4);  slice_1 = None
        view_3: "bf16[128, 6, 196, 196]" = torch.ops.aten.view.default(slice_2, _shape_param_7);  slice_2 = _shape_param_7 = None
        mul: "bf16[128, 6, 196, 196]" = torch.ops.aten.mul.Tensor(view_3, 0.125)
        convert_element_type: "f32[128, 6, 196, 196]" = torch.ops.prims.convert_element_type.default(mul, torch.float32);  mul = None
        convert_element_type_1: "f32[128, 6, 196, 196]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        mul_1: "f32[128, 6, 196, 196]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1);  convert_element_type_1 = None
        abs_1: "f32[128, 6, 196, 196]" = torch.ops.aten.abs.default(convert_element_type)
        ne: "b8[128, 6, 196, 196]" = torch.ops.aten.ne.Scalar(abs_1, inf);  abs_1 = None
        eq: "b8[128, 6, 196, 196]" = torch.ops.aten.eq.Tensor(convert_element_type, convert_element_type)
        mul_2: "b8[128, 6, 196, 196]" = torch.ops.aten.mul.Tensor(eq, ne);  eq = ne = None
        logical_not: "b8[128, 6, 196, 196]" = torch.ops.aten.logical_not.default(mul_2);  mul_2 = None
        return (bmm, convert_element_type, mul_1, logical_not)



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
