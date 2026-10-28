"""
Standalone repro captured via capture_hook.
Label: repros/models/timm/train/swin_base_patch4_window7_224/full_graph_000.py
Pattern hash: b4da8b22c3d0
Shape hash: 32090bb2
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
    def forward(self, arg0_1: "f32[3072]", arg1_1: "f32[3072, 1024]", arg2_1: "bf16[6272, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[3072, 1024]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        permute: "bf16[1024, 3072]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[6272, 3072]" = torch.ops.aten.addmm.default(convert_element_type, arg2_1, permute);  convert_element_type = arg2_1 = permute = None
        view: "bf16[128, 49, 3072]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        view_1: "bf16[128, 49, 3, 32, 32]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute_1: "bf16[3, 128, 32, 49, 32]" = torch.ops.aten.permute.default(view_1, [2, 0, 3, 1, 4]);  view_1 = None
        unbind = torch.ops.aten.unbind.int(permute_1);  permute_1 = None
        getitem: "bf16[128, 32, 49, 32]" = unbind[0]
        getitem_1: "bf16[128, 32, 49, 32]" = unbind[1];  getitem_1 = None
        getitem_2: "bf16[128, 32, 49, 32]" = unbind[2];  unbind = None
        mul: "bf16[128, 32, 49, 32]" = torch.ops.aten.mul.Tensor(getitem, 0.1767766952966369);  getitem = None
        expand: "bf16[128, 32, 49, 32]" = torch.ops.aten.expand.default(mul, _shape_param_2);  mul = _shape_param_2 = None
        clone: "bf16[128, 32, 49, 32]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_2: "bf16[4096, 49, 32]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        expand_1: "bf16[128, 32, 49, 32]" = torch.ops.aten.expand.default(getitem_2, _shape_param_4);  getitem_2 = _shape_param_4 = None
        clone_1: "bf16[128, 32, 49, 32]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_3: "bf16[4096, 49, 32]" = torch.ops.aten.view.default(clone_1, _shape_param_5);  clone_1 = _shape_param_5 = None
        constant_pad_nd: "bf16[4096, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_3, _shape_param_6);  view_3 = _shape_param_6 = None
        permute_2: "bf16[4096, 32, 49]" = torch.ops.aten.permute.default(view_2, [0, 2, 1])
        return (view_2, constant_pad_nd, permute_2)



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
