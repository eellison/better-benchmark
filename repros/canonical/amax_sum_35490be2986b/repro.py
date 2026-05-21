"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_infer_000
Pattern hash: 35490be2986b
Shape hash: dea17632
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
_shapes_config = "(T([8192, 49, 49], f32), T([49, 49], i64, gen=Index(169)), T([169, 16], f32), T([4, 49, 49], f32), S([512, 16, 49, 49]), S([49, 49, -1]), S([-1, 4, 16, 49, 49]), S([-1, 16, 49, 49]), S([512, 16, 49, 49]), S([8192, 49, 49]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_42: "f32[8192, 49, 49]", arg321_1: "i64[49, 49]", arg320_1: "f32[169, 16]", arg317_1: "f32[4, 49, 49]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        view_default: "f32[512, 16, 49, 49]" = torch.ops.aten.view.default(bmm_42, _shape_param_0);  bmm_42 = _shape_param_0 = None
        view_default_1: "i64[2401]" = torch.ops.aten.view.default(arg321_1, [-1]);  arg321_1 = None
        index_tensor: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg320_1, [view_default_1]);  arg320_1 = view_default_1 = None
        view_default_2: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_tensor, _shape_param_1);  index_tensor = _shape_param_1 = None
        permute_default: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_default_2, [2, 0, 1]);  view_default_2 = None
        clone_default: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        unsqueeze_default: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_default, 0);  clone_default = None
        add_tensor: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_default, unsqueeze_default);  view_default = unsqueeze_default = None
        view_default_3: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_tensor, _shape_param_2);  add_tensor = _shape_param_2 = None
        unsqueeze_default_1: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg317_1, 1);  arg317_1 = None
        unsqueeze_default_2: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 0);  unsqueeze_default_1 = None
        add_tensor_1: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_default_3, unsqueeze_default_2);  view_default_3 = unsqueeze_default_2 = None
        view_default_4: "f32[512, 16, 49, 49]" = torch.ops.aten.view.default(add_tensor_1, _shape_param_3);  add_tensor_1 = _shape_param_3 = None
        amax_default: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(view_default_4, [-1], True)
        sub_tensor: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_default_4, amax_default);  view_default_4 = amax_default = None
        exp_default: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        expand_default: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_tensor, _shape_param_4);  div_tensor = _shape_param_4 = None
        view_default_5: "f32[8192, 49, 49]" = torch.ops.aten.view.default(expand_default, _shape_param_5);  expand_default = _shape_param_5 = None
        return view_default_5



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
