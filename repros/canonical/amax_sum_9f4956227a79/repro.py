"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_infer_000
Pattern hash: 9f4956227a79
Shape hash: 687c0369
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
_shapes_config = "(T([4096, 49, 49], f32), T([49, 49], i64, gen=Index(169)), T([169, 32], f32), S([128, 32, 49, 49]), S([49, 49, -1]), S([128, 32, 49, 49]), S([4096, 49, 49]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_46: "f32[4096, 49, 49]", arg352_1: "i64[49, 49]", arg351_1: "f32[169, 32]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f32[128, 32, 49, 49]" = torch.ops.aten.view.default(bmm_46, _shape_param_0);  bmm_46 = _shape_param_0 = None
        view_default_1: "i64[2401]" = torch.ops.aten.view.default(arg352_1, [-1]);  arg352_1 = None
        index_tensor: "f32[2401, 32]" = torch.ops.aten.index.Tensor(arg351_1, [view_default_1]);  arg351_1 = view_default_1 = None
        view_default_2: "f32[49, 49, 32]" = torch.ops.aten.view.default(index_tensor, _shape_param_1);  index_tensor = _shape_param_1 = None
        permute_default: "f32[32, 49, 49]" = torch.ops.aten.permute.default(view_default_2, [2, 0, 1]);  view_default_2 = None
        clone_default: "f32[32, 49, 49]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        unsqueeze_default: "f32[1, 32, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_default, 0);  clone_default = None
        add_tensor: "f32[128, 32, 49, 49]" = torch.ops.aten.add.Tensor(view_default, unsqueeze_default);  view_default = unsqueeze_default = None
        amax_default: "f32[128, 32, 49, 1]" = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor: "f32[128, 32, 49, 49]" = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  add_tensor = amax_default = None
        exp_default: "f32[128, 32, 49, 49]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[128, 32, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[128, 32, 49, 49]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        expand_default: "f32[128, 32, 49, 49]" = torch.ops.aten.expand.default(div_tensor, _shape_param_2);  div_tensor = _shape_param_2 = None
        view_default_3: "f32[4096, 49, 49]" = torch.ops.aten.view.default(expand_default, _shape_param_3);  expand_default = _shape_param_3 = None
        return view_default_3



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
