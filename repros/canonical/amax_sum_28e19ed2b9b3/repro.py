"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-4-5-linux.aws.a100_graph10
Pattern hash: 28e19ed2b9b3
Shape hash: 75526631
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([256, 1024], bf16), T([16, 256, 256], bf16), T([1, 1, 256, 256], bf16), S([1, 256, 1024]), S([1, -1, 16, 64]), S([16, 256, 64]), S([1, 16, 256, 256]), S([16, 256, 256]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_68: "bf16[256, 1024]", bmm_22: "bf16[16, 256, 256]", where: "bf16[1, 1, 256, 256]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view_default: "bf16[1, 256, 1024]" = torch.ops.aten.view.default(addmm_68, _shape_param_0);  addmm_68 = _shape_param_0 = None
        view_default_1: "bf16[1, 256, 16, 64]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        permute_default: "bf16[1, 16, 256, 64]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3]);  view_default_1 = None
        view_default_2: "bf16[16, 256, 64]" = torch.ops.aten.view.default(permute_default, _shape_param_2);  permute_default = _shape_param_2 = None
        view_default_3: "bf16[1, 16, 256, 256]" = torch.ops.aten.view.default(bmm_22, _shape_param_3);  bmm_22 = _shape_param_3 = None
        add_tensor: "bf16[1, 16, 256, 256]" = torch.ops.aten.add.Tensor(view_default_3, where);  view_default_3 = where = None
        view_default_4: "bf16[16, 256, 256]" = torch.ops.aten.view.default(add_tensor, _shape_param_4);  add_tensor = _shape_param_4 = None
        convert_element_type_default: "f32[16, 256, 256]" = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32);  view_default_4 = None
        amax_default: "f32[16, 256, 1]" = torch.ops.aten.amax.default(convert_element_type_default, [-1], True)
        sub_tensor: "f32[16, 256, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default);  convert_element_type_default = amax_default = None
        exp_default: "f32[16, 256, 256]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[16, 256, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[16, 256, 256]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_1: "bf16[16, 256, 256]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        return (view_default_2, convert_element_type_default_1)


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
