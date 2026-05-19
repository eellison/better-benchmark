"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-3-5-linux.aws.a100_graph7
Pattern hash: 28b89da299c5
Shape hash: 0ee05f72
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_68: "f16[512, 768]", bmm_22: "f16[12, 512, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        view_default: "f16[1, 512, 768]" = torch.ops.aten.view.default(addmm_68, _shape_param_0);  addmm_68 = _shape_param_0 = None
        view_default_1: "f16[1, 512, 12, 64]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        permute_default: "f16[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3]);  view_default_1 = None
        view_default_2: "f16[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_22, _shape_param_2);  bmm_22 = _shape_param_2 = None
        convert_element_type_default: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        mul_tensor: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1);  convert_element_type_default = None
        amax_default: "f32[1, 12, 512, 1]" = torch.ops.aten.amax.default(mul_tensor, [-1], True)
        sub_tensor: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor, amax_default);  mul_tensor = amax_default = None
        mul_tensor_1: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(sub_tensor, 0.125);  sub_tensor = None
        exp_default: "f32[1, 12, 512, 512]" = torch.ops.aten.exp.default(mul_tensor_1);  mul_tensor_1 = None
        sum_dim_int_list: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[1, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_1: "f16[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.float16);  div_tensor = None
        native_dropout_default = torch.ops.aten.native_dropout.default(convert_element_type_default_1, 0.1, True);  convert_element_type_default_1 = None
        getitem: "f16[1, 12, 512, 512]" = native_dropout_default[0]
        getitem_1: "b8[1, 12, 512, 512]" = native_dropout_default[1];  native_dropout_default = None
        expand_default: "f16[1, 12, 512, 512]" = torch.ops.aten.expand.default(getitem, _shape_param_3);  getitem = _shape_param_3 = None
        view_default_3: "f16[12, 512, 512]" = torch.ops.aten.view.default(expand_default, _shape_param_4);  expand_default = _shape_param_4 = None
        expand_default_1: "f16[1, 12, 512, 64]" = torch.ops.aten.expand.default(permute_default, _shape_param_5);  permute_default = _shape_param_5 = None
        view_default_4: "f16[12, 512, 64]" = torch.ops.aten.view.default(expand_default_1, _shape_param_6);  expand_default_1 = _shape_param_6 = None
        return (view_default_3, view_default_4, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([12, 512, 512], dtype=torch.float16, device='cuda'),
    [1, 512, 768],  # _shape_param_0
    [1, 512, -1, 64],  # _shape_param_1
    [1, 12, 512, 512],  # _shape_param_2
    [1, 12, 512, 512],  # _shape_param_3
    [12, 512, 512],  # _shape_param_4
    [1, 12, 512, 64],  # _shape_param_5
    [12, 512, 64],  # _shape_param_6
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
