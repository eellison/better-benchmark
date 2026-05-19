"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf_cuda_h100-4-5-linux.aws.h100_graph40
Pattern hash: 3501511f0050
Shape hash: a0e287b8
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
    def forward(self, addmm_2: "f16[128, 1024]", bmm: "f16[16, 128, 128]", arg9_1: "f32[1, 1, 128, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view_default: "f16[1, 128, 1024]" = torch.ops.aten.view.default(addmm_2, _shape_param_0);  addmm_2 = _shape_param_0 = None
        view_default_1: "f16[1, 128, 16, 64]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        permute_default: "f16[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3]);  view_default_1 = None
        view_default_2: "f16[16, 128, 64]" = torch.ops.aten.view.default(permute_default, _shape_param_2);  permute_default = _shape_param_2 = None
        view_default_3: "f16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_3);  bmm = _shape_param_3 = None
        add_tensor: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_default_3, arg9_1);  view_default_3 = arg9_1 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_default: "f32[1, 16, 128, 128]" = torch.ops.aten.maximum.default(add_tensor, full_default);  add_tensor = full_default = None
        view_default_4: "f32[16, 128, 128]" = torch.ops.aten.view.default(maximum_default, _shape_param_4);  maximum_default = _shape_param_4 = None
        amax_default: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_default_4, [-1], True)
        sub_tensor: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_default_4, amax_default);  view_default_4 = amax_default = None
        exp_default: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        native_dropout_default = torch.ops.aten.native_dropout.default(div_tensor, 0.1, True);  div_tensor = None
        getitem: "f32[16, 128, 128]" = native_dropout_default[0]
        getitem_1: "b8[16, 128, 128]" = native_dropout_default[1];  native_dropout_default = None
        convert_element_type_default: "f16[16, 128, 128]" = torch.ops.prims.convert_element_type.default(getitem, torch.float16);  getitem = None
        return (view_default_2, convert_element_type_default, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([128, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([16, 128, 128], dtype=torch.float16, device='cuda'),
    torch.randn([1, 1, 128, 128], dtype=torch.float32, device='cuda'),
    [1, 128, 1024],  # _shape_param_0
    [1, 128, -1, 64],  # _shape_param_1
    [16, 128, 64],  # _shape_param_2
    [1, 16, 128, 128],  # _shape_param_3
    [16, 128, 128],  # _shape_param_4
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
