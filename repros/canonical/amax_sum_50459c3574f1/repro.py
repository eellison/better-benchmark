"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-5-5-linux.aws.a100_graph11
Pattern hash: 50459c3574f1
Shape hash: 82d9adf5
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([16, 512, 512], bf16), T([512, 1024], bf16), T([16, 512, 1024], bf16), T([512], i64, gen=Index(1)), S([16, 512, 1, 1, 512]), S([1, 16, 512, 512]), S([512, 1, 1, 16, 64]), S([512, 1, 16, 64]), S([16, 512, 64]), S([16, 512, 1, 1, 1024]), S([1, 16, 512, 1024]), S([1, 16, 1024, 512]), S([1, 16, 512, 1023]), S([16, 512, 512]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_3: "bf16[16, 512, 512]", mm_14: "bf16[512, 1024]", bmm_4: "bf16[16, 512, 1024]", iota_2: "i64[512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # No stacktrace found for following nodes
        view_default: "bf16[16, 512, 1, 1, 512]" = torch.ops.aten.view.default(bmm_3, _shape_param_0);  bmm_3 = _shape_param_0 = None
        permute_default: "bf16[1, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_default, [3, 0, 1, 4, 2]);  view_default = None
        view_default_1: "bf16[1, 16, 512, 512]" = torch.ops.aten.view.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None
        unsqueeze_default: "bf16[1, 512, 1024]" = torch.ops.aten.unsqueeze.default(mm_14, 0);  mm_14 = None
        view_default_2: "bf16[512, 1, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_default, _shape_param_2);  unsqueeze_default = _shape_param_2 = None
        permute_default_1: "bf16[512, 1, 16, 64, 1]" = torch.ops.aten.permute.default(view_default_2, [0, 2, 3, 4, 1]);  view_default_2 = None
        view_default_3: "bf16[512, 1, 16, 64]" = torch.ops.aten.view.default(permute_default_1, _shape_param_3);  permute_default_1 = _shape_param_3 = None
        unsqueeze_default_1: "bf16[512, 1, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_default_3, 4);  view_default_3 = None
        permute_default_2: "bf16[1, 1, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_default_1, [4, 1, 2, 3, 0]);  unsqueeze_default_1 = None
        permute_default_3: "bf16[16, 512, 1, 64, 1]" = torch.ops.aten.permute.default(permute_default_2, [2, 4, 1, 3, 0]);  permute_default_2 = None
        view_default_4: "bf16[16, 512, 64]" = torch.ops.aten.view.default(permute_default_3, _shape_param_4);  permute_default_3 = _shape_param_4 = None
        view_default_5: "bf16[16, 512, 1, 1, 1024]" = torch.ops.aten.view.default(bmm_4, _shape_param_5);  bmm_4 = _shape_param_5 = None
        permute_default_4: "bf16[1, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_default_5, [3, 0, 1, 4, 2]);  view_default_5 = None
        view_default_6: "bf16[1, 16, 512, 1024]" = torch.ops.aten.view.default(permute_default_4, _shape_param_6);  permute_default_4 = _shape_param_6 = None
        view_default_7: "bf16[1, 16, 1024, 512]" = torch.ops.aten.view.default(view_default_6, _shape_param_7);  view_default_6 = _shape_param_7 = None
        slice_tensor: "bf16[1, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_default_7, 2, 1, 9223372036854775807);  view_default_7 = None
        view_default_8: "bf16[1, 16, 512, 1023]" = torch.ops.aten.view.default(slice_tensor, _shape_param_8);  slice_tensor = _shape_param_8 = None
        index_tensor: "bf16[1, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_default_8, [None, None, None, iota_2]);  view_default_8 = iota_2 = None
        add_tensor: "bf16[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_default_1, index_tensor);  view_default_1 = index_tensor = None
        add_tensor_1: "bf16[1, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_tensor, 0);  add_tensor = None
        convert_element_type_default: "f32[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32);  add_tensor_1 = None
        mul_tensor: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1);  convert_element_type_default = None
        amax_default: "f32[1, 16, 512, 1]" = torch.ops.aten.amax.default(mul_tensor, [3], True)
        sub_tensor: "f32[1, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor, amax_default);  mul_tensor = amax_default = None
        mul_tensor_1: "f32[1, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_tensor, 0.125);  sub_tensor = None
        exp_default: "f32[1, 16, 512, 512]" = torch.ops.aten.exp.default(mul_tensor_1);  mul_tensor_1 = None
        sum_dim_int_list: "f32[1, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [3], True)
        div_tensor: "f32[1, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_1: "bf16[1, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        unsqueeze_default_2: "bf16[1, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_1, 4);  convert_element_type_default_1 = None
        permute_default_5: "bf16[512, 1, 16, 1, 512]" = torch.ops.aten.permute.default(unsqueeze_default_2, [2, 0, 1, 4, 3]);  unsqueeze_default_2 = None
        permute_default_6: "bf16[16, 512, 512, 1, 1]" = torch.ops.aten.permute.default(permute_default_5, [2, 0, 4, 1, 3]);  permute_default_5 = None
        view_default_9: "bf16[16, 512, 512]" = torch.ops.aten.view.default(permute_default_6, _shape_param_9);  permute_default_6 = _shape_param_9 = None
        return (view_default_4, view_default_9)


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
