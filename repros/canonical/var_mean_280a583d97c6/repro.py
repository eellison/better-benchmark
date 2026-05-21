"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_infer_005
Pattern hash: 280a583d97c6
Shape hash: 7649872c
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
_shapes_config = "(T([1, 4096, 256], f16), T([256], f16), T([256], f16), S([1, 4096, 2, 256]), S([1, 4096, 512]), S([4096, 256]), S([4096, 256]), S([4096, 256]))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f16[1, 4096, 256]", arg1_1: "f16[256]", arg2_1: "f16[256]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        unsqueeze_default: "f16[1, 4096, 1, 256]" = torch.ops.aten.unsqueeze.default(arg0_1, 2);  arg0_1 = None
        expand_default: "f16[1, 4096, 2, 256]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_0);  unsqueeze_default = _shape_param_0 = None
        clone_default: "f16[1, 4096, 2, 256]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default: "f16[1, 4096, 512]" = torch.ops.aten.view.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        split_tensor = torch.ops.aten.split.Tensor(view_default, 256, -1);  view_default = None
        getitem: "f16[1, 4096, 256]" = split_tensor[0]
        getitem_1: "f16[1, 4096, 256]" = split_tensor[1];  split_tensor = None
        clone_default_1: "f16[1, 4096, 256]" = torch.ops.aten.clone.default(getitem, memory_format = torch.contiguous_format);  getitem = None
        convert_element_type_default: "f32[1, 4096, 256]" = torch.ops.prims.convert_element_type.default(clone_default_1, torch.float32);  clone_default_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem_2: "f32[1, 4096, 1]" = var_mean_correction[0]
        getitem_3: "f32[1, 4096, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[1, 4096, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_3);  convert_element_type_default = getitem_3 = None
        add_tensor: "f32[1, 4096, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_default: "f32[1, 4096, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[1, 4096, 256]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 4096, 256]" = torch.ops.aten.mul.Tensor(mul_tensor, arg1_1);  mul_tensor = arg1_1 = None
        add_tensor_1: "f32[1, 4096, 256]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg2_1);  mul_tensor_1 = arg2_1 = None
        convert_element_type_default_1: "f16[1, 4096, 256]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None
        view_default_1: "f16[4096, 256]" = torch.ops.aten.view.default(convert_element_type_default_1, _shape_param_2);  _shape_param_2 = None
        view_default_2: "f16[4096, 256]" = torch.ops.aten.view.default(convert_element_type_default_1, _shape_param_3);  _shape_param_3 = None
        view_default_3: "f16[4096, 256]" = torch.ops.aten.view.default(convert_element_type_default_1, _shape_param_4);  convert_element_type_default_1 = _shape_param_4 = None
        return (view_default_1, view_default_2, view_default_3, getitem_1)



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
