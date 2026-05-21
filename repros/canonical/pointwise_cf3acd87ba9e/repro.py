"""
Standalone repro captured via capture_hook.
Label: torchbench_shufflenet_v2_x1_0_infer_000
Pattern hash: cf3acd87ba9e
Shape hash: 3667795d
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
_shapes_config = "(T([232], f16), T([512, 232, 7, 7], f16), T([232], f16), T([232], f16), T([232], f16), T([232], f16), T([512, 232, 7, 7], f16), T([232], f16), T([232], f16), T([232], f16), S([512, 2, 232, 7, 7]), S([512, 464, 7, 7]))"

class Repro(torch.nn.Module):
    def forward(self, arg212_1: "f16[232]", convolution_42: "f16[512, 232, 7, 7]", arg213_1: "f16[232]", arg214_1: "f16[232]", arg215_1: "f16[232]", arg227_1: "f16[232]", convolution_45: "f16[512, 232, 7, 7]", arg228_1: "f16[232]", arg229_1: "f16[232]", arg230_1: "f16[232]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[232]" = torch.ops.prims.convert_element_type.default(arg212_1, torch.float32);  arg212_1 = None
        unsqueeze_default: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, -1);  convert_element_type_default = None
        unsqueeze_default_1: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_default_1);  convolution_42 = unsqueeze_default_1 = None
        convert_element_type_default_1: "f32[232]" = torch.ops.prims.convert_element_type.default(arg213_1, torch.float32);  arg213_1 = None
        add_tensor: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1e-05);  convert_element_type_default_1 = None
        sqrt_default: "f32[232]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f16[232, 1]" = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
        unsqueeze_default_5: "f16[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f16[232, 1]" = torch.ops.aten.unsqueeze.default(arg215_1, -1);  arg215_1 = None
        unsqueeze_default_7: "f16[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[512, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        convert_element_type_default_2: "f16[512, 232, 7, 7]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None
        relu_default: "f16[512, 232, 7, 7]" = torch.ops.aten.relu.default(convert_element_type_default_2);  convert_element_type_default_2 = None
        convert_element_type_default_3: "f32[232]" = torch.ops.prims.convert_element_type.default(arg227_1, torch.float32);  arg227_1 = None
        unsqueeze_default_8: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_3, -1);  convert_element_type_default_3 = None
        unsqueeze_default_9: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1);  unsqueeze_default_8 = None
        sub_tensor_1: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_default_9);  convolution_45 = unsqueeze_default_9 = None
        convert_element_type_default_4: "f32[232]" = torch.ops.prims.convert_element_type.default(arg228_1, torch.float32);  arg228_1 = None
        add_tensor_2: "f32[232]" = torch.ops.aten.add.Tensor(convert_element_type_default_4, 1e-05);  convert_element_type_default_4 = None
        sqrt_default_1: "f32[232]" = torch.ops.aten.sqrt.default(add_tensor_2);  add_tensor_2 = None
        reciprocal_default_1: "f32[232]" = torch.ops.aten.reciprocal.default(sqrt_default_1);  sqrt_default_1 = None
        mul_tensor_3: "f32[232]" = torch.ops.aten.mul.Tensor(reciprocal_default_1, 1);  reciprocal_default_1 = None
        unsqueeze_default_10: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, -1);  mul_tensor_3 = None
        unsqueeze_default_11: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, -1);  unsqueeze_default_10 = None
        mul_tensor_4: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_11);  sub_tensor_1 = unsqueeze_default_11 = None
        unsqueeze_default_12: "f16[232, 1]" = torch.ops.aten.unsqueeze.default(arg229_1, -1);  arg229_1 = None
        unsqueeze_default_13: "f16[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, -1);  unsqueeze_default_12 = None
        mul_tensor_5: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_4, unsqueeze_default_13);  mul_tensor_4 = unsqueeze_default_13 = None
        unsqueeze_default_14: "f16[232, 1]" = torch.ops.aten.unsqueeze.default(arg230_1, -1);  arg230_1 = None
        unsqueeze_default_15: "f16[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, -1);  unsqueeze_default_14 = None
        add_tensor_3: "f32[512, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_5, unsqueeze_default_15);  mul_tensor_5 = unsqueeze_default_15 = None
        convert_element_type_default_5: "f16[512, 232, 7, 7]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float16);  add_tensor_3 = None
        relu_default_1: "f16[512, 232, 7, 7]" = torch.ops.aten.relu.default(convert_element_type_default_5);  convert_element_type_default_5 = None
        cat_default: "f16[512, 464, 7, 7]" = torch.ops.aten.cat.default([relu_default, relu_default_1], 1);  relu_default = relu_default_1 = None
        view_default: "f16[512, 2, 232, 7, 7]" = torch.ops.aten.view.default(cat_default, _shape_param_0);  cat_default = _shape_param_0 = None
        permute_default: "f16[512, 232, 2, 7, 7]" = torch.ops.aten.permute.default(view_default, [0, 2, 1, 3, 4]);  view_default = None
        clone_default: "f16[512, 232, 2, 7, 7]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_1: "f16[512, 464, 7, 7]" = torch.ops.aten.view.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        split_tensor = torch.ops.aten.split.Tensor(view_default_1, 232, 1);  view_default_1 = None
        getitem: "f16[512, 232, 7, 7]" = split_tensor[0]
        getitem_1: "f16[512, 232, 7, 7]" = split_tensor[1];  split_tensor = None
        return (getitem_1, getitem)



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
