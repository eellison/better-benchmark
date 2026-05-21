"""
Standalone repro captured via capture_hook.
Label: torchbench_shufflenet_v2_x1_0_infer
Pattern hash: 8063a740f2a2
Shape hash: 0dcafd74
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
_shapes_config = "(T([116], f16), T([512, 116, 14, 14], f16), T([116], f16), T([116], f16), T([116], f16), T([512, 116, 14, 14], f16, stride=(45472, 196, 14, 1)), S([512, 2, 116, 14, 14]), S([512, 232, 14, 14]))"

class Repro(torch.nn.Module):
    def forward(self, arg187_1: "f16[116]", convolution_37: "f16[512, 116, 14, 14]", arg188_1: "f16[116]", arg189_1: "f16[116]", arg190_1: "f16[116]", getitem_18: "f16[512, 116, 14, 14]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        convert_element_type_default: "f32[116]" = torch.ops.prims.convert_element_type.default(arg187_1, torch.float32);  arg187_1 = None
        unsqueeze_default: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, -1);  convert_element_type_default = None
        unsqueeze_default_1: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_default_1);  convolution_37 = unsqueeze_default_1 = None
        convert_element_type_default_1: "f32[116]" = torch.ops.prims.convert_element_type.default(arg188_1, torch.float32);  arg188_1 = None
        add_tensor: "f32[116]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1e-05);  convert_element_type_default_1 = None
        sqrt_default: "f32[116]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[116]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[116]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f16[116, 1]" = torch.ops.aten.unsqueeze.default(arg189_1, -1);  arg189_1 = None
        unsqueeze_default_5: "f16[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f16[116, 1]" = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_default_7: "f16[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[512, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        convert_element_type_default_2: "f16[512, 116, 14, 14]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None
        relu_default: "f16[512, 116, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_default_2);  convert_element_type_default_2 = None
        cat_default: "f16[512, 232, 14, 14]" = torch.ops.aten.cat.default([getitem_18, relu_default], 1);  getitem_18 = relu_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        reshape_default: "f16[512, 2, 116, 14, 14]" = torch.ops.aten.reshape.default(cat_default, _shape_param_0);  cat_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_default: "f16[512, 116, 2, 14, 14]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3, 4]);  reshape_default = None
        clone_default: "f16[512, 116, 2, 14, 14]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        reshape_default_1: "f16[512, 232, 14, 14]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        split_tensor = torch.ops.aten.split.Tensor(reshape_default_1, 116, 1);  reshape_default_1 = None
        getitem: "f16[512, 116, 14, 14]" = split_tensor[0]
        getitem_19: "f16[512, 116, 14, 14]" = split_tensor[1];  split_tensor = None
        return (getitem, getitem_19)



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
