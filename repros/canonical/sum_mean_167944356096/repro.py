"""
Standalone repro captured via capture_hook.
Label: torchbench_timm_resnest_infer
Pattern hash: 167944356096
Shape hash: 804d3d8e
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
_shapes_config = "(T([1024], f16), T([32, 1024, 14, 14], f16), T([1024], f16), T([1024], f16), T([1024], f16), S([32, 2, 512, 14, 14]))"

class Repro(torch.nn.Module):
    def forward(self, arg106_1: "f16[1024]", convolution_22: "f16[32, 1024, 14, 14]", arg107_1: "f16[1024]", arg108_1: "f16[1024]", arg109_1: "f16[1024]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:91 in forward, code: x = self.bn0(x)
        convert_element_type_default: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg106_1, torch.float32);  arg106_1 = None
        unsqueeze_default: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, -1);  convert_element_type_default = None
        unsqueeze_default_1: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_default_1);  convolution_22 = unsqueeze_default_1 = None
        convert_element_type_default_1: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg107_1, torch.float32);  arg107_1 = None
        add_tensor: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1e-05);  convert_element_type_default_1 = None
        sqrt_default: "f32[1024]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg108_1, -1);  arg108_1 = None
        unsqueeze_default_5: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_default_7: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        convert_element_type_default_2: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:93 in forward, code: x = self.act0(x)
        relu_default: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_default_2);  convert_element_type_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:97 in forward, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
        reshape_default: "f16[32, 2, 512, 14, 14]" = torch.ops.aten.reshape.default(relu_default, _shape_param_0);  relu_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:98 in forward, code: x_gap = x.sum(dim=1)
        sum_dim_int_list: "f16[32, 512, 14, 14]" = torch.ops.aten.sum.dim_IntList(reshape_default, [1]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:101 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_dim: "f16[32, 512, 1, 1]" = torch.ops.aten.mean.dim(sum_dim_int_list, [2, 3], True);  sum_dim_int_list = None
        return mean_dim



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
