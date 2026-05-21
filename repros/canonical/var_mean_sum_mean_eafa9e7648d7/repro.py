"""
Standalone repro captured via capture_hook.
Label: torchbench_timm_resnest_train
Pattern hash: eafa9e7648d7
Shape hash: 0cd673b0
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
_shapes_config = "(T([32, 1024, 14, 14], f32), T([1024], f32), T([1024], f32), S([32, 2, 512, 14, 14]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_22: "f32[32, 1024, 14, 14]", primals_129: "f32[1024]", primals_130: "f32[1024]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:91 in forward, code: x = self.bn0(x)
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 1024, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 1024, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_1);  convolution_22 = getitem_1 = None
        mul_tensor: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_129, -1);  primals_129 = None
        unsqueeze_default_1: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_130, -1);  primals_130 = None
        unsqueeze_default_3: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:93 in forward, code: x = self.act0(x)
        relu_default: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:97 in forward, code: x = x.reshape((B, self.radix, RC // self.radix, H, W))
        reshape_default: "f32[32, 2, 512, 14, 14]" = torch.ops.aten.reshape.default(relu_default, _shape_param_0);  relu_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:98 in forward, code: x_gap = x.sum(dim=1)
        sum_dim_int_list: "f32[32, 512, 14, 14]" = torch.ops.aten.sum.dim_IntList(reshape_default, [1]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:101 in forward, code: x_gap = x_gap.mean((2, 3), keepdim=True)
        mean_dim: "f32[32, 512, 1, 1]" = torch.ops.aten.mean.dim(sum_dim_int_list, [2, 3], True);  sum_dim_int_list = None
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
