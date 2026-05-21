"""
Standalone repro captured via capture_hook.
Label: torchbench_timm_resnest_train
Pattern hash: 76fa70324d83
Shape hash: fb138def
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
_shapes_config = "(T([32, 128, 1, 1], f32), T([32, 2, 64, 56, 56], f32), S([32, 1, 2, -1]), S([32, -1]), S([32, -1, 1, 1]), S([32, 2, 64, 1, 1]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_6: "f32[32, 128, 1, 1]", view_1: "f32[32, 2, 64, 56, 56]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:27 in forward, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
        reshape_default: "f32[32, 1, 2, 64]" = torch.ops.aten.reshape.default(convolution_6, _shape_param_0);  convolution_6 = _shape_param_0 = None
        permute_default: "f32[32, 2, 1, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:28 in forward, code: x = F.softmax(x, dim=1)
        amax_default: "f32[32, 1, 1, 64]" = torch.ops.aten.amax.default(permute_default, [1], True)
        sub_tensor: "f32[32, 2, 1, 64]" = torch.ops.aten.sub.Tensor(permute_default, amax_default);  permute_default = amax_default = None
        exp_default: "f32[32, 2, 1, 64]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[32, 1, 1, 64]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True)
        div_tensor: "f32[32, 2, 1, 64]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:29 in forward, code: x = x.reshape(batch, -1)
        reshape_default_1: "f32[32, 128]" = torch.ops.aten.reshape.default(div_tensor, _shape_param_1);  div_tensor = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:107 in forward, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        reshape_default_2: "f32[32, 128, 1, 1]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/split_attn.py:109 in forward, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        reshape_default_3: "f32[32, 2, 64, 1, 1]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        mul_tensor: "f32[32, 2, 64, 56, 56]" = torch.ops.aten.mul.Tensor(view_1, reshape_default_3);  view_1 = reshape_default_3 = None
        sum_dim_int_list_1: "f32[32, 64, 56, 56]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1]);  mul_tensor = None
        return sum_dim_int_list_1



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
