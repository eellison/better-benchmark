"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_train
Pattern hash: 36804d150a9a
Shape hash: 429ad2ec
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([131072, 144], f32), T([144], f32), T([512, 256, 144], f32), T([512, 256, 1], f32), T([512, 256, 144], f32), T([144, 144], f32), S([512, 256, 144]), S([131072, 144]))"

class Repro(torch.nn.Module):
    def forward(self, mm_68: "f32[131072, 144]", primals_111: "f32[144]", mul_121: "f32[512, 256, 144]", div_54: "f32[512, 256, 1]", add_315: "f32[512, 256, 144]", primals_109: "f32[144, 144]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default: "f32[512, 256, 144]" = torch.ops.aten.reshape.default(mm_68, _shape_param_0);  mm_68 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        mul_tensor: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(reshape_default, primals_111);  reshape_default = primals_111 = None
        mul_tensor_1: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(mul_tensor, 144)
        sum_dim_int_list: "f32[512, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_121);  mul_tensor = None
        sum_dim_int_list_1: "f32[512, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(mul_121, sum_dim_int_list_1);  mul_121 = sum_dim_int_list_1 = None
        sub_tensor: "f32[512, 256, 144]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[512, 256, 144]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[512, 256, 144]" = torch.ops.aten.mul.Tensor(div_54, sub_tensor_1);  div_54 = sub_tensor_1 = None
        add_tensor: "f32[512, 256, 144]" = torch.ops.aten.add.Tensor(add_315, mul_tensor_4);  add_315 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        reshape_default_1: "f32[131072, 144]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None
        permute_default: "f32[144, 144]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
        permute_default_1: "f32[144, 144]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_1, permute_default_1)


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
