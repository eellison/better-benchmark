"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train
Pattern hash: f2d319fbfac5
Shape hash: c225485b
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
_shapes_config = "(T([401408, 128], f32), T([128], f32), T([128, 3136, 128], f32), T([128, 3136, 1], f32), T([128, 3136, 128], f32), S([128, 3136, 128]), S([128, 56, 56, 128]), S([128, 8, 7, 8, 7, 128]), S([8192, 7, 7, 128]), S([8192, 49, 128]), S([401408, 128]))"

class Repro(torch.nn.Module):
    def forward(self, mm_197: "f32[401408, 128]", primals_14: "f32[128]", mul_5: "f32[128, 3136, 128]", div_121: "f32[128, 3136, 1]", view_1383: "f32[128, 3136, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default: "f32[128, 3136, 128]" = torch.ops.aten.reshape.default(mm_197, _shape_param_0);  mm_197 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_tensor: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(reshape_default, primals_14);  reshape_default = primals_14 = None
        mul_tensor_1: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, 128)
        sum_dim_int_list: "f32[128, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_5);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_5, sum_dim_int_list_1);  mul_5 = sum_dim_int_list_1 = None
        sub_tensor: "f32[128, 3136, 128]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[128, 3136, 128]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(div_121, sub_tensor_1);  div_121 = sub_tensor_1 = None
        add_tensor: "f32[128, 3136, 128]" = torch.ops.aten.add.Tensor(view_1383, mul_tensor_4);  view_1383 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        reshape_default_1: "f32[128, 56, 56, 128]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        reshape_default_2: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.permute.default(reshape_default_2, [0, 1, 3, 2, 4, 5]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_default: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_3: "f32[8192, 7, 7, 128]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        reshape_default_4: "f32[8192, 49, 128]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_4);  reshape_default_3 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        reshape_default_5: "f32[401408, 128]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_5);  reshape_default_4 = _shape_param_5 = None
        return reshape_default_5



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
