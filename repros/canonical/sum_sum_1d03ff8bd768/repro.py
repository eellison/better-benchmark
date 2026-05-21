"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train
Pattern hash: 1d03ff8bd768
Shape hash: f98bd35c
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
_shapes_config = "(T([6272, 1024], f32), T([1024], f32), T([6272, 1024], f32), T([128, 7, 7, 1], f32), T([128, 7, 7, 1], f32), T([128, 7, 7, 1024], f32), S([128, 49, 1024]), S([128, 7, 7, 1024]), S([128, 1, 1, 7, 7, 1024]), S([128, 7, 7, 1024]), S([128, 7, 7, 1024]), S([6272, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, mm_19: "f32[6272, 1024]", primals_334: "f32[1024]", mm_2: "f32[6272, 1024]", getitem_163: "f32[128, 7, 7, 1]", rsqrt_48: "f32[128, 7, 7, 1]", view_696: "f32[128, 7, 7, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        reshape_default: "f32[128, 49, 1024]" = torch.ops.aten.reshape.default(mm_19, _shape_param_0);  mm_19 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        reshape_default_1: "f32[128, 7, 7, 1024]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        reshape_default_2: "f32[128, 1, 1, 7, 7, 1024]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default: "f32[128, 1, 7, 1, 7, 1024]" = torch.ops.aten.permute.default(reshape_default_2, [0, 1, 3, 2, 4, 5]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        reshape_default_3: "f32[128, 7, 7, 1024]" = torch.ops.aten.reshape.default(permute_default, _shape_param_3);  permute_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_tensor: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(reshape_default_3, primals_334);  reshape_default_3 = primals_334 = None
        mul_tensor_1: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, 1024)
        sum_dim_int_list: "f32[128, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        reshape_default_4: "f32[128, 7, 7, 1024]" = torch.ops.aten.reshape.default(mm_2, _shape_param_4);  mm_2 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        sub_tensor: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(reshape_default_4, getitem_163);  reshape_default_4 = getitem_163 = None
        mul_tensor_2: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_48);  sub_tensor = None
        mul_tensor_3: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [3], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  mul_tensor_2 = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[128, 7, 7, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 1024);  rsqrt_48 = None
        mul_tensor_5: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        add_tensor: "f32[128, 7, 7, 1024]" = torch.ops.aten.add.Tensor(view_696, mul_tensor_5);  view_696 = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        reshape_default_5: "f32[6272, 1024]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_5);  add_tensor = _shape_param_5 = None
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
