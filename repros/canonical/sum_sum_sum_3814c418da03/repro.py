"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train
Pattern hash: 3814c418da03
Shape hash: 690e88a9
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
_shapes_config = "(T([6272, 1024], f32), T([1024], f32), T([128, 49, 1024], f32), T([128, 49, 1], f32), T([128, 49, 1024], f32), T([128, 1, 1, 1], b8), S([128, 49, 1024]), S([128, 7, 7, 1024]), S([128, 1, 7, 1, 7, 1024]), S([128, 7, 7, 1024]), S([128, 49, 1024]), S([6272, 1024]), S([1024]))"

class Repro(torch.nn.Module):
    def forward(self, mm_15: "f32[6272, 1024]", primals_342: "f32[1024]", mul_230: "f32[128, 49, 1024]", div_74: "f32[128, 49, 1]", view_689: "f32[128, 49, 1024]", lt_42: "b8[128, 1, 1, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default: "f32[128, 49, 1024]" = torch.ops.aten.reshape.default(mm_15, _shape_param_0);  mm_15 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_tensor: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(reshape_default, primals_342);  primals_342 = None
        mul_tensor_1: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, 1024)
        sum_dim_int_list: "f32[128, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_230);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_230, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[128, 49, 1024]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[128, 49, 1024]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(div_74, sub_tensor_1);  div_74 = sub_tensor_1 = None
        mul_tensor_5: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(reshape_default, mul_230);  mul_230 = None
        sum_dim_int_list_2: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(reshape_default, [0, 1]);  reshape_default = None
        add_tensor: "f32[128, 49, 1024]" = torch.ops.aten.add.Tensor(view_689, mul_tensor_4);  view_689 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        reshape_default_1: "f32[128, 7, 7, 1024]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_default: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_42, torch.float32);  lt_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_tensor: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.9043478220701218);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_tensor_6: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(reshape_default_1, div_tensor);  reshape_default_1 = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        reshape_default_2: "f32[128, 1, 7, 1, 7, 1024]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_2);  mul_tensor_6 = _shape_param_2 = None
        permute_default: "f32[128, 1, 1, 7, 7, 1024]" = torch.ops.aten.permute.default(reshape_default_2, [0, 1, 3, 2, 4, 5]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        reshape_default_3: "f32[128, 7, 7, 1024]" = torch.ops.aten.reshape.default(permute_default, _shape_param_3);  permute_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        reshape_default_4: "f32[128, 49, 1024]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_4);  reshape_default_3 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        reshape_default_5: "f32[6272, 1024]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_5);  reshape_default_4 = _shape_param_5 = None
        permute_default_1: "f32[1024, 6272]" = torch.ops.aten.permute.default(reshape_default_5, [1, 0])
        sum_dim_int_list_4: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(reshape_default_5, [0], True);  reshape_default_5 = None
        reshape_default_6: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, _shape_param_6);  sum_dim_int_list_4 = _shape_param_6 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, permute_default_1, reshape_default_6)



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
