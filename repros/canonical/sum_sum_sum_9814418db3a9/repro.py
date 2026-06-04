"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train
Pattern hash: 9814418db3a9
Shape hash: d4777b22
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([25088, 512], f32), T([512], f32), T([128, 196, 512], f32), T([128, 196, 1], f32), T([128, 196, 512], f32), T([128, 1, 1, 1], b8), T([14], i64, gen=Index(14)), S([128, 196, 512]), S([128, 14, 14, 512]), S([128, 2, 7, 2, 7, 512]), S([512, 7, 7, 512]), S([512, 49, 512]), S([25088, 512]), S([512]))"

class Repro(torch.nn.Module):
    def forward(self, mm_153: "f32[25088, 512]", primals_93: "f32[512]", mul_58: "f32[128, 196, 512]", div_109: "f32[128, 196, 1]", view_1219: "f32[128, 196, 512]", lt_8: "b8[128, 1, 1, 1]", fmod_8: "i64[14]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(mm_153, _shape_param_0);  mm_153 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_tensor: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(reshape_default, primals_93);  primals_93 = None
        mul_tensor_1: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, 512)
        sum_dim_int_list: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_58);  mul_tensor = None
        sum_dim_int_list_1: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_58, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_109, sub_tensor_1);  div_109 = sub_tensor_1 = None
        mul_tensor_5: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(reshape_default, mul_58);  mul_58 = None
        sum_dim_int_list_2: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_3: "f32[512]" = torch.ops.aten.sum.dim_IntList(reshape_default, [0, 1]);  reshape_default = None
        add_tensor: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_1219, mul_tensor_4);  view_1219 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        reshape_default_1: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_default: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_8, torch.float32);  lt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_tensor: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_default, 0.9782608672976494);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_tensor_6: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(reshape_default_1, div_tensor);  reshape_default_1 = div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_tensor: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(mul_tensor_6, [None, None, fmod_8]);  mul_tensor_6 = None
        index_tensor_1: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_tensor, [None, fmod_8]);  index_tensor = fmod_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        reshape_default_2: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(index_tensor_1, _shape_param_2);  index_tensor_1 = _shape_param_2 = None
        permute_default: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(reshape_default_2, [0, 1, 3, 2, 4, 5]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_default: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_3: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        reshape_default_4: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_4);  reshape_default_3 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        reshape_default_5: "f32[25088, 512]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_5);  reshape_default_4 = _shape_param_5 = None
        permute_default_1: "f32[512, 25088]" = torch.ops.aten.permute.default(reshape_default_5, [1, 0])
        sum_dim_int_list_4: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(reshape_default_5, [0], True);  reshape_default_5 = None
        reshape_default_6: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, _shape_param_6);  sum_dim_int_list_4 = _shape_param_6 = None
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
