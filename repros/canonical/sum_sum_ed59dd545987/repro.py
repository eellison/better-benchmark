"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_training
Pattern hash: ed59dd545987
Shape hash: 2306385d
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_193: "f32[100352, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, fmod_2: "i64[56]", primals_20: "f32[128]", mul_10: "f32[32, 56, 56, 128]", div_120: "f32[32, 56, 56, 1]", view_1358: "f32[32, 56, 56, 128]", _shape_param_4, _shape_param_5, primals_18: "f32[128, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        reshape_default: "f32[2048, 49, 128]" = torch.ops.aten.reshape.default(mm_193, _shape_param_0);  mm_193 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        reshape_default_1: "f32[2048, 7, 7, 128]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        reshape_default_2: "f32[32, 8, 8, 7, 7, 128]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default: "f32[32, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(reshape_default_2, [0, 1, 3, 2, 4, 5]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_default: "f32[32, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_3: "f32[32, 56, 56, 128]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_tensor: "f32[32, 56, 56, 128]" = torch.ops.aten.index.Tensor(reshape_default_3, [None, None, fmod_2]);  reshape_default_3 = None
        index_tensor_1: "f32[32, 56, 56, 128]" = torch.ops.aten.index.Tensor(index_tensor, [None, fmod_2]);  index_tensor = fmod_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_tensor: "f32[32, 56, 56, 128]" = torch.ops.aten.mul.Tensor(index_tensor_1, primals_20);  index_tensor_1 = primals_20 = None
        mul_tensor_1: "f32[32, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, 128)
        sum_dim_int_list: "f32[32, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True)
        mul_tensor_2: "f32[32, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_10);  mul_tensor = None
        sum_dim_int_list_1: "f32[32, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [3], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[32, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_10, sum_dim_int_list_1);  mul_10 = sum_dim_int_list_1 = None
        sub_tensor: "f32[32, 56, 56, 128]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[32, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[32, 56, 56, 128]" = torch.ops.aten.mul.Tensor(div_120, sub_tensor_1);  div_120 = sub_tensor_1 = None
        add_tensor: "f32[32, 56, 56, 128]" = torch.ops.aten.add.Tensor(view_1358, mul_tensor_4);  view_1358 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        reshape_default_4: "f32[32, 3136, 128]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_4);  add_tensor = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default_5: "f32[100352, 128]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_5);  reshape_default_4 = _shape_param_5 = None
        permute_default_1: "f32[512, 128]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        permute_default_2: "f32[128, 512]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None
        return (reshape_default_5, permute_default_2)


def _default_make_inputs():
    return [
    torch.randn([100352, 128], dtype=torch.float32, device='cuda'),
    [2048, 49, 128],  # _shape_param_0
    [2048, 7, 7, 128],  # _shape_param_1
    [32, 8, 8, 7, 7, 128],  # _shape_param_2
    [32, 56, 56, 128],  # _shape_param_3
    torch.randint(0, 2, [56], dtype=torch.int64, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 56, 56, 128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 56, 56, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 56, 56, 128], dtype=torch.float32, device='cuda'),
    [32, 3136, 128],  # _shape_param_4
    [100352, 128],  # _shape_param_5
    torch.randn([128, 512], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
