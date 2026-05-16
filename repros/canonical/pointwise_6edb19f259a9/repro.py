"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_training
Pattern hash: 6edb19f259a9
Shape hash: d47042a4
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_153: "f32[32, 14, 14, 1]", view_573: "f32[32, 14, 14, 512]", getitem_154: "f32[32, 14, 14, 1]", primals_316: "f32[512]", primals_317: "f32[512]", fmod_8: "i64[14]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, primals_319: "f32[1536, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_tensor: "f32[32, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_153, 1e-05);  getitem_153 = None
        rsqrt_default: "f32[32, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[32, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_573, getitem_154);  view_573 = getitem_154 = None
        mul_tensor: "f32[32, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_316);  mul_tensor = primals_316 = None
        add_tensor_1: "f32[32, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_317);  mul_tensor_1 = primals_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_tensor: "f32[32, 14, 14, 512]" = torch.ops.aten.index.Tensor(add_tensor_1, [None, fmod_8]);  add_tensor_1 = None
        index_tensor_1: "f32[32, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_tensor, [None, None, fmod_8]);  index_tensor = fmod_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        reshape_default: "f32[32, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(index_tensor_1, _shape_param_0);  index_tensor_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_default: "f32[32, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(reshape_default, [0, 1, 3, 2, 4, 5]);  reshape_default = None
        clone_default: "f32[32, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[128, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        reshape_default_2: "f32[128, 49, 512]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        reshape_default_3: "f32[6272, 512]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_1: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_319, [1, 0]);  primals_319 = None
        return (reshape_default_3, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([32, 14, 14, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 14, 14, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 14, 14, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [14], dtype=torch.int64, device='cuda'),
    [32, 2, 7, 2, 7, 512],  # _shape_param_0
    [-1, 7, 7, 512],  # _shape_param_1
    [-1, 49, 512],  # _shape_param_2
    [6272, 512],  # _shape_param_3
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
