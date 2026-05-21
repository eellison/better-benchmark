"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train
Pattern hash: 2760f9114894
Shape hash: fa8ccc84
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
_shapes_config = "(T([401408, 128], f32), T([128, 3136, 128], f32), T([128], f32), T([128], f32), S([128, 3136, 128]), S([128, 56, 56, 128]), S([128, 8, 7, 8, 7, 128]), S([-1, 7, 7, 128]), S([-1, 49, 128]), S([401408, 128]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_3: "f32[401408, 128]", view_20: "f32[128, 3136, 128]", primals_20: "f32[128]", primals_21: "f32[128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[128, 3136, 128]" = torch.ops.aten.reshape.default(addmm_3, _shape_param_0);  addmm_3 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        add_tensor: "f32[128, 3136, 128]" = torch.ops.aten.add.Tensor(view_20, reshape_default);  view_20 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        reshape_default_1: "f32[128, 56, 56, 128]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default_1, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 56, 56, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 56, 56, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[128, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[128, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(reshape_default_1, getitem_1);  reshape_default_1 = getitem_1 = None
        mul_tensor: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_20);  mul_tensor = primals_20 = None
        add_tensor_2: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_21);  mul_tensor_1 = primals_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_default: "i64[56]" = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_3: "i64[56]" = torch.ops.aten.add.Tensor(iota_default, 3);  iota_default = None
        fmod_scalar: "i64[56]" = torch.ops.aten.fmod.Scalar(add_tensor_3, 56);  add_tensor_3 = None
        index_tensor: "f32[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(add_tensor_2, [None, fmod_scalar]);  add_tensor_2 = None
        index_tensor_1: "f32[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(index_tensor, [None, None, fmod_scalar]);  index_tensor = fmod_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        reshape_default_2: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.reshape.default(index_tensor_1, _shape_param_2);  index_tensor_1 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        permute_default: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.permute.default(reshape_default_2, [0, 1, 3, 2, 4, 5]);  reshape_default_2 = None
        clone_default: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_3: "f32[8192, 7, 7, 128]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        reshape_default_4: "f32[8192, 49, 128]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_4);  reshape_default_3 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
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
