"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_infer
Pattern hash: e98948176b90
Shape hash: 84e25831
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
_shapes_config = "(T([25088, 512], f32), T([128, 14, 14, 512], f32), T([512], f32), T([512], f32), S([512, 49, 512]), S([-1, 7, 7, 512]), S([-1, 2, 2, 7, 7, 512]), S([-1, 14, 14, 512]), S([128, -1, 512]), S([25088, 512]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_81: "f32[25088, 512]", view_547: "f32[128, 14, 14, 512]", arg309_1: "f32[512]", arg310_1: "f32[512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        reshape_default: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(addmm_81, _shape_param_0);  addmm_81 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        reshape_default_1: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        reshape_default_2: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        permute_default: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(reshape_default_2, [0, 1, 3, 2, 4, 5]);  reshape_default_2 = None
        clone_default: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_3: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        add_tensor: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_547, reshape_default_3);  view_547 = reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        reshape_default_4: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_4);  add_tensor = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default_4, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 196, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 196, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(reshape_default_4, getitem_1);  reshape_default_4 = getitem_1 = None
        add_tensor_1: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, arg309_1);  mul_tensor = arg309_1 = None
        add_tensor_2: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg310_1);  mul_tensor_1 = arg310_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        reshape_default_5: "f32[25088, 512]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_5);  add_tensor_2 = _shape_param_5 = None
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
