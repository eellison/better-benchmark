"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_inference
Pattern hash: 89991ab608ac
Shape hash: d0fac59f
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, add_148: "f32[128, 16, 240]", getitem_104: "f32[128, 16, 1]", getitem_103: "f32[128, 16, 1]", arg261_1: "f32[240]", arg262_1: "f32[240]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:264 in forward, code: x = self.norm(x)
        sub_tensor: "f32[128, 16, 240]" = torch.ops.aten.sub.Tensor(add_148, getitem_104);  add_148 = getitem_104 = None
        add_tensor: "f32[128, 16, 1]" = torch.ops.aten.add.Tensor(getitem_103, 1e-05);  getitem_103 = None
        rsqrt_default: "f32[128, 16, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[128, 16, 240]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[128, 16, 240]" = torch.ops.aten.mul.Tensor(mul_tensor, arg261_1);  mul_tensor = arg261_1 = None
        add_tensor_1: "f32[128, 16, 240]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg262_1);  mul_tensor_1 = arg262_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:268 in forward, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        reshape_default: "f32[32, 4, 16, 240]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_0);  add_tensor_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:269 in forward, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
        permute_default: "f32[32, 240, 16, 4]" = torch.ops.aten.permute.default(reshape_default, [0, 3, 2, 1]);  reshape_default = None
        clone_default: "f32[32, 240, 16, 4]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[30720, 4, 2, 2]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:271 in forward, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        permute_default_1: "f32[30720, 2, 4, 2]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None
        clone_default_1: "f32[30720, 2, 4, 2]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_2: "f32[32, 240, 8, 8]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_2);  clone_default_1 = _shape_param_2 = None
        return reshape_default_2


def _default_make_inputs():
    return [
    torch.randn([128, 16, 240], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 16, 1], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    torch.randn([240], dtype=torch.float32, device='cuda'),
    [32, 4, 16, -1],  # _shape_param_0
    [30720, 4, 2, 2],  # _shape_param_1
    [32, 240, 8, 8],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
