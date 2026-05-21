"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_infer
Pattern hash: cb76408a0b60
Shape hash: 70a8a66f
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
_shapes_config = "(T([6272, 3072], f32), S([128, 49, 3072]), S([128, 49, 3, 32, -1]), S([128, 32, 49, 32]), S([4096, 49, 32]), S([128, 32, 32, 49]), S([4096, 32, 49]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_88: "f32[6272, 3072]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        reshape_default: "f32[128, 49, 3072]" = torch.ops.aten.reshape.default(addmm_88, _shape_param_0);  addmm_88 = _shape_param_0 = None
        reshape_default_1: "f32[128, 49, 3, 32, 32]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[3, 128, 32, 49, 32]" = torch.ops.aten.permute.default(reshape_default_1, [2, 0, 3, 1, 4]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        unbind_int = torch.ops.aten.unbind.int(permute_default);  permute_default = None
        getitem: "f32[128, 32, 49, 32]" = unbind_int[0]
        getitem_1: "f32[128, 32, 49, 32]" = unbind_int[1]
        getitem_2: "f32[128, 32, 49, 32]" = unbind_int[2];  unbind_int = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_tensor: "f32[128, 32, 49, 32]" = torch.ops.aten.mul.Tensor(getitem, 0.1767766952966369);  getitem = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_default: "f32[128, 32, 49, 32]" = torch.ops.aten.expand.default(mul_tensor, _shape_param_2);  mul_tensor = _shape_param_2 = None
        clone_default: "f32[128, 32, 49, 32]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_2: "f32[4096, 49, 32]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        permute_default_1: "f32[128, 32, 32, 49]" = torch.ops.aten.permute.default(getitem_1, [0, 1, 3, 2]);  getitem_1 = None
        expand_default_1: "f32[128, 32, 32, 49]" = torch.ops.aten.expand.default(permute_default_1, _shape_param_4);  permute_default_1 = _shape_param_4 = None
        clone_default_1: "f32[128, 32, 32, 49]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_3: "f32[4096, 32, 49]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_5);  clone_default_1 = _shape_param_5 = None
        return (getitem_2, reshape_default_2, reshape_default_3)



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
