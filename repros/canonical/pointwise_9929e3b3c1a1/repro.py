"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_inference
Pattern hash: 9929e3b3c1a1
Shape hash: 6d182c60
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_44: "f32[6304, 2304]", _shape_param_0, _shape_param_1, arg210_1: "i64[197, 197]", arg209_1: "f32[732, 12]", _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        reshape_default: "f32[32, 197, 2304]" = torch.ops.aten.reshape.default(addmm_44, _shape_param_0);  addmm_44 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:219 in forward, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        reshape_default_1: "f32[32, 197, 3, 12, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[3, 32, 12, 197, 64]" = torch.ops.aten.permute.default(reshape_default_1, [2, 0, 3, 1, 4]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:220 in forward, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        unbind_int = torch.ops.aten.unbind.int(permute_default);  permute_default = None
        getitem: "f32[32, 12, 197, 64]" = unbind_int[0]
        getitem_1: "f32[32, 12, 197, 64]" = unbind_int[1]
        getitem_2: "f32[32, 12, 197, 64]" = unbind_int[2];  unbind_int = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default_2: "i64[38809]" = torch.ops.aten.reshape.default(arg210_1, [-1]);  arg210_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_tensor: "f32[38809, 12]" = torch.ops.aten.index.Tensor(arg209_1, [reshape_default_2]);  arg209_1 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default_3: "f32[197, 197, 12]" = torch.ops.aten.reshape.default(index_tensor, _shape_param_2);  index_tensor = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_1: "f32[12, 197, 197]" = torch.ops.aten.permute.default(reshape_default_3, [2, 0, 1]);  reshape_default_3 = None
        clone_default: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_default: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_default, 0);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        constant_pad_nd_default: "f32[1, 12, 197, 200]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_default, [0, 3], 0.0);  unsqueeze_default = None
        slice_tensor: "f32[1, 12, 197, 197]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default, -1, 0, 197);  constant_pad_nd_default = None
        expand_default: "f32[32, 12, 197, 197]" = torch.ops.aten.expand.default(slice_tensor, _shape_param_3);  slice_tensor = _shape_param_3 = None
        return (expand_default, getitem, getitem_1, getitem_2)


def _default_make_inputs():
    return [
    torch.randn([6304, 2304], dtype=torch.float32, device='cuda'),
    [32, 197, 2304],  # _shape_param_0
    [32, 197, 3, 12, -1],  # _shape_param_1
    torch.randint(0, 2, [197, 197], dtype=torch.int64, device='cuda'),
    torch.randn([732, 12], dtype=torch.float32, device='cuda'),
    [197, 197, -1],  # _shape_param_2
    [32, 12, 197, 197],  # _shape_param_3
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
