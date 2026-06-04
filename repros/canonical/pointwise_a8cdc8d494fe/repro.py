"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_infer
Pattern hash: a8cdc8d494fe
Shape hash: 4c5ebe7f
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([197, 197], i64, gen=Index(732)), T([732, 12], f32), S([197, 197, -1]), S([128, 12, 197, 197]))"

class Repro(torch.nn.Module):
    def forward(self, arg210_1: "i64[197, 197]", arg209_1: "f32[732, 12]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default: "i64[38809]" = torch.ops.aten.reshape.default(arg210_1, [-1]);  arg210_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:191 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_tensor: "f32[38809, 12]" = torch.ops.aten.index.Tensor(arg209_1, [reshape_default]);  arg209_1 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:192 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(
        reshape_default_1: "f32[197, 197, 12]" = torch.ops.aten.reshape.default(index_tensor, _shape_param_0);  index_tensor = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:195 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default: "f32[12, 197, 197]" = torch.ops.aten.permute.default(reshape_default_1, [2, 0, 1]);  reshape_default_1 = None
        clone_default: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:196 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_default: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_default, 0);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:231 in forward, code: x = F.scaled_dot_product_attention(
        constant_pad_nd_default: "f32[1, 12, 197, 200]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_default, [0, 3], 0.0);  unsqueeze_default = None
        slice_tensor: "f32[1, 12, 197, 197]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default, -1, 0, 197);  constant_pad_nd_default = None
        expand_default: "f32[128, 12, 197, 197]" = torch.ops.aten.expand.default(slice_tensor, _shape_param_1);  slice_tensor = _shape_param_1 = None
        return expand_default

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
