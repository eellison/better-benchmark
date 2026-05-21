"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train
Pattern hash: ecfcc0e2eca7
Shape hash: 26d0807f
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
_shapes_config = "(T([2048, 8, 49, 49], f32), T([49, 49], i64, gen=Index(169)), T([2048, 8, 49, 49], f32), T([49, 49], i64, gen=Index(169)), S([2401, 8]), S([2401, 8]))"

class Repro(torch.nn.Module):
    def forward(self, fma_20: "f32[2048, 8, 49, 49]", primals_58: "i64[49, 49]", fma_21: "f32[2048, 8, 49, 49]", primals_43: "i64[49, 49]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_dim_int_list: "f32[1, 8, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_20, [0], True);  fma_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim: "f32[8, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list, 0);  sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default: "f32[49, 49, 8]" = torch.ops.aten.permute.default(squeeze_dim, [1, 2, 0]);  squeeze_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default: "f32[2401, 8]" = torch.ops.aten.reshape.default(permute_default, _shape_param_0);  permute_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        full_default: "f32[169, 8]" = torch.ops.aten.full.default([169, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_1: "i64[2401]" = torch.ops.aten.reshape.default(primals_58, [-1]);  primals_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default: "f32[169, 8]" = torch.ops.aten.index_put.default(full_default, [reshape_default_1], reshape_default, True);  reshape_default_1 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_dim_int_list_1: "f32[1, 8, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_21, [0], True);  fma_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_1: "f32[8, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_1, 0);  sum_dim_int_list_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_1: "f32[49, 49, 8]" = torch.ops.aten.permute.default(squeeze_dim_1, [1, 2, 0]);  squeeze_dim_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_2: "f32[2401, 8]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_1);  permute_default_1 = _shape_param_1 = None
        reshape_default_3: "i64[2401]" = torch.ops.aten.reshape.default(primals_43, [-1]);  primals_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_1: "f32[169, 8]" = torch.ops.aten.index_put.default(full_default, [reshape_default_3], reshape_default_2, True);  full_default = reshape_default_3 = reshape_default_2 = None
        return (index_put_default, index_put_default_1)



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
