"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train
Pattern hash: 4d7a27f102ca
Shape hash: 814a9920
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
_shapes_config = "(T([8192, 4, 49, 49], f32), T([49, 49], i64, gen=Index(169)), T([32768, 49, 49], f32), T([8192, 4, 49, 49], f32, stride=(9728, 2432, 49, 1)), T([49, 49], i64, gen=Index(169)), S([2401, 4]), S([8192, 4, 49, 49]), S([2401, 4]), S([32768, 49, 49]))"

class Repro(torch.nn.Module):
    def forward(self, fma_22: "f32[8192, 4, 49, 49]", primals_26: "i64[49, 49]", bmm_141: "f32[32768, 49, 49]", div: "f32[8192, 4, 49, 49]", primals_11: "i64[49, 49]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_dim_int_list: "f32[1, 4, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_22, [0], True);  fma_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim: "f32[4, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list, 0);  sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default: "f32[49, 49, 4]" = torch.ops.aten.permute.default(squeeze_dim, [1, 2, 0]);  squeeze_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default: "f32[2401, 4]" = torch.ops.aten.reshape.default(permute_default, _shape_param_0);  permute_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        full_default: "f32[169, 4]" = torch.ops.aten.full.default([169, 4], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_1: "i64[2401]" = torch.ops.aten.reshape.default(primals_26, [-1]);  primals_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default: "f32[169, 4]" = torch.ops.aten.index_put.default(full_default, [reshape_default_1], reshape_default, True);  reshape_default_1 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        reshape_default_2: "f32[8192, 4, 49, 49]" = torch.ops.aten.reshape.default(bmm_141, _shape_param_1);  bmm_141 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        mul_tensor: "f32[8192, 4, 49, 49]" = torch.ops.aten.mul.Tensor(reshape_default_2, div);  reshape_default_2 = None
        sum_dim_int_list_1: "f32[8192, 4, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True)
        neg_default: "f32[8192, 4, 49, 49]" = torch.ops.aten.neg.default(div);  div = None
        fma_default: "f32[8192, 4, 49, 49]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list_1, mul_tensor);  neg_default = sum_dim_int_list_1 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_dim_int_list_2: "f32[1, 4, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_default, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_1: "f32[4, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_2, 0);  sum_dim_int_list_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_1: "f32[49, 49, 4]" = torch.ops.aten.permute.default(squeeze_dim_1, [1, 2, 0]);  squeeze_dim_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_3: "f32[2401, 4]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_2);  permute_default_1 = _shape_param_2 = None
        reshape_default_4: "i64[2401]" = torch.ops.aten.reshape.default(primals_11, [-1]);  primals_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_1: "f32[169, 4]" = torch.ops.aten.index_put.default(full_default, [reshape_default_4], reshape_default_3, True);  full_default = reshape_default_4 = reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        reshape_default_5: "f32[32768, 49, 49]" = torch.ops.aten.reshape.default(fma_default, _shape_param_3);  fma_default = _shape_param_3 = None
        return (index_put_default, index_put_default_1, reshape_default_5)



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
