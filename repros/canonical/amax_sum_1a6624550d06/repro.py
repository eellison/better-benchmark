"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_infer
Pattern hash: 1a6624550d06
Shape hash: dea17632
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
_shapes_config = "(T([8192, 49, 49], f32), T([49, 49], i64, gen=Index(169)), T([169, 16], f32), T([4, 49, 49], f32), S([512, 16, 49, 49]), S([49, 49, -1]), S([-1, 4, 16, 49, 49]), S([-1, 16, 49, 49]), S([512, 16, 49, 49]), S([8192, 49, 49]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_42: "f32[8192, 49, 49]", arg321_1: "i64[49, 49]", arg320_1: "f32[169, 16]", arg317_1: "f32[4, 49, 49]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        reshape_default: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_42, _shape_param_0);  bmm_42 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_1: "i64[2401]" = torch.ops.aten.reshape.default(arg321_1, [-1]);  arg321_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_tensor: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg320_1, [reshape_default_1]);  arg320_1 = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_2: "f32[49, 49, 16]" = torch.ops.aten.reshape.default(index_tensor, _shape_param_1);  index_tensor = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default: "f32[16, 49, 49]" = torch.ops.aten.permute.default(reshape_default_2, [2, 0, 1]);  reshape_default_2 = None
        clone_default: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_default: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_default, 0);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_tensor: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(reshape_default, unsqueeze_default);  reshape_default = unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        reshape_default_3: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_2);  add_tensor = _shape_param_2 = None
        unsqueeze_default_1: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg317_1, 1);  arg317_1 = None
        unsqueeze_default_2: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 0);  unsqueeze_default_1 = None
        add_tensor_1: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(reshape_default_3, unsqueeze_default_2);  reshape_default_3 = unsqueeze_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        reshape_default_4: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_3);  add_tensor_1 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_default: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(reshape_default_4, [-1], True)
        sub_tensor: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(reshape_default_4, amax_default);  reshape_default_4 = amax_default = None
        exp_default: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_default: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_tensor, _shape_param_4);  div_tensor = _shape_param_4 = None
        reshape_default_5: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_default, _shape_param_5);  expand_default = _shape_param_5 = None
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
