"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_inference
Pattern hash: 42dbf7c0f66d
Shape hash: e2e77dfa
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_42: "f32[2048, 49, 49]", _shape_param_0, arg321_1: "i64[49, 49]", arg320_1: "f32[169, 16]", _shape_param_1, _shape_param_2, arg317_1: "f32[4, 49, 49]", _shape_param_3, _shape_param_4, _shape_param_5, getitem_157: "f32[128, 16, 49, 32]", _shape_param_6, _shape_param_7):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        reshape_default: "f32[128, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_42, _shape_param_0);  bmm_42 = _shape_param_0 = None

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
        add_tensor: "f32[128, 16, 49, 49]" = torch.ops.aten.add.Tensor(reshape_default, unsqueeze_default);  reshape_default = unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        reshape_default_3: "f32[32, 4, 16, 49, 49]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_2);  add_tensor = _shape_param_2 = None
        unsqueeze_default_1: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg317_1, 1);  arg317_1 = None
        unsqueeze_default_2: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 0);  unsqueeze_default_1 = None
        add_tensor_1: "f32[32, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(reshape_default_3, unsqueeze_default_2);  reshape_default_3 = unsqueeze_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        reshape_default_4: "f32[128, 16, 49, 49]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_3);  add_tensor_1 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        amax_default: "f32[128, 16, 49, 1]" = torch.ops.aten.amax.default(reshape_default_4, [-1], True)
        sub_tensor: "f32[128, 16, 49, 49]" = torch.ops.aten.sub.Tensor(reshape_default_4, amax_default);  reshape_default_4 = amax_default = None
        exp_default: "f32[128, 16, 49, 49]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[128, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[128, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        expand_default: "f32[128, 16, 49, 49]" = torch.ops.aten.expand.default(div_tensor, _shape_param_4);  div_tensor = _shape_param_4 = None
        reshape_default_5: "f32[2048, 49, 49]" = torch.ops.aten.reshape.default(expand_default, _shape_param_5);  expand_default = _shape_param_5 = None
        expand_default_1: "f32[128, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_157, _shape_param_6);  getitem_157 = _shape_param_6 = None
        clone_default_1: "f32[128, 16, 49, 32]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_6: "f32[2048, 49, 32]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_7);  clone_default_1 = _shape_param_7 = None
        return (reshape_default_5, reshape_default_6)


def _default_make_inputs():
    return [
    torch.randn([2048, 49, 49], dtype=torch.float32, device='cuda'),
    [128, 16, 49, 49],  # _shape_param_0
    torch.randint(0, 2, [49, 49], dtype=torch.int64, device='cuda'),
    torch.randn([169, 16], dtype=torch.float32, device='cuda'),
    [49, 49, -1],  # _shape_param_1
    [-1, 4, 16, 49, 49],  # _shape_param_2
    torch.randn([4, 49, 49], dtype=torch.float32, device='cuda'),
    [-1, 16, 49, 49],  # _shape_param_3
    [128, 16, 49, 49],  # _shape_param_4
    [2048, 49, 49],  # _shape_param_5
    torch.randn(9632768, dtype=torch.float32, device='cuda').as_strided([128, 16, 49, 32], [75264, 32, 1536, 1]),  # getitem_157
    [128, 16, 49, 32],  # _shape_param_6
    [2048, 49, 32],  # _shape_param_7
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
